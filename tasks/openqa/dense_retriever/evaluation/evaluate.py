import os
import shutil
import math
import json
import torch
from megatron import get_args, print_rank_0, get_tokenizer, mpu
from megatron.training import get_model
from megatron.model.dualencoder_model import dualencoder_model_provider
from megatron.checkpointing import load_dualencoder_checkpoint
from megatron.data.orqa_wiki_dataset import get_open_retrieval_wiki_dataset
from tasks.openqa.dense_retriever.evaluation.data import get_qa_dataset, get_one_epoch_qa_dataloader, process_qa_batch
from tasks.openqa.dense_retriever.evaluation.qa_validation import calculate_matches
from megatron.data.emdr2_index import OpenRetreivalDataStore, FaissMIPSIndex


class OpenRetrievalEvaluator(object):
    def __init__(self):
        args = get_args()
        self.embedding_size = args.hidden_size
        self.faiss_use_gpu = args.faiss_use_gpu
        self.evidence_embedder_obj = None
        self.evidence_dataset = None
        self.mips_index = None
        self.eval_dataset = None
        self.get_evidence_dataset()

        # Load query encoder checkpoint
        only_query_model = True
        model = get_model(lambda: dualencoder_model_provider(only_query_model=only_query_model))
        self.model = load_dualencoder_checkpoint(model,
                                                 only_query_model=only_query_model)
        self.model.eval()
        self.faiss_wrapper()

    def get_evidence_embedding(self):
        # This will load the embedding from the embedding path
        self.evidence_embedder_obj = OpenRetreivalDataStore(load_from_path=True)

    def get_evidence_dataset(self):
        self.evidence_dataset = get_open_retrieval_wiki_dataset()

    def faiss_wrapper(self):
        # Initialize FAISS wrapper on local rank = 0 as the evidence embeddings is distributed over all the GPUs in a node
        args = get_args()
        if args.local_rank == 0:
            self.get_evidence_embedding()

            assert self.evidence_embedder_obj is not None
            self.mips_index = FaissMIPSIndex(embed_size=self.embedding_size,
                                             embed_data=self.evidence_embedder_obj,
                                             use_gpu=self.faiss_use_gpu)

        # Wait for the FAISS index to be initialized in all the nodes
        torch.distributed.barrier()

    def generate_query_vectors(self, qa_file, split):
        self.eval_dataset = get_qa_dataset(qa_file, split)
        dataloader = iter(get_one_epoch_qa_dataloader(self.eval_dataset))

        tokenizer = get_tokenizer()
        query_vectors = []
        query_list = []
        reference_list = []

        while True:
            try:
                batch = next(dataloader)
            except (StopIteration, IndexError):
                break

            # batch also has query_tokens and query_pad_data
            query_tokens, query_mask, query_types, \
            query_len, reference = process_qa_batch(batch)

            unwrapped_model = self.model
            while not hasattr(unwrapped_model, 'embed_text'):
                unwrapped_model = unwrapped_model.module

            with torch.no_grad():
                query_logits = unwrapped_model.embed_text(unwrapped_model.query_model,
                                                          query_tokens,
                                                          query_mask,
                                                          query_types)

            for i in range(len(query_tokens)):
                query_list.append(tokenizer.decode(query_tokens[i].tolist()[:query_len[i]]))

            reference_list.extend(reference)
            query_vectors.extend(query_logits.split(1, dim=0))
            if len(query_vectors) % 100 == 0:
                print_rank_0('Encoded queries {}'.format(len(query_vectors) * mpu.get_data_parallel_world_size()))

        query_tensor = torch.cat(query_vectors, dim=0)
        return query_list, query_tensor, reference_list

    def evaluate(self, qa_file, split):
        args = get_args()
        query_list, query_tensor, reference_list = self.generate_query_vectors(qa_file, split)

        local_rank = args.local_rank
        rank = torch.distributed.get_rank()
        device_count = torch.cuda.device_count()
        num_nodes = torch.distributed.get_world_size() // device_count
        node_id = rank // device_count

        for node in range(num_nodes):
            start_rank = node * device_count
            end_rank = (node + 1) * device_count
            ranks_list = list(range(start_rank, end_rank))
            node_group = torch.distributed.new_group(ranks=ranks_list)

            if node_id == node:
                device_start_rank = start_rank
                group = node_group

        input_ = torch.empty_like(query_tensor).copy_(query_tensor).detach_()
        all_query_tensor, allsizes = varsize_gather_nograd(input_, group)
        print_rank_0(all_query_tensor.shape)
        num_rows = len(all_query_tensor)

        if local_rank == 0 and self.mips_index is not None:
            all_query_tensor = all_query_tensor.contiguous()
            distance, topkindex = self.mips_index.search_mips_index(all_query_tensor,
                                                                    top_k=args.topk_retrievals,
                                                                    reconstruct=False)
            distance = torch.from_numpy(distance).cuda()
            topkindex = torch.LongTensor(topkindex).cuda()

        if local_rank != 0:
            distance = torch.empty(len(all_query_tensor), args.topk_retrievals, dtype=torch.float32).cuda()
            topkindex = torch.empty(len(all_query_tensor), args.topk_retrievals, dtype=torch.int64).cuda()

        torch.distributed.broadcast(distance, src=device_start_rank, group=group)
        torch.distributed.broadcast(topkindex, src=device_start_rank, group=group)

        distance = torch.split(distance, allsizes, dim=0)[local_rank]
        topkindex = torch.split(topkindex, allsizes, dim=0)[local_rank]

        del all_query_tensor

        topk_sim_scores = distance #/ math.sqrt(args.hidden_size)

        top_ids_and_scores = []
        for darray, topkarray in zip(topk_sim_scores, topkindex):
            top_ids_and_scores.append((topkarray.tolist(), darray.tolist()))

        passages = self.evidence_dataset.id2text
        match_stats = calculate_matches(passages,
                                        reference_list,
                                        top_ids_and_scores,
                                        workers_num=args.num_workers,
                                        match_type=args.match)

        doc_hits = match_stats.questions_doc_hits
        top_k_hits = torch.FloatTensor(match_stats.top_k_hits).cuda()

        # Accumulating and summing top-k hits scores from all the ranks
        torch.distributed.all_reduce(top_k_hits, torch.distributed.ReduceOp.SUM)

        print_rank_0("{} SET RESULTS".format(split))
        top_k_hits = [v / num_rows for v in top_k_hits]

        for i in args.report_topk_accuracies:
            print_rank_0("top-{}: {:.2f}".format(i, top_k_hits[i-1] * 100))

        if args.save_topk_outputs_path is not None:
            all_data = []
            for i, (q, d, r) in enumerate(zip(query_list, doc_hits, reference_list)):
                ctx_list = []
                for j in range(args.topk_retrievals):

                    ctx = {"id": top_ids_and_scores[i][0][j],
                           "score": top_ids_and_scores[i][1][j],
                           "has_answer": d[j]}
                    ctx_list.append(ctx)
                item = {"question": q,
                        "answers": r,
                        "ctxs": ctx_list}
                all_data.append(item)

            temp_dir_name = os.path.join(args.save_topk_outputs_path,
                                         "_tmp_reranker_{}".format(os.getenv("SLURM_JOBID")))
            save_shard(all_data, temp_dir_name)
            del all_data
            torch.distributed.barrier()

            if mpu.get_data_parallel_rank() == 0:
                file_name = os.path.splitext(os.path.basename(qa_file))[0]
                all_data = merge_shards_and_save(args.save_topk_outputs_path, temp_dir_name, file_name)
                # make sure that every single piece of data was embedded
                assert len(all_data) == len(self.eval_dataset)
                del all_data

        torch.distributed.barrier()
        return


@torch.no_grad()
def varsize_gather_nograd(x, group):
    """gather tensors of different sizes along the first dimension"""

    #determine max size
    size = torch.tensor([x.shape[0]], device=x.device, dtype=torch.int)
    allsizes = [torch.zeros_like(size) for _ in range(mpu.get_data_parallel_world_size())]
    torch.distributed.all_gather(allsizes, size, group=group)
    max_size = max([size.cpu().max() for size in allsizes])

    padded = torch.empty(
                max_size,
                *x.shape[1:],
                dtype=x.dtype,
                device=x.device
            )
    padded[:x.shape[0]] = x
    output = [torch.zeros_like(padded) for _ in range(mpu.get_data_parallel_world_size())]
    torch.distributed.all_gather(output, padded, group=group)

    output = [tensor[:allsizes[k]] for k, tensor in enumerate(output)]
    output = torch.cat(output, dim=0)

    return output, allsizes


def save_shard(data, temp_dir_name):
    """
    Save the block data that was created this in this process
    """
    if not os.path.isdir(temp_dir_name):
        os.makedirs(temp_dir_name, exist_ok=True)

    outpath = os.path.join(temp_dir_name, "rank{}.json".format(mpu.get_data_parallel_rank()))
    with open(outpath, "w") as writer:
        writer.write(json.dumps(data, indent=4) + "\n")


def merge_shards_and_save(output_dir_path, temp_dir_name, file_name):
    """Combine all the shards made using self.save_shard()"""
    shard_names = os.listdir(temp_dir_name)
    all_data = []

    for fname in os.listdir(temp_dir_name):
        shard_size = 0
        old_size = len(all_data)
        fpath = '{}/{}'.format(temp_dir_name, fname)
        with open(fpath, 'r') as f:
            data = json.load(f)
            shard_size = len(data)
            all_data.extend(data)

        assert len(all_data) == old_size + shard_size
        os.remove(fpath)

    # save the consolidated shards
    outpath = os.path.join(output_dir_path, "{}.json".format(file_name))

    with open(outpath, 'w') as writer:
        writer.write(json.dumps(all_data, indent=4) + "\n")

    print("Finished merging {} shards for a total of {} embeds".format(
        len(shard_names), len(all_data)), flush=True)

    shutil.rmtree(temp_dir_name, ignore_errors=True)

    return all_data
