import math
import json
import torch
import torch.nn.functional as F
from megatron import get_args, print_rank_0, get_tokenizer
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
        dataloader = get_one_epoch_qa_dataloader(self.eval_dataset)

        tokenizer = get_tokenizer()
        query_vectors = []
        query_list = []
        reference_list = []

        for batch in dataloader:
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
                print_rank_0('Encoded queries {}'.format(len(query_vectors)))

        query_tensor = torch.cat(query_vectors, dim=0)
        print_rank_0('Total encoded queries tensor {}'.format(query_tensor.size()))

        assert query_tensor.size(0) == len(self.eval_dataset)
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
        tensor_list = [torch.empty_like(input_) for _ in range(device_count)]
        torch.distributed.all_gather(tensor_list, query_tensor, group=group)

        if local_rank == 0 and self.mips_index is not None:
            all_query_tensor = torch.cat(tensor_list, dim=0).contiguous()
            print(all_query_tensor.shape)

            distance, topkindex = self.mips_index.search_mips_index(all_query_tensor,
                                                                    top_k=args.topk_retrievals,
                                                                    reconstruct=False)
            distance = torch.from_numpy(distance).cuda()
            topkindex = torch.LongTensor(topkindex).cuda()

        if local_rank != 0:
            distance = torch.empty(device_count * len(query_tensor), args.topk_retrievals, dtype=torch.float32).cuda()
            topkindex = torch.empty(device_count * len(query_tensor), args.topk_retrievals, dtype=torch.int64).cuda()

        torch.distributed.broadcast(distance, src=device_start_rank, group=group)
        torch.distributed.broadcast(topkindex, src=device_start_rank, group=group)

        distance = torch.split(distance, len(query_tensor), dim=0)[local_rank]
        topkindex = torch.split(topkindex, len(query_tensor), dim=0)[local_rank]

        topk_sim_scores = distance / math.sqrt(args.hidden_size)
        topk_probs = F.softmax(topk_sim_scores, dim=1)

        top_ids_and_scores = []
        for darray, topkarray in zip(topk_probs, topkindex):
            top_ids_and_scores.append((topkarray.tolist(), darray.tolist()))

        passages = self.evidence_dataset.id2text
        match_stats = calculate_matches(passages,
                                        reference_list,
                                        top_ids_and_scores,
                                        workers_num=args.num_workers,
                                        match_type=args.match)
        top_k_hits = match_stats.top_k_hits
        doc_hits = match_stats.questions_doc_hits

        print_rank_0("{} SET RESULTS".format(split))
        # print_rank_0("topk-{} documents hits {}".format(args.topk_retrievals, top_k_hits))
        top_k_hits = [v / len(top_ids_and_scores) for v in top_k_hits]
        # print_rank_0("top-k documents hits accuracy {}".format(top_k_hits))

        for i in args.report_topk_accuracies:
            print_rank_0("top-{}: {:.2f}".format(i, top_k_hits[i-1] * 100))

        all_data = []
        for i, (q, d, r) in enumerate(zip(query_list, doc_hits, reference_list)):
            ctx_list = []
            for j in range(args.topk_retrievals):
                # string_template = "question={}|hits={}|answers={}|doc={}|prob={}"
                # string_template = string_template.format(q, d[j], r, passages[top_ids_and_scores[i][0][j]],
                #                                          top_ids_and_scores[i][1][j])
                # print_rank_0(string_template)
                ctx = {"id": top_ids_and_scores[i][0][j],
                       "title": passages[top_ids_and_scores[i][0][j]][1],
                       "text": passages[top_ids_and_scores[i][0][j]][0],
                       "score": top_ids_and_scores[i][1][j],
                       "has_answer": d[j]}
                ctx_list.append(ctx)

            item = {"question": q,
                    "answers": r,
                    "ctxs": ctx_list}

            all_data.append(item)

        if torch.distributed.get_rank() == 0:
            with open(qa_file + ".retout.json", "w") as writer:
                writer.write(json.dumps(all_data, indent=4) + "\n")

        torch.distributed.barrier()

        return
