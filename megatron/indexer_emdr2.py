import torch
import torch.distributed as dist
from megatron import get_args
from megatron import mpu
from megatron.checkpointing import load_dualencoder_checkpoint
from megatron.data.samplers import DistributedBatchSampler
from megatron.data.orqa_wiki_dataset import get_open_retrieval_wiki_dataset, get_open_retrieval_batch
from megatron.data.emdr2_index import detach, OpenRetreivalDataStore
from megatron.model.dualencoder_model import dualencoder_model_provider
from megatron.training import get_model
from megatron.mpu.initialize import get_data_parallel_group



def get_one_epoch_dataloader(dataset, batch_size=None):
    args = get_args()

    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    if batch_size is None:
        batch_size = args.batch_size
    global_batch_size = batch_size * world_size
    num_workers = args.num_workers

    sampler = torch.utils.data.SequentialSampler(dataset)
    batch_sampler = DistributedBatchSampler(sampler,
                                            batch_size=global_batch_size,
                                            drop_last=False,
                                            rank=rank,
                                            world_size=world_size)

    return torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       num_workers=num_workers,
                                       pin_memory=True)


class IndexBuilder(object):

    def __init__(self, call_load_attributes_func=True):
        args = get_args()
        self.model = None
        self.dataloader = None
        self.evidence_embedder_obj = None
        self.dataset = get_open_retrieval_wiki_dataset()

        # need to know whether we're using a EMDR2 checkpoint (args.load) or ICT checkpoint
        assert not (args.load and args.ict_load)

        self.log_interval = args.indexer_log_interval
        self.batch_size = args.indexer_batch_size

        if call_load_attributes_func:
            self.load_attributes()
        self.is_main_builder = mpu.get_data_parallel_rank() == 0
        self.num_total_builders = mpu.get_data_parallel_world_size()

    def load_attributes(self, custom_load_path=None, key_list=None):
        only_context_model = True
        model = get_model(lambda: dualencoder_model_provider(only_context_model=only_context_model))
        self.model = load_dualencoder_checkpoint(model,
                                                 only_context_model=only_context_model,
                                                 custom_load_path=custom_load_path,
                                                 key_list=key_list)
        self.model.eval()
        self.dataloader = iter(get_one_epoch_dataloader(self.dataset,
                                                        self.batch_size))
        self.evidence_embedder_obj = OpenRetreivalDataStore(load_from_path=False)
        self.iteration = self.total_processed = 0

    def track_and_report_progress(self, batch_size):
        self.iteration += 1
        self.total_processed += batch_size * self.num_total_builders
        if self.is_main_builder and self.iteration % self.log_interval == 0:
            print('Batch {:10d} | Total {:10d}'.format(self.iteration, self.total_processed), flush=True)

    def build_and_save_index(self):
        unwrapped_model = self.model
        while not hasattr(unwrapped_model, 'embed_text'):
            unwrapped_model = unwrapped_model.module

        while True:
            try:
                # batch also has query_tokens and query_pad_data
                row_id, context_tokens, context_mask, context_types, \
                context_pad_mask = get_open_retrieval_batch(self.dataloader)
            except (StopIteration, IndexError):
                break

            assert context_mask.dtype == torch.bool
            context_logits = unwrapped_model.embed_text(unwrapped_model.context_model,
                                                        context_tokens,
                                                        context_mask,
                                                        context_types)
            context_logits = detach(context_logits)
            row_id = detach(row_id)

            self.evidence_embedder_obj.add_block_data(row_id, context_logits)
            self.track_and_report_progress(batch_size=len(row_id))

        # This process signals to finalize its shard and then synchronize with the other processes
        self.evidence_embedder_obj.save_shard()
        torch.distributed.barrier(get_data_parallel_group())
        del self.model

        # rank 0 process builds the final copy
        if self.is_main_builder:
            self.evidence_embedder_obj.merge_shards_and_save()
            # make sure that every single piece of data was embedded
            assert len(self.evidence_embedder_obj.embed_data) == len(self.dataset)
        self.evidence_embedder_obj.clear()

        # complete building the final copy
        torch.distributed.barrier(get_data_parallel_group())
