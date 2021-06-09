import os

import torch
import torch.distributed as dist

from megatron import get_args
from megatron import print_rank_0
from megatron.indexer_emdr2 import IndexBuilder
from megatron.checkpointing import get_checkpoint_tracker_filename
from megatron.initialize import init_distributed, _init_autoresume, _set_random_seed, \
    _write_args_to_tensorboard, _initialize_mem_buffs
from megatron.mpu.initialize import set_data_parallel_group, set_model_parallel_group, init_emdr2_groups, \
    get_train_group, get_index_group, get_data_parallel_group, get_new_chkpt_ready, get_new_index_ready, \
    get_gloo_comm_group, get_exit_interval, initialize_mips_group, mips_is_initialized


NEW_INDEX_READY = None
NEW_CHKPT_READY = None
EXIT_INTERVAL = None


def pprint(*args):
    print(*args, flush=True)


def initialize_and_run_async_megatron(allow_no_cuda=False):
    if not allow_no_cuda:
        assert torch.cuda.is_available(), "Megatron required CUDA."

    args = get_args()
    assert args.async_indexer and args.max_training_rank is not None

    init_distributed()
    setup_emdr2_groups_and_vars()
    pprint("finished setting up EMDR2 groups")

    if mips_is_initialized():
        print('MIPS group is already initialized')
    else:
        initialize_mips_group()

    _initialize_mem_buffs()

    # _init_autoresume()
    # pprint("finished setting up autoresume")

    # Random seeds for reproducibility.
    if args.rank == 0:
        pprint('> setting random seeds to {} ...'.format(args.seed))
    _set_random_seed(args.seed)

    # Write arguments to tensorboard.
    _write_args_to_tensorboard()
    pprint('finished writing args to tensorboard')

    torch.distributed.barrier()

    if torch.distributed.get_rank() < args.max_training_rank:
        torch.distributed.barrier(get_data_parallel_group())
        print_rank_0("Trainer Group: All trainers ready.")
        return
    else:
        runner = AsyncIndexBuilder(args.rank)
        torch.distributed.barrier(get_data_parallel_group())
        print_rank_0("Indexer Group: All indexers ready.")
        runner.run_async()


def setup_emdr2_groups_and_vars():
    args = get_args()
    world_size = dist.get_world_size()
    max_training_rank = args.max_training_rank

    # assuming no model parallelism right now
    set_model_parallel_group(dist.new_group([args.rank]))
    init_emdr2_groups(max_training_rank, world_size)

    if args.rank < max_training_rank:
        set_data_parallel_group(get_train_group())
    else:
        set_data_parallel_group(get_index_group())


class AsyncIndexBuilder(IndexBuilder):
    def __init__(self, rank):
        super().__init__(call_load_attributes_func=False)

        self.rank = rank
        args = get_args()
        self.main_builder_idx = args.max_training_rank
        self.exit_handle = None

        # Get the path of the correct model to load
        iteration = 0
        tracker_filename = get_checkpoint_tracker_filename(args.load)
        if os.path.isfile(tracker_filename):
            with open(tracker_filename, 'r') as f:
                iteration = int(f.read().strip())

        if iteration > 0:
            model_load_path = args.load
            key_list = ['retriever/biencoder_model']
        else:
            model_load_path = args.pretrained_dpr_load
            key_list = None

        # Load the context encoder weights
        self.load_attributes(custom_load_path=model_load_path, key_list=key_list)

        global NEW_INDEX_READY
        NEW_INDEX_READY = get_new_index_ready()
        global NEW_CHKPT_READY
        NEW_CHKPT_READY = get_new_chkpt_ready()


    def run_async(self):
        args = get_args()
        global NEW_CHKPT_READY

        # When the indexing starts, wait for the NEW_CHKPT_READY signal from trainer process of rank=0
        dist.broadcast(NEW_CHKPT_READY, 0, group=get_gloo_comm_group())

        while True:
            if self.is_main_builder:
                print("Starting Indexing again!", flush=True)
            self.build_and_save_index()
            self.send_index_ready_signal()
            self.load_attributes(custom_load_path=args.load,
                                 key_list=['retriever/biencoder_model'])

    def send_index_ready_signal(self):
        global NEW_INDEX_READY
        global NEW_CHKPT_READY

        # send handle
        if self.is_main_builder:
            print("indexer group: broadcasting NEW INDEX READY MESSAGE", flush=True)
        dist.broadcast(NEW_INDEX_READY,
                       self.main_builder_idx,
                       group=get_gloo_comm_group(),
                       async_op=True)

        # recv handle
        dist.broadcast(NEW_CHKPT_READY, 0, group=get_gloo_comm_group())
