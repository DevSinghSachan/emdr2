# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Model and data parallel groups."""
import datetime

import torch

from .utils import ensure_divisibility


# Model parallel group that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None
# MIPS group that current rank belongs to
_MIPS_GROUP = None
# Start rank of a node
_NODE_FIRST_RANK = None

# Trainer and indexer groups for EMDR2 training
_GLOO_COMM_GROUP = None
_TRAIN_GROUP = None
_INDEX_GROUP = None

_NEW_INDEX_READY = None
_NEW_CHKPT_READY = None
_EXIT_INTERVAL = None

# These values enable us to change the mpu sizes on the fly.
_MPU_WORLD_SIZE = None
_MPU_RANK = None


def is_unitialized():
    """Useful for code segments that may be accessed with or without mpu initialization"""
    return _DATA_PARALLEL_GROUP is None


def initialize_model_parallel(model_parallel_size_):
    """
    Initialize model data parallel groups.

    Arguments:
        model_parallel_size: number of GPUs used to parallelize model.

    Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
    use 2 GPUs to parallelize the model. The present function will
    create 4 model parallel groups and 2 data parallel grous as:
        4 model parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 data parallel groups:
            [g0, g2, g4, g6], [g1, g3, g5, g7]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    if torch.distributed.get_rank() == 0:
        print('> initializing model parallel with size {}'.format(
            model_parallel_size_))
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    model_parallel_size = min(model_parallel_size_, world_size)
    ensure_divisibility(world_size, model_parallel_size)
    rank = torch.distributed.get_rank()

    # Build the data parallel groups.
    global _DATA_PARALLEL_GROUP
    assert _DATA_PARALLEL_GROUP is None, \
        'data parallel group is already initialized'
    for i in range(model_parallel_size):
        ranks = range(i, world_size, model_parallel_size)
        group = torch.distributed.new_group(ranks)
        if i == (rank % model_parallel_size):
            _DATA_PARALLEL_GROUP = group

    # Build the model parallel groups.
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, \
        'model parallel group is already initialized'
    for i in range(world_size // model_parallel_size):
        ranks = range(i * model_parallel_size,
                      (i + 1) * model_parallel_size)
        group = torch.distributed.new_group(ranks)
        if i == (rank // model_parallel_size):
            _MODEL_PARALLEL_GROUP = group


def initialize_mips_group():
    from megatron import get_args
    args = get_args()

    if torch.distributed.get_rank() == 0:
        print('> initializing FAISS retriever groups')
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    device_count = torch.cuda.device_count()
    # This assumes that each node contains the same number of GPUs
    if world_size == 1:
        num_nodes = 1
    else:
        num_nodes = world_size // device_count
    node_id = rank // device_count

    # Build the MIPS group
    global _MIPS_GROUP
    assert _MIPS_GROUP is None, '_MIPS_GROUP is already initialized'

    global _NODE_FIRST_RANK
    assert _NODE_FIRST_RANK is None, '_NODE_FIRST_RANK is already initialized'

    for node in range(num_nodes):
        start_rank = node * device_count
        if world_size == 1:
            end_rank = 1
        else:
            # end_rank = (node + 1) * device_count
            # TODO: This is a temporary fix. It will not work in more than 1 node.
            end_rank = args.max_training_rank
        ranks_list = list(range(start_rank, end_rank))
        node_group = torch.distributed.new_group(ranks=ranks_list)

        if node_id == node:
            _NODE_FIRST_RANK = start_rank
            _MIPS_GROUP = node_group


def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    if _MODEL_PARALLEL_GROUP is None or _DATA_PARALLEL_GROUP is None:
        return False
    return True


def mips_is_initialized():
    """Check if mips group is initialized."""
    if _MIPS_GROUP is None or _NODE_FIRST_RANK is None:
        return False
    return True


def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, \
        'model parallel group is not initialized'
    return _MODEL_PARALLEL_GROUP


def set_model_parallel_group(group):
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, \
        'model parallel group has already been initialized'
    _MODEL_PARALLEL_GROUP = group


def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP is not None, \
        'data parallel group is not initialized'
    return _DATA_PARALLEL_GROUP


def set_data_parallel_group(group):
    global _DATA_PARALLEL_GROUP
    assert _DATA_PARALLEL_GROUP is None, \
        'data parallel group has already been initialized'
    _DATA_PARALLEL_GROUP = group


def set_model_parallel_world_size(world_size):
    """Set the model parallel size"""
    global _MPU_WORLD_SIZE
    _MPU_WORLD_SIZE = world_size


def get_model_parallel_world_size():
    """Return world size for the model parallel group."""
    global _MPU_WORLD_SIZE
    if _MPU_WORLD_SIZE is not None:
        return _MPU_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_model_parallel_group())


def set_model_parallel_rank(rank):
    """Set model parallel rank."""
    global _MPU_RANK
    _MPU_RANK = rank


def get_model_parallel_rank():
    """Return my rank for the model parallel group."""
    global _MPU_RANK
    if _MPU_RANK is not None:
        return _MPU_RANK
    return torch.distributed.get_rank(group=get_model_parallel_group())


def get_model_parallel_src_rank():
    """Calculate the global rank corresponding to a local rank zeor
    in the model parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size


def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    return torch.distributed.get_world_size(group=get_data_parallel_group())


def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    return torch.distributed.get_rank(group=get_data_parallel_group())


def destroy_model_parallel():
    """Set the groups to none."""
    global _MODEL_PARALLEL_GROUP
    _MODEL_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None


def get_mips_group():
    global _MIPS_GROUP
    assert _MIPS_GROUP is not None, \
        'MIPS group is not initialized'
    return _MIPS_GROUP


def get_node_first_rank():
    global _NODE_FIRST_RANK
    assert _NODE_FIRST_RANK is not None, \
        'NODE FIRST RANK is not initialized'
    return _NODE_FIRST_RANK


def init_emdr2_groups(max_training_rank, world_size):
    global _GLOO_COMM_GROUP
    _GLOO_COMM_GROUP = torch.distributed.new_group(list(range(world_size)),
                                                   backend="gloo",
                                                   timeout=datetime.timedelta(0, 14400))
    global _TRAIN_GROUP
    _TRAIN_GROUP = torch.distributed.new_group(list(range(max_training_rank)))
    global _INDEX_GROUP
    _INDEX_GROUP = torch.distributed.new_group(list(range(max_training_rank, world_size)))

    # torch.dist.broadcast(_NEW_INDEX_READY, ..., async_op=True) will be
    # asynchronously checked in the trainer group to begin the index reload and save model checkpoint
    # but only after args.reload_index_interval iterations have been done in the training loop
    global _NEW_INDEX_READY
    _NEW_INDEX_READY = torch.zeros(1)

    # torch.dist.broadcast(_NEW_CHKPT_READY, ..., async_op=False) will be
    # used as a synchronization point between the trainer and indexer groups to signal that the new model
    # checkpoint has been saved and so the indexer group can start building the next index
    # and the trainer group will return to the training loop
    global _NEW_CHKPT_READY
    _NEW_CHKPT_READY = torch.zeros(1)

    global _EXIT_INTERVAL
    _EXIT_INTERVAL = torch.zeros(1)


def get_gloo_comm_group():
    global _GLOO_COMM_GROUP
    assert _GLOO_COMM_GROUP is not None
    return _GLOO_COMM_GROUP


def get_train_group():
    global _TRAIN_GROUP
    assert _TRAIN_GROUP is not None
    return _TRAIN_GROUP


def get_index_group():
    global _INDEX_GROUP
    assert _INDEX_GROUP is not None
    return _INDEX_GROUP


def get_new_index_ready():
    global _NEW_INDEX_READY
    assert _NEW_INDEX_READY is not None
    return _NEW_INDEX_READY


def get_new_chkpt_ready():
    global _NEW_CHKPT_READY
    assert _NEW_CHKPT_READY is not None
    return _NEW_CHKPT_READY


def get_exit_interval():
    global _EXIT_INTERVAL
    assert _EXIT_INTERVAL is not None
    return _EXIT_INTERVAL
