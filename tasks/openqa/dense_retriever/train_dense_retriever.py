from collections import OrderedDict
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from megatron import get_args, get_tokenizer
from megatron import get_timers
from megatron import mpu
from megatron import print_rank_0
from megatron.checkpointing import load_checkpoint
from megatron.checkpointing import save_checkpoint
from megatron.training import setup_model_and_optimizer
from megatron.training import train_step
from megatron.training import training_log
from megatron.utils import reduce_losses
from megatron.indexer_emdr2 import IndexBuilder
from tasks.openqa.dense_retriever.evaluation.evaluate import OpenRetrievalEvaluator


def get_group_world_size_rank():
    group = mpu.get_data_parallel_group()
    rank = torch.distributed.get_rank(group=group)
    world_size = torch.distributed.get_world_size(group=group)
    return group, rank, world_size


class CustomDataLoader(DataLoader):
    def __init__(self, dataset, eval=False, **kwargs):
        if kwargs.get('collate_fn', None) is None:
            kwargs['collate_fn'] = self._collate_fn
        self.eval = eval
        super().__init__(dataset, **kwargs)

    def _collate_fn(self, batch_data):
        # batch_size = len(batch_data)
        tensorized = OrderedDict()
        for d in batch_data:
            for k, v in d.items():
                tensorized.setdefault(k, []).append(v)

        tensorized['query'] = torch.LongTensor(tensorized['query'])
        tensorized['query_mask'] = torch.LongTensor(tensorized['query_mask'])
        tensorized['query_types'] = torch.LongTensor(tensorized['query_types'])
        tensorized['query_pad_mask'] = torch.LongTensor(tensorized['query_pad_mask'])

        tensorized['context'] = torch.LongTensor(tensorized['context'])
        tensorized['context_mask'] = torch.LongTensor(tensorized['context_mask'])
        tensorized['context_types'] = torch.LongTensor(tensorized['context_types'])
        tensorized['context_pad_mask'] = torch.LongTensor(tensorized['context_pad_mask'])

        if 'neg_context' in tensorized:
            tensorized['neg_context'] = torch.LongTensor(np.concatenate(tensorized['neg_context']))
            tensorized['neg_context_mask'] = torch.LongTensor(np.concatenate(tensorized['neg_context_mask']))
            tensorized['neg_context_types'] = torch.LongTensor(np.concatenate(tensorized['neg_context_types']))

        return tensorized


def process_batch(batch):
    """Process batch and produce inputs for the model."""
    query_tokens = batch['query'].long().cuda()
    query_mask = (batch['query_mask'] < 0.5).cuda()
    query_types = batch['query_types'].long().cuda()
    query_pad_mask = batch['query_pad_mask'].long().cuda()

    context_tokens = batch['context'].long().cuda()
    context_mask = (batch['context_mask'] < 0.5).cuda()
    context_types = batch['context_types'].long().cuda()
    context_pad_mask = batch['context_pad_mask'].long().cuda()

    if 'neg_context' in batch:
        neg_context_tokens = batch['neg_context'].long().cuda()
        neg_context_mask = (batch['neg_context_mask'] < 0.5).cuda()
        neg_context_types = batch['neg_context_types'].long().cuda()
    else:
        neg_context_tokens = None
        neg_context_mask = None
        neg_context_types = None

    reference = batch['reference']

    return query_tokens, query_mask, query_types, query_pad_mask, \
           context_tokens, context_mask, context_types, context_pad_mask, \
           neg_context_tokens, neg_context_mask, neg_context_types, reference


def _cross_entropy_forward_step(batch, model):
    """Simple forward step with cross-entropy loss."""
    args = get_args()
    timers = get_timers()
    tokenizer = get_tokenizer()

    # Get the batch.
    timers('batch generator').start()
    try:
        batch_ = next(batch)
    except BaseException:
        batch_ = batch

    query_tokens, query_mask, query_types, query_pad_mask, \
    context_tokens, context_mask, context_types, context_pad_mask, \
    neg_context_tokens, neg_context_mask, neg_context_types, \
    reference = process_batch(batch_)

    timers('batch generator').stop()
    local_batch_size = query_tokens.shape[0]

    # Text representation of query and context
    query_list, context_list = [], []
    for i in range(local_batch_size):
        query_list.append(tokenizer.decode(query_tokens[i].tolist()))
        context_list.append(tokenizer.decode(context_tokens[i].tolist()))

    if neg_context_tokens is not None:
        context_tokens = torch.cat([context_tokens, neg_context_tokens])
        context_mask = torch.cat([context_mask, neg_context_mask])
        context_types = torch.cat([context_types, neg_context_types])

    # Forward model.
    query_logits, context_logits = model(query_tokens,
                                         query_mask,
                                         query_types,
                                         context_tokens,
                                         context_mask,
                                         context_types)

    group, rank, world_size = get_group_world_size_rank()
    global_batch_size = world_size * local_batch_size  # recall we assert that model_parallel_size == 1

    if world_size > 1:
        input_ = torch.empty_like(context_logits).copy_(context_logits).detach_()
        tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
        tensor_list[rank].copy_(input_)
        torch.distributed.all_gather(tensor_list, input_, group=group)

        # Check if all-gather happens in order
        assert tensor_list[rank].sum().item() == context_logits.sum().item()

        # Preserves the gradient
        tensor_list[rank] = context_logits
        all_context_logits = torch.cat(tensor_list, dim=0).contiguous()

        # Query tensors
        input_ = torch.empty_like(query_logits).copy_(query_logits).detach_()
        tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
        tensor_list[rank].copy_(input_)
        torch.distributed.all_gather(tensor_list, input_, group=group)

        # Check if all-gather happens in order
        assert tensor_list[rank].sum().item() == query_logits.sum().item()

        # Preserves the gradient
        tensor_list[rank] = query_logits
        all_query_logits = torch.cat(tensor_list, dim=0).contiguous()
    else:
        all_query_logits = query_logits
        all_context_logits = context_logits

    retrieval_scores = torch.matmul(all_query_logits,
                                    torch.transpose(all_context_logits, 0, 1))
    # Scaling the retrieval scores
    if args.retriever_score_scaling:
        retrieval_scores = retrieval_scores / math.sqrt(args.hidden_size)

    if args.train_with_neg:
        # if the world size is 3, local batch size is 4, and local context size is 8, what we want is
        # labels = [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19]
        labels = []
        local_context_size = context_tokens.shape[0]
        for i in range(world_size):
            j = i * local_context_size
            labels.extend(list(range(j, j + local_batch_size)))
        labels = torch.LongTensor(labels).cuda()
        assert len(labels) == global_batch_size
    else:
        labels = torch.arange(global_batch_size).long().cuda()

    # Cross-entropy loss.
    softmax_scores = F.log_softmax(retrieval_scores, dim=1)

    loss = F.nll_loss(softmax_scores, labels, reduction='mean')

    max_score, max_idxs = torch.max(softmax_scores, 1)
    correct_predictions_count = (max_idxs == labels).sum().float()

    # Reduce loss for logging.
    reduced_loss = reduce_losses([loss, correct_predictions_count])

    # Loss scaling for correct losses in Supervised Retrieval
    loss = loss * mpu.get_data_parallel_world_size()

    return loss, {'lm loss': reduced_loss[0],
                  'correct_prediction_count': reduced_loss[1]}


def build_data_loader(dataset, batch_size, num_workers, drop_last, shuffle=True):
    """Data loader. Note that batch-size is the local (per GPU) batch-size."""

    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                              num_replicas=world_size,
                                                              rank=rank,
                                                              shuffle=shuffle)
    # Data loader. Note that batch size is the per GPU batch size.
    data_loader = CustomDataLoader(dataset,
                                   batch_size=batch_size,
                                   sampler=sampler,
                                   shuffle=False,
                                   num_workers=num_workers,
                                   drop_last=drop_last,
                                   pin_memory=True)
    return data_loader


def _build_infinite_size_dataloader(dataloader):
    """Build a looped dataloader with infinite size."""

    iterator = dataloader.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = dataloader.__iter__()


def _build_train_valid_dataloaders(train_dataset, valid_dataset):
    """Traing and validation dataloaders."""
    args = get_args()

    print_rank_0('building train and validation dataloaders ...')
    train_dataloader = build_data_loader(train_dataset,
                                         args.batch_size,
                                         args.num_workers,
                                         not args.keep_last)
    # Set the training iterations.
    args.train_iters_per_epoch = len(train_dataloader)
    args.train_iters = args.epochs * args.train_iters_per_epoch

    # Validation dataset. For this dataset, we do not need to set up
    # shuffling so we can just use a simple infinite loop.
    valid_dataloader_ = build_data_loader(valid_dataset,
                                          args.batch_size,
                                          args.num_workers,
                                          not args.keep_last)
    valid_dataloader = _build_infinite_size_dataloader(valid_dataloader_)

    return train_dataloader, valid_dataloader


def _train(model, optimizer, lr_scheduler, forward_step,
           train_dataloader, end_of_epoch_callback):
    """Train the model."""
    args = get_args()
    timers = get_timers()

    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    losses_dict_sum = {}

    # Starting epoch and iteration
    start_epoch = args.iteration // args.train_iters_per_epoch
    start_iteration = args.iteration % args.train_iters_per_epoch
    iteration = args.iteration

    # Memory reporting flag.
    report_memory_flag = True

    # For each remaining epoch
    timers('interval time').start()
    for epoch in range(start_epoch, args.epochs):
        print_rank_0('working on epoch {} ...'.format(epoch + 1))

        # Set the data loader epoch to shuffle the index iterator.
        train_dataloader.sampler.set_epoch(args.seed + epoch)

        # For all the batches in the dataset.
        for iteration_, batch in enumerate(train_dataloader):

            # Ignore the iterations before starting value
            if iteration_ < start_iteration:
                continue
            # Set to zero so the next epoch does not skip any batches.
            start_iteration = 0

            # Train for one step.
            losses_dict, skipped_iter = train_step(forward_step, batch, model,
                                                   optimizer, lr_scheduler)
            iteration += 1

            # Logging.
            report_memory_flag = training_log(losses_dict, losses_dict_sum,
                                              optimizer.param_groups[0]['lr'],
                                              iteration, optimizer.loss_scale,
                                              report_memory_flag, skipped_iter)

            # Checkpointing
            if args.save and args.save_interval and \
                    iteration % args.save_interval == 0:
                save_checkpoint(iteration, model, optimizer, lr_scheduler)

        # Checkpointing at the end of each epoch.
        if args.save:
            save_checkpoint(iteration, model, optimizer, lr_scheduler)

        # Callback at the end of each epoch.
        if end_of_epoch_callback is not None:
            end_of_epoch_callback(model, epoch + 1)


def train(train_valid_datasets_provider, model_provider, forward_step=_cross_entropy_forward_step,
          end_of_epoch_callback_provider=None):

    args = get_args()
    timers = get_timers()

    timers('train/valid/test dataset/dataloder').start()
    if args.epochs > 0:
        train_dataset, valid_dataset = train_valid_datasets_provider()
        train_dataloader, valid_dataloader = _build_train_valid_dataloaders(train_dataset, valid_dataset)
    timers('train/valid/test dataset/dataloder').stop()

    timers('callback function').start()
    end_of_epoch_callback = None
    if end_of_epoch_callback_provider is not None:
        end_of_epoch_callback = end_of_epoch_callback_provider(args.valid_data)
    timers('callback function').stop()

    # Build model, optimizer and learning rate scheduler.
    timers('model and optimizer').start()
    model, optimizer, lr_scheduler = setup_model_and_optimizer(model_provider)
    timers('model and optimizer').stop()

    # If pretrained checkpoint is provided and we have not trained for any iteration (i.e., iteration is zero),
    # then load the pretrained checkpoint.
    timers('pretrained checkpoint').start()
    if args.iteration == 0 and args.pretrained_checkpoint is not None:
        original_load = args.load
        args.load = args.pretrained_checkpoint
        _ = load_checkpoint(model, None, None)
        args.load = original_load
        # This is critical when only model is loaded. We should make sure
        # master parameters are also updated.
        if args.fp16:
            optimizer._model_params_to_master_params()
    timers('pretrained checkpoint').stop()

    print_rank_0('done with setups ...')
    timers.log(['train/valid/test dataset/dataloder', 'callback function', 'model and optimizer',
                'pretrained checkpoint'])
    print_rank_0('training ...')

    # Finetune the model.
    if args.epochs > 0:
        _train(model,
               optimizer,
               lr_scheduler,
               forward_step,
               train_dataloader,
               end_of_epoch_callback)

    del model
    del optimizer
    del lr_scheduler

    build_evidence_index()
    get_retrieval_score()

    print_rank_0('done :-)')


# Build Index
def build_evidence_index():
    index_builder = IndexBuilder()
    index_builder.build_and_save_index()


# Compute retrieval score
def get_retrieval_score():
    args = get_args()
    evaluator = OpenRetrievalEvaluator()
    if args.qa_file_dev is not None:
        evaluator.evaluate(args.qa_file_dev, "DEV")
        torch.distributed.barrier()
    if args.qa_file_test is not None:
        evaluator.evaluate(args.qa_file_test, "TEST")
        torch.distributed.barrier()


def accuracy_func_provider(single_dataset_provider, datapath):
    """Provide function that calculates accuracies."""
    args = get_args()

    # Build dataloaders.
    dataset = single_dataset_provider(datapath)

    drop_last = False
    if mpu.get_data_parallel_world_size() > 1:
        drop_last = True

    dataloader = build_data_loader(dataset,
                                   args.eval_batch_size,
                                   num_workers=args.num_workers,
                                   drop_last=drop_last,
                                   shuffle=False)
    dataloaders = (dataset.dataset_name, dataloader)

    def metrics_func(model, epoch):
        print_rank_0('calculating metrics ...')
        name, dataloader = dataloaders

        output = retrieval_loss(model,
                                dataloader)
        stats_dict, total = output
        format_string = ""

        for k, v in stats_dict.items():
            format_string += "|{} = {:.2f}".format(k, v / total)
        print_rank_0("epoch:{}{}".format(epoch, format_string))

    return metrics_func


def retrieval_loss(model, dataloader):
    args = get_args()
    total = 0
    topk_stats_dict = {'top{}_acc'.format(k): 0 for k in args.report_topk_accuracies}
    stats_dict = dict(rank=0, **topk_stats_dict)

    model.eval()
    with torch.no_grad():
        # For all the batches in the dataset.
        for batch in dataloader:
            # Run the model forward.
            query_tokens, query_mask, query_types, _, \
            context_tokens, context_mask, context_types, _, \
            neg_context_tokens, neg_context_mask, neg_context_types, \
            reference = process_batch(batch)

            query_logits, context_logits = model(query_tokens,
                                                 query_mask,
                                                 query_types,
                                                 torch.cat([context_tokens, neg_context_tokens]),
                                                 torch.cat([context_mask, neg_context_mask]),
                                                 torch.cat([context_types, neg_context_types])
                                                 )

            retrieval_scores = torch.matmul(query_logits,
                                            torch.transpose(context_logits, 0, 1))
            if args.retriever_score_scaling:
                retrieval_scores = retrieval_scores / math.sqrt(args.hidden_size)

            local_batch_size = query_logits.shape[0]
            labels = torch.arange(local_batch_size).long().cuda()

            softmax_scores = F.softmax(retrieval_scores, dim=1)
            sorted_vals, sorted_indices = torch.topk(softmax_scores,
                                                     k=softmax_scores.shape[1],
                                                     sorted=True)

            def topk_accuracy(k):
                return torch.cuda.FloatTensor(
                    [sum([int(labels[i] in sorted_indices[i, :k]) for i in range(local_batch_size)])])

            def get_rank():
                return torch.cuda.FloatTensor(
                    [sum([torch.nonzero(labels[i] == sorted_indices[i])[0][0] for i in range(local_batch_size)])])

            topk_accs = [topk_accuracy(k) for k in args.report_topk_accuracies]
            rank = get_rank()
            losses = reduce_losses([rank, *topk_accs])

            # create stats_dict with retrieval loss and all specified top-k accuracies
            topk_acc_dict = {'top{}_acc'.format(k): v * 100 for k, v in zip(args.report_topk_accuracies, losses[1:])}
            temp_stats_dict = dict(rank=losses[0], **topk_acc_dict)
            for k in stats_dict.keys():
                stats_dict[k] += temp_stats_dict[k]
            total += local_batch_size

    model.train()

    return stats_dict, total
