from collections import OrderedDict
import time
import sys
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from megatron import get_args
from megatron import get_timers
from megatron import mpu
from megatron import print_rank_0
from megatron.checkpointing import load_checkpoint
from megatron.checkpointing import save_checkpoint
from megatron.training import setup_model_and_optimizer
from megatron.training import train_step
from megatron.training import training_log
from megatron.utils import reduce_losses
from megatron import get_t5_tokenizer
from megatron.mpu import vocab_parallel_cross_entropy as cross_entropy
from megatron.model.search_strategy import SampleOrGreedySearch, BeamSearch
from tasks.openqa.e2eqa.eval_utils import exact_match_score, metric_max_over_ground_truths
from megatron.mpu.initialize import get_new_index_ready, get_new_chkpt_ready, get_gloo_comm_group


NEW_INDEX_READY = None
NEW_CHKPT_READY = None


def process_batch(batch):
    query_uid = batch['query_uid'].cuda()
    query_ids_bert = batch['query_ids_bert'].cuda()
    query_types = batch['query_types'].cuda()
    query_mask_bert = (batch['query_mask_bert'] < 0.5).cuda()
    query_ids_t5 = batch['query_ids_t5'].cuda()
    query_ids_t5_len = batch['query_ids_t5_len'].cuda()
    dec_ids = batch['dec_ids'].cuda()
    labels = batch['labels'].cuda()
    loss_mask = batch['loss_mask'].cuda()
    reference = batch['reference']

    return query_uid, query_ids_bert, query_types, query_mask_bert, \
           query_ids_t5, query_ids_t5_len, dec_ids, labels, loss_mask, reference


class CustomDataLoader(DataLoader):
    def __init__(self, dataset, eval=False, **kwargs):
        if kwargs.get('collate_fn', None) is None:
            kwargs['collate_fn'] = self._collate_fn
        self.eval = eval
        super().__init__(dataset, **kwargs)

    def _collate_fn(self, batch_data):
        tensorized = OrderedDict()
        for d in batch_data:
            for k, v in d.items():
                tensorized.setdefault(k, []).append(v)
        assert len(tensorized) == 10

        tensorized['query_uid'] = torch.LongTensor(tensorized['query_uid'])
        tensorized['query_ids_bert'] = torch.LongTensor(tensorized['query_ids_bert'])
        tensorized['query_types'] = torch.LongTensor(tensorized['query_types'])
        tensorized['query_mask_bert'] = torch.LongTensor(tensorized['query_mask_bert'])
        tensorized['query_ids_t5'] = torch.LongTensor(tensorized['query_ids_t5'])
        tensorized['query_ids_t5_len'] = torch.LongTensor(tensorized['query_ids_t5_len'])
        tensorized['dec_ids'] = torch.LongTensor(tensorized['dec_ids'])
        tensorized['labels'] = torch.LongTensor(tensorized['labels'])
        tensorized['loss_mask'] = torch.FloatTensor(tensorized['loss_mask'])
        # The final key is the reference, which is already appended.
        return tensorized


# Most of this code is implemented based on the open-source REALM codebase
def get_loss_and_retriever_utility(lm_logits, topk_log_probs, labels, loss_mask, eos_id):
    """
    This function computes loss in a stable manner, and also computes retriever utility
    """
    # Converting the tensors datatype to float
    lm_logits = lm_logits.float()
    topk_log_probs = topk_log_probs.float()

    topk = lm_logits.shape[1]
    # [B, K, L, V]
    lm_log_probs = F.log_softmax(lm_logits, dim=-1)

    # Converting the loss mask to bool tensor and inverting it.
    # and replacing the -1 in labels with 0
    labels = labels.masked_fill(~loss_mask.to(torch.bool), 0)

    # labels: [B, L] -> tiled_labels: [B, K, L]
    tiled_labels = torch.repeat_interleave(labels.unsqueeze(1), topk, dim=1)

    # [B, K, L] -> [B, K, L, 1]
    tiled_labels = tiled_labels.unsqueeze(-1)

    # [B, K, L, 1]
    gold_log_probs = torch.gather(lm_log_probs, dim=-1, index=tiled_labels)

    # [B, K, L, 1] -> [B, K, L]
    gold_log_probs = gold_log_probs.squeeze(-1)

    # [B, K, L]
    joint_gold_log_probs = topk_log_probs.unsqueeze(-1) + gold_log_probs

    # [B, L] -> [B, K, L]
    marginal_gold_log_probs = torch.logsumexp(joint_gold_log_probs, dim=1)

    # Applying mask to marginal loss
    lm_loss = -1 * torch.sum(marginal_gold_log_probs * loss_mask) / torch.sum(loss_mask)

    # Retriever Utility is marginal log prob minus the log prob of NULL Block
    # NULL block probability is the last one: gold_log_probs[:, -1, :]
    # [B, K]
    retriever_utility = marginal_gold_log_probs - gold_log_probs[:, -1, :]
    # Mask out the EOS id and sentinel tokens to compute retriever utility.
    # We are masking out the EOS and sentinel tokens because, their loss is very for both with and without retrieved contexts
    # and hence they are not useful to ascertain the utility of the retrieved tokens.
    # The trick used here is that the sentinel tokens have ids higher than the EOS id, so they all will be replaced with 0.
    loss_mask_retriever_utility = loss_mask.masked_fill(labels >= eos_id, 0)
    assert torch.sum(loss_mask_retriever_utility) > 0
    retriever_utility = torch.sum(retriever_utility * loss_mask_retriever_utility) / torch.sum(loss_mask_retriever_utility)

    null_block_lm_loss = -1 * torch.sum(gold_log_probs[:, -1, :] * loss_mask) / torch.sum(loss_mask)

    return lm_loss, retriever_utility, null_block_lm_loss


def _cross_entropy_forward_step(batch, model):
    """Simple forward step with cross-entropy loss."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch generator').start()
    try:
        batch_ = next(batch)
    except BaseException:
        batch_ = batch

    query_uid, query_ids_bert, query_types, query_mask_bert, \
    query_ids_t5, query_ids_t5_len, dec_ids, labels, loss_mask, reference = process_batch(batch_)
    assert torch.all(query_uid < 0), "query uid can't be positive"

    timers('batch generator').stop()

    # Forward model.
    lm_logits, topk_log_probs, lm_logits_one_context = model(query_uid,
                                                             query_ids_bert,
                                                             query_types,
                                                             query_mask_bert,
                                                             query_ids_t5,
                                                             query_ids_t5_len,
                                                             dec_ids)
    lm_logits = lm_logits.float()

    bsize, dec_seq_length, units = lm_logits.shape
    lm_logits = lm_logits.view(bsize * dec_seq_length, units)

    # Cross-entropy loss.
    loss_func = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=0)
    loss_ = loss_func(lm_logits.contiguous().float(), labels.view(-1))
    lm_loss = torch.sum(loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()

    # Retriever loss
    retriever_loss = torch.FloatTensor([0]).cuda()
    if args.update_retriever:
        if args.ret_kldiv:
            retriever_loss = get_kl_div_retriever(lm_logits_one_context,
                                              topk_log_probs,
                                              labels,
                                              loss_mask)
        else:
            t5_tokenizer = get_t5_tokenizer()
            eos_id = t5_tokenizer.eos_token_id
            retriever_loss, weighted_ru, null_block_loss = get_loss_and_retriever_utility(lm_logits_one_context,
                                                                                          topk_log_probs,
                                                                                          labels,
                                                                                          loss_mask,
                                                                                          eos_id)
    net_loss = lm_loss + retriever_loss
    reduced_loss = reduce_losses([lm_loss, retriever_loss])

    return net_loss, {'lm_loss': reduced_loss[0], 'retriever_loss': reduced_loss[1]}


def get_kl_div_retriever(lm_logits, topk_log_probs, labels, loss_mask):

    # Converting the tensors datatype to float
    lm_logits = lm_logits.float()
    topk_log_probs = topk_log_probs.float()

    topk = lm_logits.shape[1]
    # [B, K, L, V]
    lm_log_probs = F.log_softmax(lm_logits, dim=-1)

    # Converting the loss mask to bool tensor and inverting it.
    # and replacing the -1 in labels with 0
    labels = labels.masked_fill(~loss_mask.to(torch.bool), 0)

    # labels: [B, L] -> tiled_labels: [B, K, L]
    tiled_labels = torch.repeat_interleave(labels.unsqueeze(1), topk, dim=1)

    # [B, K, L] -> [B, K, L, 1]
    tiled_labels = tiled_labels.unsqueeze(-1)

    # [B, K, L, 1]
    gold_log_probs = torch.gather(lm_log_probs, dim=-1, index=tiled_labels)

    # [B, K, L, 1] -> [B, K, L]
    gold_log_probs = gold_log_probs.squeeze(-1)

    teacher_log_probs = torch.sum(gold_log_probs * loss_mask.unsqueeze(1), dim=2) / torch.sum(loss_mask.unsqueeze(1), dim=2)
    teacher_probs = torch.softmax(teacher_log_probs, dim=1)

    loss = F.kl_div(topk_log_probs, teacher_probs, reduction='batchmean')
    return loss


def reader_em_score(model, dataloader, topk_retrievals):
    args = get_args()
    tokenizer = get_t5_tokenizer()
    hypothesis_list, reference_list = [], []
    score_list, quid_list = [], []
    total = 0
    overall_score = 0

    model.eval()
    with torch.no_grad():
        for batch in dataloader:

            def generate_text():
                query_uid, query_ids_bert, query_types, query_mask_bert, \
                query_ids_t5, query_ids_t5_len, dec_ids, labels, loss_mask, reference = process_batch(batch)
                assert torch.all(query_uid < 0), "query uid can't be positive"

                if args.beam_size == 1:
                    obj = SampleOrGreedySearch(max_decode_len=args.max_decode_len,
                                               bos_id=tokenizer.bos_token_id,
                                               eos_id=tokenizer.eos_token_id,
                                               sample=False,
                                               topk_evidence=topk_retrievals)
                elif args.beam_size > 1:
                    obj = BeamSearch(max_decode_len=args.max_decode_len,
                                     bos_id=tokenizer.bos_token_id,
                                     eos_id=tokenizer.eos_token_id,
                                     beam_size=args.beam_size,
                                     topk_evidence=topk_retrievals)
                else:
                    raise AssertionError("--beam-size < 1 is not supported for ORQA reader.")

                hypothesis = obj.generate_output(model,
                                                 query_uid,
                                                 query_ids_bert,
                                                 query_types,
                                                 query_mask_bert,
                                                 query_ids_t5,
                                                 query_ids_t5_len)
                return query_uid, reference, hypothesis

            query_uid, reference, hypothesis = generate_text()

            for quid, ref, hyp in zip(query_uid.tolist(), reference, hypothesis):
                hyp_text = tokenizer.decode(hyp)
                score = metric_max_over_ground_truths(exact_match_score, hyp_text, ref)
                overall_score += score
                score_list.append(score)
                quid_list.append(quid)
                hypothesis_list.append(hyp_text)
                reference_list.append(ref)
                total += 1
    model.train()

    if args.async_indexer:
        num_trainers = args.max_training_rank
    else:
        num_trainers = torch.distributed.get_world_size()

    # Aggregating scores from all train workers
    score_tensor = torch.FloatTensor(score_list).cuda()
    tensor_list = [torch.empty_like(score_tensor) for _ in range(num_trainers)]
    torch.distributed.all_gather(tensor_list, score_tensor, group=mpu.get_data_parallel_group())
    all_score_tensor = torch.cat(tensor_list, dim=0).contiguous()

    quid_tensor = torch.LongTensor(quid_list).cuda()
    tensor_list = [torch.empty_like(quid_tensor) for _ in range(num_trainers)]
    torch.distributed.all_gather(tensor_list, quid_tensor, group=mpu.get_data_parallel_group())
    all_quid_tensor = torch.cat(tensor_list, dim=0).contiguous()

    score_dict = {}
    # Storing the quid and scores in a dict, so that duplicates would be overwritten
    for quid, score in zip(all_quid_tensor.tolist(), all_score_tensor.tolist()):
        score_dict[quid] = score

    return {'Exact Match Score': sum(score_dict.values())}, len(score_dict)


def validation_loss(model, dataloader):
    total = 0
    score = 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            query_uid, query_ids_bert, query_types, query_mask_bert, \
            query_ids_t5, query_ids_t5_len, dec_ids, labels, loss_mask, reference = process_batch(batch)

            # Forward model.
            lm_logits, topk_log_probs, _, _ = model(query_uid,
                                                    query_ids_bert,
                                                    query_types,
                                                    query_mask_bert,
                                                    query_ids_t5,
                                                    query_ids_t5_len,
                                                    dec_ids)
            # Calculating LM Loss as is commonly done.
            lm_loss_ = cross_entropy(lm_logits.float(), labels)
            lm_loss = torch.sum(lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()
            total += 1
            score += lm_loss
    model.train()

    unreduced = torch.cuda.FloatTensor([score, total])
    torch.distributed.all_reduce(unreduced,
                                 group=mpu.get_data_parallel_group())

    return {'Validation Loss': unreduced[0]}, unreduced[1]



def accuracy_func_provider(single_dataset_provider, datapath):
    args = get_args()
    dataset = single_dataset_provider(datapath)
    drop_last = False

    dataloader = build_data_loader(dataset,
                                   args.eval_batch_size,
                                   num_workers=args.num_workers,
                                   drop_last=drop_last,
                                   shuffle=False)
    dataloaders = (dataset.dataset_name, dataloader)

    def metrics_func(model, epoch):
        print_rank_0('calculating metrics ...')
        name, dataloader = dataloaders
        output = reader_em_score(model, dataloader, args.topk_retrievals)
        stats_dict, total = output
        format_string = "|total_questions: {}".format(total)
        for k, v in stats_dict.items():
            format_string += "|{} = {:.2f}".format(k, (v * 100) / total)
        print_rank_0("epoch:{}{}".format(epoch, format_string))

    return metrics_func


def build_data_loader(dataset, batch_size, num_workers, drop_last, shuffle=True, rank0sampler=False):
    """Data loader. Note that batch-size is the local (per GPU) batch-size."""

    if rank0sampler:
        if shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
    else:
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
    # Training dataset.
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
           train_dataloader, valid_dataloader, end_of_epoch_callback, end_of_epoch_callback2):
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

    # For async updates
    last_reload_iteration = None
    new_index_recv_handle = None

    # Async Index Update Part
    if args.async_indexer:
        global NEW_INDEX_READY
        NEW_INDEX_READY = get_new_index_ready()

        global NEW_CHKPT_READY
        NEW_CHKPT_READY = get_new_chkpt_ready()

        # Broad this, so that the indexer group can start indexing
        torch.distributed.broadcast(NEW_CHKPT_READY,
                                    src=0,
                                    group=get_gloo_comm_group())

        new_index_recv_handle = torch.distributed.broadcast(NEW_INDEX_READY,
                                                            src=args.max_training_rank,
                                                            group=get_gloo_comm_group(),
                                                            async_op=True)
        last_reload_iteration = iteration


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

            # if enough iterations have gone by to consider reloading the index during EMDR2 training
            if args.async_indexer and iteration >= last_reload_iteration + args.index_reload_interval:
                while True:
                    # if index has been completed, so reload and save checkpoint before proceeding
                    if new_index_recv_handle.is_completed():
                        print_rank_0(">Training Group: Saving model and reloading index")
                        save_checkpoint(iteration, model, optimizer, lr_scheduler)

                        # send handle
                        torch.distributed.broadcast(NEW_CHKPT_READY,
                                                    src=0,
                                                    group=get_gloo_comm_group())

                        print_rank_0("Training Group: Updating MIPS Index")
                        # get model without FP16 and/or TorchDDP wrappers
                        unwrapped_model = model
                        while hasattr(unwrapped_model, 'module'):
                            unwrapped_model = unwrapped_model.module
                        unwrapped_model.evidence_retriever.update_evidence_embedding()
                        print_rank_0("Training Group: MIPS Index Updated")

                        # new recv handle
                        new_index_recv_handle = torch.distributed.broadcast(NEW_INDEX_READY,
                                                                            src=args.max_training_rank,
                                                                            group=get_gloo_comm_group(),
                                                                            async_op=True)
                        print_rank_0("Training Group: Async wait for NEW_INDEX_READY")
                        last_reload_iteration = iteration
                        break

                    # wait for indexer to finish first
                    else:
                        time.sleep(5)

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

            # Evaluation
            if args.eval_interval and iteration % args.eval_interval == 0:
                end_of_epoch_callback(model, iteration)
                end_of_epoch_callback2(model, iteration)

            if args.exit_interval and iteration % args.exit_interval == 0:
                torch.distributed.barrier(mpu.get_data_parallel_group())
                rank = torch.distributed.get_rank()
                print_rank_0('rank: {} | exiting the program at iteration {}'.format(rank, iteration))
                if args.async_indexer:
                    while True:
                        if new_index_recv_handle.is_completed():
                            print_rank_0(">Training Group: Saving model before exiting")
                            save_checkpoint(iteration, model, optimizer, lr_scheduler)
                            sys.exit(0)
                        else:
                            time.sleep(5)
                else:
                    sys.exit(0)

        # Checkpointing at the end of each epoch.
        if args.save:
            save_checkpoint(iteration, model, optimizer, lr_scheduler)

        # Callback at the end of each epoch.
        if end_of_epoch_callback is not None:
            end_of_epoch_callback(model, epoch + 1)
            end_of_epoch_callback2(model, epoch + 1)


def train(train_valid_datasets_provider, model_provider,
          forward_step=_cross_entropy_forward_step,
          end_of_epoch_callback_provider=None,
          end_of_training_callback_provider=None):

    args = get_args()
    timers = get_timers()

    # Train and validation data loaders.
    timers('train/valid/test dataset/dataloder').start()
    if args.epochs > 0:
        train_dataset, valid_dataset = train_valid_datasets_provider()
        train_dataloader, valid_dataloader = _build_train_valid_dataloaders(
            train_dataset, valid_dataset)
    timers('train/valid/test dataset/dataloder').stop()

    # Build calback function.
    timers('callback function').start()
    end_of_epoch_callback = None
    end_of_epoch_callback2 = None
    if args.epochs > 0 and end_of_epoch_callback_provider is not None:
        end_of_epoch_callback = end_of_epoch_callback_provider(args.valid_data)
        end_of_epoch_callback2 = end_of_epoch_callback_provider(args.test_data)
    timers('callback function').stop()

    # Build model, optimizer and learning rate scheduler.
    timers('model and optimizer').start()
    model, optimizer, lr_scheduler = setup_model_and_optimizer(model_provider)
    timers('model and optimizer').stop()

    # If pretrained checkpoint is provided and we have not trained for
    # any iteration (i.e., iteration is zero), then load the pretrained
    # checkpoint.
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

    # Print setup timing.
    print_rank_0('done with setups ...')
    timers.log(['train/valid/test dataset/dataloder', 'callback function',
                'model and optimizer', 'pretrained checkpoint'])
    print_rank_0('training ...')

    # Finetune the model.
    if args.epochs > 0:
        _train(model,
               optimizer,
               lr_scheduler,
               forward_step,
               train_dataloader,
               valid_dataloader,
               end_of_epoch_callback,
               end_of_epoch_callback2)

    # Evaluate after the training step on validation data
    end_of_training_validation_callback = None
    if end_of_training_callback_provider is not None:
        end_of_training_validation_callback = end_of_training_callback_provider(args.valid_data)

    if end_of_training_validation_callback is not None:
        print_rank_0('evaluating on validation data, setting epoch to -1')
        torch.distributed.barrier(mpu.get_data_parallel_group())
        end_of_training_validation_callback(model, epoch=-1,
                                            output_predictions=True)
        torch.distributed.barrier(mpu.get_data_parallel_group())

    # Evaluate after the training step on test data
    if args.test_data is not None:
        end_of_training_test_callback = None
        if end_of_training_callback_provider is not None:
            end_of_training_test_callback = end_of_training_callback_provider(args.test_data)

        if end_of_training_test_callback is not None:
            print_rank_0('evaluating on test data, setting epoch to -1')
            torch.distributed.barrier(mpu.get_data_parallel_group())
            end_of_training_test_callback(model, epoch=-1,
                                          output_predictions=True)
            torch.distributed.barrier(mpu.get_data_parallel_group())

    print_rank_0('done :-)')
