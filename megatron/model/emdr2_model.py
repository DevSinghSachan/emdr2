import warnings
import math
import numpy as np
import torch
import torch.nn.functional as F
from megatron import get_args, print_rank_0
from megatron.checkpointing import load_dualencoder_checkpoint, load_t5_checkpoint
from megatron.model import T5Model
from megatron.module import MegatronModule
from megatron.model.dualencoder_model import dualencoder_model_provider
from megatron.mpu import get_mips_group, get_node_first_rank
from megatron import get_tokenizer, get_t5_tokenizer
from megatron.tokenizer.tokenizer import vocab_size_with_padding
from megatron.data.mask_creation_utils import make_attention_mask_3d, make_history_mask_3d
from megatron.data.emdr2_index import OpenRetreivalDataStore, DistributedBruteForceIndex
from tools.inverted_title_index import WikiTitleDocMap
from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset
from megatron.data.orqa_wiki_dataset import build_tokens_types_paddings_from_ids as context_bert_format
from megatron.mpu.initialize import get_data_parallel_group


def flatten(ids, types):
    _, _, max_seq_len = ids.shape
    # B x K x T -> B K x T
    ids = ids.reshape(-1, max_seq_len)
    # B x K x T -> B K x T
    types = types.reshape(-1, max_seq_len)
    return ids, types


class EMDR2Model(MegatronModule):
    def __init__(self, evidence_retriever):
        super(EMDR2Model, self).__init__()
        args = get_args()
        self.topk = args.topk_retrievals

        print_rank_0('building Reader for EMDR2 ...')
        t5_tokenizer = get_t5_tokenizer()
        t5_vocab_size = vocab_size_with_padding(t5_tokenizer.vocab_size,
                                                args)
        self.language_model = T5Model(num_tokentypes=2,
                                      parallel_output=True,
                                      vocab_size=t5_vocab_size)
        self._language_model_key = 'encoder/t5_model'

        if args.topk_retrievals > 0:
            bert_tokenizer = get_tokenizer()
            bert_vocab_size = vocab_size_with_padding(bert_tokenizer.vocab_size,
                                                      args)
            print_rank_0('building Retriever for EMDR2 ...')
            self.retriever_model = dualencoder_model_provider(only_context_model=False,
                                                              only_query_model=False,
                                                              vocab_size=bert_vocab_size)
            self._retriever_model_key = 'retriever/biencoder_model'

            self.evidence_retriever = evidence_retriever

        # We have two tokenizers: (1) for BERT as the retriever models are trained using BERT.
        # and (2) for T5 as the reader model uses T5 tokenization (which is also an extension of BERT tokenization in Megatron.)
        self.bert_tokenizer = get_tokenizer()
        self.t5_tokenizer = get_t5_tokenizer()

    def retriever_embedder(self, tokens, mask, types, embedder_type, disable_dropout=False):
        unwrapped_model = self.retriever_model
        while not hasattr(unwrapped_model, 'embed_text'):
            unwrapped_model = unwrapped_model.module

        if embedder_type == "query":
            if disable_dropout:
                unwrapped_model.query_model.eval()
            logits = unwrapped_model.embed_text(unwrapped_model.query_model,
                                                tokens,
                                                mask,
                                                types)
            return logits
        elif embedder_type == "context":
            if disable_dropout:
                unwrapped_model.context_model.eval()
            logits = unwrapped_model.embed_text(unwrapped_model.context_model,
                                                tokens,
                                                mask,
                                                types)
            return logits
        else:
            raise ValueError("Invalid embedder type.")

    def forward(self, query_uid, query_ids_bert, query_types, query_mask_bert,
                query_ids_t5, query_ids_t5_len, dec_ids,
                all_query_context_hidden_states=None, all_query_context_ids_unflat=None,
                topk_log_probs=None):

        args = get_args()
        topk = self.topk
        bsize, max_seq_len = query_ids_bert.shape

        if all_query_context_hidden_states is None:
            # Compute "fresh" query logits
            query_logits = self.retriever_embedder(query_ids_bert,
                                                   query_mask_bert,
                                                   query_types,
                                                   embedder_type="query",
                                                   disable_dropout=args.disable_retriever_dropout)
            if args.no_query_embedder_training:
                query_logits = query_logits.detach()

            # Get topk evidence data for the BERT tokenized query
            with torch.no_grad():
                topk_evidence_data, stale_topk_sim = self.evidence_retriever.get_topk(query_logits.clone().detach())

            with torch.no_grad():
                output = postprocess(query_uid,
                                     query_ids_t5,
                                     query_ids_t5_len,
                                     topk_evidence_data)
                all_context_ids, all_context_types, all_query_extended_context_ids, query_one_context_ids = output

            # reshape the all_context_tokens, all_context_mask, and seq lengths
            all_context_ids, all_context_types = flatten(all_context_ids, all_context_types)
            all_context_mask = make_attention_mask_3d(all_context_ids, all_context_ids)
            all_context_mask = all_context_mask < 0.5

            # Compute "fresh" context logits
            all_context_logits = self.retriever_embedder(all_context_ids,
                                                         all_context_mask,
                                                         all_context_types,
                                                         embedder_type="context",
                                                         disable_dropout=args.disable_retriever_dropout)
            all_context_logits = all_context_logits.reshape(bsize, topk, -1)

            if args.no_context_embedder_training:
                all_context_logits = all_context_logits.detach()

            # [B, 1, dim]
            query_logits = query_logits.unsqueeze(1).float()
            all_context_logits = all_context_logits.float()

            # [B, 1, K]
            topk_sim_scores = torch.bmm(query_logits, all_context_logits.transpose(1, 2))
            if args.retriever_score_scaling:
                topk_sim_scores = topk_sim_scores / math.sqrt(args.hidden_size)

            # [B, 1, K]
            topk_log_probs = F.log_softmax(topk_sim_scores, dim=2)
            # B x 1 x K -> B x K
            topk_log_probs = topk_log_probs.squeeze(1)

            # MASK Handling of topk-augmented encoders
            all_query_context_mask = make_attention_mask_3d(all_query_extended_context_ids, all_query_extended_context_ids)
            all_query_context_mask = all_query_context_mask < 0.5

            # [batch_size x k, args.seq_length_dec, hidden_size]
            all_query_context_hidden_states = self.language_model(encoder_input_ids=all_query_extended_context_ids,
                                                                  decoder_input_ids=dec_ids,
                                                                  encoder_attn_mask=all_query_context_mask,
                                                                  decoder_attn_mask=None,
                                                                  encoder_decoder_attn_mask=None,
                                                                  output_enc_hidden=True)
            # Reshape the query context hidden states
            all_query_context_hidden_states = all_query_context_hidden_states.reshape(bsize,
                                                                                      args.topk_retrievals * args.seq_length,
                                                                                      args.hidden_size)

            all_query_context_ids_unflat = all_query_extended_context_ids.reshape(bsize,
                                                                                  args.topk_retrievals * args.seq_length)

        enc_dec_mask = make_attention_mask_3d(dec_ids, all_query_context_ids_unflat)
        enc_dec_mask = enc_dec_mask < 0.5

        dec_mask = make_attention_mask_3d(dec_ids, dec_ids)
        dec_mask = dec_mask * make_history_mask_3d(dec_ids)
        dec_mask = dec_mask < 0.5

        # Calculate the LM logits
        # When we already have the encoder's hidden states, then all_query_context_ids_unflat is not important
        # Truncating the max sequence length to limit to max-sequence-length
        temp_ids = all_query_context_ids_unflat[:, :args.seq_length]

        lm_logits, _ = self.language_model(temp_ids,
                                           dec_ids,
                                           encoder_attn_mask=None,
                                           decoder_attn_mask=dec_mask,
                                           encoder_decoder_attn_mask=enc_dec_mask,
                                           enc_hidden_states=all_query_context_hidden_states)

        if self.training:
            lm_logits_one_context = None
            if args.update_retriever:
                with torch.no_grad():
                    dec_ids_repeated = torch.repeat_interleave(dec_ids, topk, dim=0)

                    query_one_context_mask = make_attention_mask_3d(query_one_context_ids,
                                                                    query_one_context_ids)
                    query_one_context_mask = query_one_context_mask < 0.5

                    enc_dec_mask = make_attention_mask_3d(dec_ids_repeated, query_one_context_ids)
                    enc_dec_mask = enc_dec_mask < 0.5

                    dec_mask = make_attention_mask_3d(dec_ids_repeated, dec_ids_repeated)
                    dec_mask = dec_mask * make_history_mask_3d(dec_ids_repeated)
                    dec_mask = dec_mask < 0.5

                    flat_lm_logits_one_context, _ = self.language_model(encoder_input_ids=query_one_context_ids,
                                                                        decoder_input_ids=dec_ids_repeated,
                                                                        encoder_attn_mask=query_one_context_mask,
                                                                        decoder_attn_mask=dec_mask,
                                                                        encoder_decoder_attn_mask=enc_dec_mask,
                                                                        tokentype_ids=None)
                    _, decoder_seq_length, vocab_size = flat_lm_logits_one_context.shape
                    # B K x T x V -> B x K x T x V
                    lm_logits_one_context = flat_lm_logits_one_context.reshape(bsize, topk, decoder_seq_length, vocab_size)

            return lm_logits, topk_log_probs, lm_logits_one_context
        else:
            return lm_logits, topk_log_probs, all_query_context_hidden_states, all_query_context_ids_unflat


    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        """For easy load when model is combined with other heads, add an extra key."""
        state_dict_ = dict()
        state_dict_[self._language_model_key] = self.language_model.state_dict_for_save_checkpoint(destination,
                                                                                                   prefix,
                                                                                                   keep_vars)
        state_dict_[self._retriever_model_key] = self.retriever_model.state_dict_for_save_checkpoint(destination,
                                                                                                     prefix,
                                                                                                     keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Load the state dicts of each of the models"""
        self.language_model.load_state_dict(state_dict[self._language_model_key], strict)
        self.retriever_model.load_state_dict(state_dict[self._retriever_model_key], strict)

    def init_state_dict_from_dpr_and_t5(self):
        """Initialize the state from pre-trained DPR model and pre-trained T5 mode on iteration zero of pretraining"""
        args = get_args()

        if args.pretrained_dpr_load is None or args.pretrained_t5_load is None or args.stale_checkpoint_path is None:
            warnings.warn("Pretrained Checkpoints are not found. Initializing from random weights")
            return

        print("Initializing reader model from pretrained T5", flush=True)
        load_t5_checkpoint(self.language_model,
                           custom_load_path=args.pretrained_t5_load)

        print("Initializing retriever model from pretrained BERT", flush=True)
        load_dualencoder_checkpoint(self.retriever_model,
                                    custom_load_path=args.pretrained_dpr_load)


def postprocess(query_uid, query_ids_t5, query_ids_t5_len, topk_evidence_data):
    args = get_args()
    query_uid = query_uid.tolist()
    t5_tokenizer = get_t5_tokenizer()

    all_context_ids, all_context_types = [], []
    all_query_extended_context_ids = []
    all_query_one_context_ids = []

    for qid, query_t5, query_t5_len, (topkids, text_list) in zip(query_uid, query_ids_t5, query_ids_t5_len, topk_evidence_data):
        k = 0
        query_t5 = query_t5.tolist()[:query_t5_len]
        context_ids_list, context_types_list = [], []
        for eid, (context_doc_list, main_doc_idx, title_ids) in zip(topkids, text_list):
            # We should ignore the evidence from which query originates
            if qid != eid and k < args.topk_retrievals:
                k += 1
                # Except for the masked tokens from extra-vocab-ids, BERT tokenizer and T5 tokenizer output the same encodings
                context_ids = context_doc_list[main_doc_idx]

                ids, types, pad_mask = context_bert_format(title_ids + [t5_tokenizer.sep] + context_ids,
                                                           args.seq_length_ret,
                                                           t5_tokenizer.cls,
                                                           t5_tokenizer.sep,
                                                           t5_tokenizer.pad)
                context_ids_list.append(ids)
                context_types_list.append(types)

                # Jointly encode the query and ...extended context tokens...
                query_context_ids = query_extended_context_t5_format(query_t5,
                                                                     title_ids,
                                                                     context_doc_list,
                                                                     main_doc_idx,
                                                                     args.seq_length,
                                                                     t5_tokenizer.sep,
                                                                     t5_tokenizer.pad)
                all_query_extended_context_ids.append(query_context_ids)

                # Jointly encode the query and ...single context tokens...
                query_one_context_ids = query_single_context_t5_format(query_t5,
                                                                       title_ids,
                                                                       context_ids,
                                                                       args.seq_length,
                                                                       t5_tokenizer.sep,
                                                                       t5_tokenizer.pad)
                all_query_one_context_ids.append(query_one_context_ids)

        all_context_ids.append(np.array(context_ids_list))
        all_context_types.append(np.array(context_types_list))

    return torch.cuda.LongTensor(all_context_ids), \
           torch.cuda.LongTensor(all_context_types), \
           torch.cuda.LongTensor(all_query_extended_context_ids), \
           torch.cuda.LongTensor(all_query_one_context_ids)


def query_extended_context_t5_format(query_ids, title_ids, context_doc_list, main_doc_idx, max_seq_length, sep_id, pad_id):
    enc_ids = query_ids + title_ids + [sep_id]

    def prepare_context_ids(maxlen):
        context_ids = context_doc_list[main_doc_idx]

        if len(context_ids) > maxlen or len(context_doc_list) == 1:
            context_ids = context_ids[0: maxlen]
            return context_ids
        else:
            extra_len = maxlen - len(context_ids)
            if main_doc_idx == 0:
                extra_context_ids = []
                for item in context_doc_list[1:]:
                    extra_context_ids.extend(item)
                if len(extra_context_ids) > extra_len:
                    extra_context_ids = extra_context_ids[0: extra_len]
                context_ids = context_ids + extra_context_ids
                return context_ids
            elif main_doc_idx == -1:
                extra_context_ids = []
                for item in context_doc_list[:-1]:
                    extra_context_ids.extend(item)
                if len(extra_context_ids) > extra_len:
                    offset = len(extra_context_ids) - extra_len + 1
                    extra_context_ids = extra_context_ids[offset:]
                context_ids = extra_context_ids + context_ids
                return context_ids
            else:  # for condition main_doc_idx=1
                left_extra_context_ids = context_doc_list[0]
                if len(left_extra_context_ids) > extra_len:
                    offset = len(left_extra_context_ids) - extra_len + 1
                    left_extra_context_ids = left_extra_context_ids[offset:]
                    context_ids = left_extra_context_ids + context_ids
                    return context_ids
                context_ids = left_extra_context_ids + context_ids
                if len(context_doc_list) == 3:
                    right_extra_context_ids = context_doc_list[2]
                    len_remaining = extra_len - len(left_extra_context_ids)
                    if len(right_extra_context_ids) > len_remaining:
                        right_extra_context_ids = right_extra_context_ids[:len_remaining]
                    context_ids = context_ids + right_extra_context_ids
                return context_ids

    remaining_len = max(0, max_seq_length - len(enc_ids) - 1)
    extended_context_ids = prepare_context_ids(remaining_len)
    enc_ids.extend(extended_context_ids)
    enc_ids.append(sep_id)

    padding_length = max_seq_length - len(enc_ids)
    if padding_length > 0:
        enc_ids.extend([pad_id] * padding_length)

    return enc_ids


def query_single_context_t5_format(query_ids, title_ids, context_ids, max_seq_length, sep_id, pad_id):
    enc_ids = []
    src_ids = query_ids + title_ids + [sep_id] + context_ids
    enc_ids.extend(src_ids)

    if len(enc_ids) > max_seq_length - 1:
        enc_ids = enc_ids[0: max_seq_length - 1]

    enc_ids.append(sep_id)

    padding_length = max_seq_length - len(enc_ids)
    if padding_length > 0:
        enc_ids.extend([pad_id] * padding_length)

    return enc_ids


class PreComputedEvidenceDocsRetriever(object):
    def __init__(self):
        args = get_args()
        self.topk = args.topk_retrievals
        self.embedding_size = args.hidden_size
        self.evidence_embedder_obj = None
        self.mips_index = None

        self.precomputed_index_wrapper()

        self.allow_trivial_doc = args.allow_trivial_doc
        if not args.allow_trivial_doc:
            self.topk = self.topk + 1

        self.local_rank = args.local_rank
        world_size = torch.distributed.get_world_size()
        if world_size == 1:
            self.device_count = 1
        else:
            # Note: This will work only for 1 node. May need to fix this?
            self.device_count = min(torch.cuda.device_count(), args.max_training_rank)

        self.passages_map = make_indexed_dataset(args.indexed_evidence_data_path,
                                                 impl=args.data_impl,
                                                 skip_warmup=(not args.mmap_warmup))

        self.title_map = make_indexed_dataset(args.indexed_title_data_path,
                                              impl=args.data_impl,
                                              skip_warmup=(not args.mmap_warmup))
        self.wikititledocmap = WikiTitleDocMap(args.evidence_data_path)

    def get_evidence_embedding(self, path):
        self.evidence_embedder_obj = OpenRetreivalDataStore(path,
                                                            load_from_path=True)

    def precomputed_index_wrapper(self):
        args = get_args()
        if get_node_first_rank() == torch.distributed.get_rank():
            self.get_evidence_embedding(args.embedding_path)
            assert self.evidence_embedder_obj is not None
            self.mips_index = DistributedBruteForceIndex(embed_size=self.embedding_size,
                                                         embed_data=self.evidence_embedder_obj,
                                                         use_gpu=args.faiss_use_gpu)
        # Wait for the index to be initialized in all the GPUs
        torch.distributed.barrier(get_data_parallel_group())


    def update_evidence_embedding(self):
        """Reload index with new data loaded from disk. Should be performed after each indexer job completes."""
        if get_node_first_rank() == torch.distributed.get_rank():
            self.mips_index.update_index()

        # Wait for the MIPS index to be initialized in all the nodes
        torch.distributed.barrier(get_data_parallel_group())


    def get_topk(self, query_tensor):
        local_bsize = query_tensor.shape[0]
        input_ = torch.empty_like(query_tensor).copy_(query_tensor).detach_()
        tensor_list = [torch.empty_like(input_) for _ in range(self.device_count)]
        torch.distributed.all_gather(tensor_list, query_tensor, group=get_mips_group())

        if get_node_first_rank() == torch.distributed.get_rank():
            assert self.mips_index is not None, "MIPS Index is not initialized"
            all_query_tensor = torch.cat(tensor_list, dim=0).contiguous()
            distance, topkindex = self.mips_index.search_mips_index(all_query_tensor,
                                                                    top_k=self.topk,
                                                                    reconstruct=False)
        else:
            distance = torch.empty(self.device_count * local_bsize, self.topk, dtype=torch.float16).cuda()
            topkindex = torch.empty(self.device_count * local_bsize, self.topk, dtype=torch.int32).cuda()

        torch.distributed.broadcast(distance, src=get_node_first_rank(), group=get_mips_group())
        torch.distributed.broadcast(topkindex, src=get_node_first_rank(), group=get_mips_group())

        distance = torch.split(distance, local_bsize, dim=0)[self.local_rank]
        topkindex = torch.split(topkindex, local_bsize, dim=0)[self.local_rank]

        topk_data = []
        for topkarray in topkindex:
            # The idx contains passage text and title
            topkarray = topkarray.tolist()
            text_list = []
            for idx in topkarray:
                doc_idxs, main_doc_idx = self.wikititledocmap.get_neighbour_paragraphs(idx)
                doc_list = [self.passages_map[doc_id-1].tolist() for doc_id in doc_idxs]
                text_list.append((doc_list,
                                  main_doc_idx,
                                  self.title_map[idx-1].tolist()))
            topk_data.append((topkarray, text_list))

        return topk_data, distance
