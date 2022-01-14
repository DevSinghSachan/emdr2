import os
import torch
from megatron import get_args, print_rank_0
from megatron.checkpointing import get_checkpoint_tracker_filename, get_checkpoint_name
from megatron.module import MegatronModule
from megatron import mpu, get_tokenizer
from megatron.model.utils import init_method_normal
from megatron.model.language_model import get_language_model
from megatron.model.utils import scaled_init_method_normal
from megatron.model.bert_model import bert_attention_mask_func, bert_position_ids


def dualencoder_model_provider(only_query_model=False, only_context_model=False, vocab_size=None):
    args = get_args()
    assert args.model_parallel_size == 1, "Model parallel size > 1 not supported."
    print_rank_0('building DualEncoderModel...')

    # simpler to just keep using 2 tokentypes since the LM we initialize with has 2 tokentypes
    model = DualEncoderModel(num_tokentypes=2,
                             parallel_output=True,
                             only_query_model=only_query_model,
                             only_context_model=only_context_model,
                             vocab_size=vocab_size)
    return model


class DualEncoderModel(MegatronModule):
    def __init__(self,
                 num_tokentypes=1,
                 parallel_output=True,
                 only_query_model=False,
                 only_context_model=False,
                 vocab_size=None):
        super(DualEncoderModel, self).__init__()
        args = get_args()

        bert_kwargs = dict(
            num_tokentypes=num_tokentypes,
            parallel_output=parallel_output,
            vocab_size=vocab_size
        )

        assert not (only_context_model and only_query_model)
        self.use_context_model = not only_query_model
        self.use_query_model = not only_context_model

        if self.use_query_model:
            # this model embeds (pseudo-)queries - Embed_input in the paper
            self.query_model = PretrainedBertModel(**bert_kwargs)
            self._query_key = 'query_model'

        if self.use_context_model:
            # this model embeds evidence blocks - Embed_doc in the paper
            self.context_model = PretrainedBertModel(**bert_kwargs)
            self._context_key = 'context_model'

    def forward(self, query_tokens, query_attention_mask, query_types,
                context_tokens, context_attention_mask, context_types):
        """Run a forward pass for each of the models and return the respective embeddings."""
        if self.use_query_model:
            query_logits = self.embed_text(self.query_model,
                                           query_tokens,
                                           query_attention_mask,
                                           query_types)
        else:
            raise ValueError("Cannot embed query without the query model.")
        if self.use_context_model:
            context_logits = self.embed_text(self.context_model,
                                             context_tokens,
                                             context_attention_mask,
                                             context_types)
        else:
            raise ValueError("Cannot embed block without the block model.")
        return query_logits, context_logits

    @staticmethod
    def embed_text(model, tokens, attention_mask, token_types):
        """Embed a batch of tokens using the model"""
        pooled_logits = model(tokens,
                              attention_mask,
                              token_types)
        return pooled_logits

    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        """Save dict with state dicts of each of the models."""
        state_dict_ = {}

        if self.use_query_model:
            state_dict_[self._query_key] = \
                self.query_model.state_dict_for_save_checkpoint(
                    destination, prefix, keep_vars)

        if self.use_context_model:
            state_dict_[self._context_key] = \
                self.context_model.state_dict_for_save_checkpoint(
                    destination, prefix, keep_vars)

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Load the state dicts of each of the models"""

        if self.use_query_model:
            print_rank_0("Loading query model")
            self.query_model.load_state_dict(state_dict[self._query_key], strict=strict)

        if self.use_context_model:
            print_rank_0("Loading context model")
            self.context_model.load_state_dict(state_dict[self._context_key], strict=strict)

    def init_state_dict_from_bert(self):
        """Initialize the state from a pretrained BERT model on iteration zero of ICT pretraining"""
        args = get_args()

        if args.bert_load is None:
            return

        tracker_filename = get_checkpoint_tracker_filename(args.bert_load)
        if not os.path.isfile(tracker_filename):
            raise FileNotFoundError("Could not find BERT checkpoint for Biencoder")
        with open(tracker_filename, 'r') as f:
            iteration = int(f.read().strip())
            assert iteration > 0

        checkpoint_name = get_checkpoint_name(args.bert_load, iteration, False)
        if mpu.get_data_parallel_rank() == 0:
            print('global rank {} is loading BERT checkpoint {}'.format(
                torch.distributed.get_rank(), checkpoint_name))

        try:
            state_dict = torch.load(checkpoint_name, map_location='cpu')
        except BaseException:
            raise ValueError("Could not load BERT checkpoint")

        # load the LM state dict into each model
        model_dict = state_dict['model']['language_model']

        if self.use_query_model:
            self.query_model.language_model.load_state_dict(model_dict)
            # give each model the same ict_head to begin with as well

        if self.use_context_model:
            self.context_model.language_model.load_state_dict(model_dict)


class PretrainedBertModel(MegatronModule):
    def __init__(self, num_tokentypes=2, parallel_output=True, vocab_size=None):
        super(PretrainedBertModel, self).__init__()

        args = get_args()
        tokenizer = get_tokenizer()
        self.pad_id = tokenizer.pad
        self.parallel_output = parallel_output
        init_method = init_method_normal(args.init_method_std)
        scaled_init_method = scaled_init_method_normal(args.init_method_std,
                                                       args.num_layers)

        self.language_model, self._language_model_key = get_language_model(
            attention_mask_func=bert_attention_mask_func,
            num_tokentypes=num_tokentypes,
            add_pooler=False,
            init_method=init_method,
            scaled_init_method=scaled_init_method,
            vocab_size=vocab_size)

    def forward(self, input_ids, attention_mask, tokentype_ids=None):
        extended_attention_mask = attention_mask.unsqueeze(1)
        position_ids = bert_position_ids(input_ids)

        lm_output = self.language_model(input_ids,
                                        position_ids,
                                        extended_attention_mask,
                                        tokentype_ids=tokentype_ids)

        # Taking the representation of the [CLS] token.
        pooled_output = lm_output[:, 0, :]

        # Converting to float16 dtype
        pooled_output = pooled_output.to(lm_output.dtype)

        return pooled_output

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        state_dict_ = {}
        state_dict_[self._language_model_key] = self.language_model.state_dict_for_save_checkpoint(destination,
                                                                                                   prefix,
                                                                                                   keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        print_rank_0("loading BERT weights")
        self.language_model.load_state_dict(state_dict[self._language_model_key],
                                            strict=strict)
