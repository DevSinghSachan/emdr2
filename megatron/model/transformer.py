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

"""Transformer."""

import math
import torch
import torch.nn.functional as F

from megatron import get_args
from megatron import mpu
from megatron.mpu import LayerNorm
from megatron.module import MegatronModule
from megatron.checkpointing import get_checkpoint_version
from megatron.model.fused_softmax import FusedScaleMaskSoftmax
from megatron.model.fused_bias_gelu import bias_gelu_impl
from megatron.model.utils import openai_gelu, erf_gelu

# flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

""" We use the following notation throughout this file:
     h: hidden size
     n: number of attention heads
     p: number of model parallel partitions
     np: n/p
     hp: h/p
     hn: h/n
     b: batch size
     s: sequence length
     l: number of layers
    Transformer takes input of size [s, b, h] and returns a
    tensor of the same size. We use the following arguments:
        hyperparameters: transformer hyperparameters
        attention_mask_func: a function that takes `unmaksed-attention-scores`
            with size [b, np, s, s] and an `attention-mask` and will apply
            the masking. The function should return a masked score of the
            same size [b, np, s, s].
               masked-attention-scores = attention_mask_func(
                                     unmaksed-attention-scores, attention-mask)
"""

class ParallelMLP(MegatronModule):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.
    """

    def __init__(self, init_method, output_layer_init_method):
        super(ParallelMLP, self).__init__()
        args = get_args()

        # Project to 4h.
        self.dense_h_to_4h = mpu.ColumnParallelLinear(
            args.hidden_size,
            args.ffn_hidden_size,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True)

        self.bias_gelu_fusion = args.bias_gelu_fusion
        self.activation_func = F.gelu
        if args.openai_gelu:
            self.activation_func = openai_gelu
        elif args.onnx_safe:
            self.activation_func = erf_gelu

        # Project back to h.
        self.dense_4h_to_h = mpu.RowParallelLinear(
            args.ffn_hidden_size,
            args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True)

    def forward(self, hidden_states):

        # [s, b, 4hp]
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)

        if self.bias_gelu_fusion:
             intermediate_parallel = \
                     bias_gelu_impl(intermediate_parallel, bias_parallel)
        else:
            intermediate_parallel = \
                self.activation_func(intermediate_parallel + bias_parallel)

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, output_bias


class ParallelAttention(MegatronModule):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(self, attention_mask_func, init_method,
                 output_layer_init_method, layer_number,
                 attention_type="self"):
        super(ParallelAttention, self).__init__()
        args = get_args()
        self.fp16 = args.fp16

        self.attention_mask_func = attention_mask_func
        self.apply_query_key_layer_scaling = args.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = args.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type

        projection_size = args.kv_channels * args.num_attention_heads

        # Per attention head and per partition values.
        world_size = mpu.get_model_parallel_world_size()
        self.hidden_size_per_partition = mpu.divide(projection_size,
                                                    world_size)
        self.hidden_size_per_attention_head = mpu.divide(
            projection_size, args.num_attention_heads)
        self.num_attention_heads_per_partition = mpu.divide(
            args.num_attention_heads, world_size)

        # Strided linear layer.
        if attention_type == "self":
            self.query_key_value = mpu.ColumnParallelLinear(
                args.hidden_size,
                3 * projection_size,
                gather_output=False,
                init_method=init_method)
        else:
            assert attention_type == "cross"
            self.query = mpu.ColumnParallelLinear(
                args.hidden_size,
                projection_size,
                gather_output=False,
                init_method=init_method)

            self.key_value = mpu.ColumnParallelLinear(
                args.hidden_size,
                2 * projection_size,
                gather_output=False,
                init_method=init_method)

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            self.fp16,
            args.scaled_upper_triang_masked_softmax_fusion,
            args.scaled_masked_softmax_fusion,
            self.attention_mask_func,
            self.attention_softmax_in_fp32,
            coeff)

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(args.attention_dropout)

        # Output.
        self.dense = mpu.RowParallelLinear(
            projection_size,
            args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True)

    def _transpose_last_dim(self, mixed_layer, num_splits):
        """[s, b, num_splits * hp] 
        -->(view) [s, b, num_splits, hp] 
        -->(tranpose) [s, b, hp, num_splits] 
        -->(view) [s, b, num_splits * hp] """

        input_shape = mixed_layer.size();
        last_dim = input_shape[-1]
        assert last_dim % num_splits == 0, "expected QKV dimension"
        last_dim_split = last_dim // num_splits
        
        intermediate_shape = input_shape[:-1] +\
            (num_splits, last_dim_split)

        mixed_layer = mixed_layer.view(*intermediate_shape)
        mixed_layer = mixed_layer.transpose(-1, -2).contiguous()
        mixed_layer = mixed_layer.view(*input_shape)
        
        return mixed_layer

    def forward(self, hidden_states, attention_mask, layer_past=None,
                get_key_value=False, encoder_output=None, sim_scores=None):
        # hidden_states: [sq, b, h]

        # =====================
        # Query, Key, and Value
        # =====================


        if self.attention_type == "self":
            # Attention heads [sq, b, h] --> [sq, b, hp * 3]
            mixed_x_layer, _ = self.query_key_value(hidden_states)

            checkpoint_version = get_checkpoint_version()
            if checkpoint_version is not None and \
               checkpoint_version == 0:
                # [s, b, 3 * hp] --> [s, b, hp * 3]
                mixed_x_layer = self._transpose_last_dim(mixed_x_layer, 3)

            # [sq, b, hp * 3] --> [sq, b, np, hn, 3]
            new_tensor_shape = mixed_x_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                 self.hidden_size_per_attention_head, 3)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, np, hn, 3] --> 3 [sq, b, np, hn]
            query_layer = mixed_x_layer[:,:,:,:,0]
            key_layer = mixed_x_layer[:,:,:,:,1]
            value_layer = mixed_x_layer[:,:,:,:,2]
        else:
            # Attention heads [sk, b, h] --> [sk, b, hp * 2]
            mixed_kv_layer, _ = self.key_value(encoder_output)

            checkpoint_version = get_checkpoint_version()
            if checkpoint_version is not None and \
               checkpoint_version == 0:
                # [s, b, 2 * hp] --> [s, b, hp * 2]
                mixed_kv_layer = self._transpose_last_dim(mixed_kv_layer, 2)
                
            # [sk, b, hp * 2] --> [sk, b, np, hn, 2]  
            new_tensor_shape = mixed_kv_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                 self.hidden_size_per_attention_head, 2)
            mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

            # [sk, b, np, hn, 2] --> 2 [sk, b, np, hn]
            key_layer = mixed_kv_layer[:,:,:,:,0]
            value_layer = mixed_kv_layer[:,:,:,:,1]

            # Attention head [sq, b, h] --> [sq, b, hp]
            query_layer, _ = self.query(hidden_states)
             # [sq, b, hp] --> [sq, b, np, hn]  
            new_tensor_shape = query_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                 self.hidden_size_per_attention_head)
            query_layer = query_layer.view(*new_tensor_shape)

        # ==================================
        # Adjust key and value for inference
        # ==================================

        if layer_past is not None:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer),
                                   key_layer), dim=0)
            value_layer = torch.cat((past_value.type_as(value_layer),
                                     value_layer), dim=0)
        if get_key_value:
            present = (key_layer, value_layer)


        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================
        
        # [b, np, sq, sk]
        output_size = (query_layer.size(1), 
                       query_layer.size(2), 
                       query_layer.size(0), 
                       key_layer.size(0))
        
        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2],
                                       output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn
        key_layer = key_layer.view(output_size[3],
                                   output_size[0] * output_size[1], -1)

        # preallocting result tensor: [b * np, sq, sk]
        matmul_result = torch.empty(
            output_size[0]*output_size[1], 
            output_size[2], 
            output_size[3],
            dtype=query_layer.dtype, 
            device=torch.cuda.current_device())

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(matmul_result, 
            query_layer.transpose(0, 1),   # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  #[b * np, hn, sk]
            beta=0.0, alpha=(1.0/self.norm_factor))

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ==================================================
        # Update attention mask for inference. [b, np, sq, sk]
        # ==================================================

        if get_key_value:
            with torch.no_grad():
                if layer_past is not None:
                    attention_mask = attention_mask[
                        ...,
                        attention_scores.size(3) - 1,
                        :attention_scores.size(3)].unsqueeze(2)
                else:
                    attention_mask = attention_mask[
                        ...,
                        :attention_scores.size(3),
                        :attention_scores.size(3)]


        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores,
                                                  attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        with mpu.get_cuda_rng_tracker().fork():
            attention_probs = self.attention_dropout(attention_probs)


        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(1), 
                       value_layer.size(2), 
                       query_layer.size(0), 
                       value_layer.size(3)) 

        # change view [sk, b * np, hn] 
        value_layer = value_layer.view(value_layer.size(0),
                                       output_size[0] * output_size[1], -1)
        
        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1],
                                               output_size[2], -1)
        
        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0,1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + \
            (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)


        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.dense(context_layer)

        if get_key_value:
            output = [output, present]

        return output, bias


def bias_dropout_add(x, bias, residual, prob, training) :
    # type: (Tensor, Tensor, Tensor, float, bool) -> Tensor
    out = torch.nn.functional.dropout(x + bias, p=prob, training=training)
    out = residual + out
    return out


def get_bias_dropout_add(training):
    def _bias_dropout_add(x, bias, residual, prob):
        return bias_dropout_add(x, bias, residual, prob, training)
    return _bias_dropout_add


@torch.jit.script
def bias_dropout_add_fused_train(x, bias, residual, prob) :
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    return bias_dropout_add(x, bias, residual, prob, True)


@torch.jit.script
def bias_dropout_add_fused_inference(x, bias, residual, prob) :
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    return bias_dropout_add(x, bias, residual, prob, False)


class ParallelTransformerLayer(MegatronModule):
    """A single transformer layer.

    Transformer layer takes input with size [b, s, h] and returns an
    output of the same size.
    """

    def __init__(self, attention_mask_func, init_method, 
                 output_layer_init_method, layer_number,
                 layer_type='encoder'):
        args = get_args()

        super(ParallelTransformerLayer, self).__init__()
        self.layer_number = layer_number
        self.layer_type = layer_type

        self.apply_residual_connection_post_layernorm \
            = args.apply_residual_connection_post_layernorm

        # Layernorm on the input data.
        self.input_layernorm = LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon)

        # Self attention.
        self.self_attention = ParallelAttention(attention_mask_func,
                                           init_method,
                                           output_layer_init_method,
                                           layer_number)
        self.hidden_dropout = args.hidden_dropout
        self.bias_dropout_fusion = args.bias_dropout_fusion

        # Layernorm on the attention output.
        self.post_attention_layernorm = LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon)

        if self.layer_type == "decoder":
            self.inter_attention = ParallelAttention(attention_mask_func,
                                                     init_method,
                                                     output_layer_init_method,
                                                     layer_number,
                                                     attention_type="cross")
            # Layernorm on the attention output.
            self.post_inter_attention_layernorm = LayerNorm(
                args.hidden_size,
                eps=args.layernorm_epsilon)
   
        # MLP
        self.mlp = ParallelMLP(init_method,
                               output_layer_init_method)

    def forward(self, hidden_states, attention_mask,
                encoder_output=None, enc_dec_attn_mask=None, sim_scores=None,
                layer_past=None, get_key_value=False):
        # hidden_states: [b, s, h]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output, attention_bias = \
            self.self_attention(layernorm_output,
                           attention_mask,
                           layer_past=layer_past,
                           get_key_value=get_key_value)

        if get_key_value:
            attention_output, presents = attention_output
    
        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # jit scripting for a nn.module (with dropout) is not 
        # trigerring the fusion kernel. For now, we use two 
        # different nn.functional routines to account for varying
        # dropout semantics during training and inference phases.
        if self.bias_dropout_fusion:
            if self.training:
                bias_dropout_add_func = bias_dropout_add_fused_train
            else:
                bias_dropout_add_func = bias_dropout_add_fused_inference
        else:
            bias_dropout_add_func = get_bias_dropout_add(self.training)

        #re-enable torch grad to enable fused optimization.
        with torch.enable_grad():
            layernorm_input = bias_dropout_add_func(
                attention_output,
                attention_bias.expand_as(residual),
                residual,
                self.hidden_dropout)

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        if self.layer_type == "decoder":
            attention_output, attention_bias = \
                self.inter_attention(layernorm_output,
                                     enc_dec_attn_mask,
                                     encoder_output=encoder_output,
                                     sim_scores=sim_scores)
            # residual connection
            if self.apply_residual_connection_post_layernorm: 
                residual = layernorm_output
            else: 
                residual = layernorm_input
            
            #re-enable torch grad to enable fused optimization.
            with torch.enable_grad():
                layernorm_input = bias_dropout_add_func(
                    attention_output,
                    attention_bias.expand_as(residual),
                    residual,
                    self.hidden_dropout)   
                   
            # Layer norm post the decoder attention
            layernorm_output = self.post_inter_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output, mlp_bias = self.mlp(layernorm_output)
        
        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        #re-enable torch grad to enable fused optimization.
        with torch.enable_grad():
            output = bias_dropout_add_func(
                mlp_output,
                mlp_bias.expand_as(residual),
                residual,
                self.hidden_dropout)

        if get_key_value:
            output = [output, presents]

        return output


class ParallelTransformer(MegatronModule):
    """Transformer class."""

    def __init__(self, attention_mask_func,
                 init_method, output_layer_init_method,
                 layer_type='encoder'):
        super(ParallelTransformer, self).__init__()
        args = get_args()

        # Store activation checkpoiting flag.
        self.checkpoint_activations = args.checkpoint_activations
        self.checkpoint_num_layers = args.checkpoint_num_layers

        # Number of layers:
        self.num_layers = args.num_layers
        self.num_unique_layers = args.num_unique_layers
        if self.num_unique_layers is None:
            self.num_unique_layers = self.num_layers
        assert self.num_layers % self.num_unique_layers == 0, \
            'number of layers should be divisible by number of unique layers'
        self.param_sharing_style = args.param_sharing_style

        # Transformer layers.
        def build_layer(layer_number):
            return ParallelTransformerLayer(
                attention_mask_func, init_method,
                output_layer_init_method, layer_number,
                layer_type=layer_type)
        self.layers = torch.nn.ModuleList(
            [build_layer(i + 1) for i in range(self.num_unique_layers)])

        # Print layer ordering.
        if self.num_layers != self.num_unique_layers:
            if torch.distributed.get_rank() == 0:
                print('> will be using the following layer ordering:')
                for i in range(self.num_layers):
                    print('   layer id: {:3d} --> unique layer id: '
                          '{:3d}'.format(i, self._get_layer_index(i)),
                          flush=True)

        # Final layer norm before output.
        self.final_layernorm = LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon)

    def _get_layer_index(self, layer_number):
        if self.param_sharing_style == 'grouped':
            return layer_number % self.num_unique_layers
        if self.param_sharing_style == 'spaced':
            return layer_number // (self.num_layers // self.num_unique_layers) 
        assert False, 'should not be here'

    def _get_layer(self, layer_number):
        return self.layers[self._get_layer_index(layer_number)]

    def _checkpointed_forward(self, hidden_states, attention_mask,
                              encoder_output, enc_dec_attn_mask, sim_scores):
        """Forward method with activation checkpointing."""
        def custom(start, end):
            def custom_forward(*inputs):
                x_ = inputs[0]
                attention_mask = inputs[1]
                encoder_output = inputs[2]
                enc_dec_attn_mask = inputs[3]
                sim_scores = inputs[4]
                for index in range(start, end):
                    layer = self._get_layer(index)
                    x_ = layer(x_, attention_mask, encoder_output, enc_dec_attn_mask, sim_scores)
                return x_
            return custom_forward

        # Make sure memory is freed.
        mpu.reset_checkpointed_activations_memory_buffer()
        l = 0
        while l < self.num_layers:
            hidden_states = mpu.checkpoint(
                custom(l, l + self.checkpoint_num_layers),
                hidden_states, attention_mask, encoder_output, enc_dec_attn_mask, sim_scores)
            l += self.checkpoint_num_layers

        return hidden_states

    def forward(self, hidden_states, attention_mask, layer_past=None,
                get_key_value=False, encoder_output=None, enc_dec_attn_mask=None, sim_scores=None):

        # Checks
        if layer_past is not None:
            assert get_key_value, \
                'for not None values in layer_past, ' \
                'expected get_key_value to be set'
        if get_key_value:
            assert not self.checkpoint_activations, \
                'get_key_value does not work with ' \
                'activation checkpointing'

        # data format change to avoid explicit tranposes : [b s h] --> [s b h]
        hidden_states = hidden_states.transpose(0, 1).contiguous()
        if encoder_output is not None:
            encoder_output = encoder_output.transpose(0, 1).contiguous()

        if self.checkpoint_activations:
            hidden_states = self._checkpointed_forward(hidden_states,
                                                       attention_mask,
                                                       encoder_output,
                                                       enc_dec_attn_mask,
                                                       sim_scores)
        else:
            if get_key_value:
                presents = []
            for index in range(self.num_layers):
                layer = self._get_layer(index)
                past = None
                if layer_past is not None:
                    past = layer_past[index]
                hidden_states = layer(hidden_states,
                                      attention_mask,
                                      encoder_output=encoder_output,
                                      enc_dec_attn_mask=enc_dec_attn_mask,
                                      sim_scores=sim_scores,
                                      layer_past=past,
                                      get_key_value=get_key_value)
                if get_key_value:
                    hidden_states, present = hidden_states
                    presents.append(present)
        
        # reverting data format change [s b h] --> [b s h]
        hidden_states = hidden_states.transpose(0, 1).contiguous()

        # Final layer norm.
        output = self.final_layernorm(hidden_states)
        if get_key_value:
            output = [output, presents]

        return output
