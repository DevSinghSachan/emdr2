/* coding=utf-8
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include "THC/THC.h"
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "scaled_masked_softmax.h"

namespace multihead_attn {
namespace fused_softmax {
namespace scaled_masked_softmax {

torch::Tensor fwd_cuda(
    torch::Tensor const& input,
    torch::Tensor const& mask,
    float scale_factor)
{
  // input is a 4d tensor with dimensions [batches, attn_heads, seq_len, seq_len]
  const int batches = input.size(0);
  const int pad_batches = mask.size(0);
  const int attn_heads = input.size(1);
  const int seq_len = input.size(2);
  TORCH_INTERNAL_ASSERT(seq_len <= 2048);
  TORCH_INTERNAL_ASSERT(pad_batches == 1 || pad_batches == batches);
  TORCH_INTERNAL_ASSERT(mask.size(1) == 1);
  TORCH_INTERNAL_ASSERT(mask.size(2) == seq_len);
  TORCH_INTERNAL_ASSERT(mask.size(3) == seq_len);

  // Output 
  auto act_options = input.options().requires_grad(false);
  torch::Tensor softmax_results = 
      torch::empty({batches, attn_heads, seq_len, seq_len}, act_options);

  // Softmax Intermediate Result Ptr
  void* input_ptr = static_cast<void*>(input.data_ptr());
  void* mask_ptr = static_cast<void*>(mask.data_ptr());
  void* softmax_results_ptr = static_cast<void*>(softmax_results.data_ptr());

  dispatch_scaled_masked_softmax_forward<half, half, float>(
      reinterpret_cast<half*>(softmax_results_ptr),
      reinterpret_cast<const half*>(input_ptr),
      reinterpret_cast<const uint8_t*>(mask_ptr),
      scale_factor,
      seq_len,
      seq_len,
      batches,
      attn_heads,
      pad_batches);
  return softmax_results;
}

torch::Tensor bwd_cuda(
    torch::Tensor const& output_grads_, 
    torch::Tensor const& softmax_results_, 
    float scale_factor)  {
	
  auto output_grads = output_grads_.contiguous();
  auto softmax_results = softmax_results_.contiguous();

  //output grads is a 4d tensor with dimensions [batches, attn_heads, seq_len, seq_len]
  const int batches = output_grads.size(0);
  const int attn_heads = output_grads.size(1);
  const int seq_len = output_grads.size(2);
  TORCH_INTERNAL_ASSERT(output_grads.size(2) == output_grads.size(3));

  void* output_grads_ptr = static_cast<void*>(output_grads.data_ptr());

  //Softmax Grad
  dispatch_scaled_masked_softmax_backward<half, half, float>(
      reinterpret_cast<half*>(output_grads_ptr), 
      reinterpret_cast<half*>(output_grads_ptr), 
      reinterpret_cast<half const*>(softmax_results.data_ptr()),
      scale_factor,
      seq_len,
      seq_len,
      batches,
      attn_heads);
  
  //backward pass is completely in-place
  return output_grads;
}
}
}
}
