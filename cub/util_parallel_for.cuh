/******************************************************************************
 * 简单的 parallel_for 实现（绕过 thrust）
 * 仅用于 CUB 测试验证
 ******************************************************************************/

#pragma once

#include <musa_runtime.h>

CUB_NAMESPACE_BEGIN

namespace detail {

/// 简单的 parallel_for 内核
template <typename F, typename Size>
__global__ void ParallelForKernel(F f, Size num_items) {
  Size idx = static_cast<Size>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_items) {
    f(idx);
  }
}

/// 简单的 parallel_for 函数
template <typename F, typename Size>
inline musaError_t ParallelFor(F f, Size num_items, musaStream_t stream = 0) {
  if (num_items == 0) {
    return musaSuccess;
  }

  const int block_size = 256;
  const int num_blocks = static_cast<int>((num_items + block_size - 1) / block_size);

  ParallelForKernel<<<num_blocks, block_size, 0, stream>>>(f, num_items);

  return musaPeekAtLastError();
}

} // namespace detail

CUB_NAMESPACE_END
