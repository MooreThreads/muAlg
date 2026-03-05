/*******************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

// No-thrust version of test_thread_sort.cu for MUSA platform testing

#include "test_util.h"
#include "cub/thread/thread_sort.cuh"

#include <musa_runtime.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

struct CustomLess
{
  template <typename DataType>
  __host__ __device__ bool operator()(DataType &lhs, DataType &rhs)
  {
    return lhs < rhs;
  }
};


template <typename KeyT,
          typename ValueT,
          int ItemsPerThread>
__global__ void kernel(const KeyT *keys_in,
                       KeyT *keys_out,
                       const ValueT *values_in,
                       ValueT *values_out)
{
  KeyT thread_keys[ItemsPerThread];
  KeyT thread_values[ItemsPerThread];

  const auto thread_offset = ItemsPerThread * threadIdx.x;
  keys_in += thread_offset;
  keys_out += thread_offset;
  values_in += thread_offset;
  values_out += thread_offset;

  for (int item = 0; item < ItemsPerThread; item++)
  {
    thread_keys[item] = keys_in[item];
    thread_values[item] = values_in[item];
  }

  cub::StableOddEvenSort(thread_keys, thread_values, CustomLess{});

  for (int item = 0; item < ItemsPerThread; item++)
  {
    keys_out[item] = thread_keys[item];
    values_out[item] = thread_values[item];
  }
}

//==============================================================================
// Host-side reference sort (stable sort by key)
template <typename KeyT, typename ValueT, int ItemsPerThread>
void ReferenceSort(std::vector<KeyT>& keys, std::vector<ValueT>& values, unsigned int threads_in_block)
{
  for (unsigned int tid = 0; tid < threads_in_block; tid++)
  {
    const auto thread_begin = tid * ItemsPerThread;
    const auto thread_end = thread_begin + ItemsPerThread;

    // Stable sort within each thread's range
    // Create indices for stable sort
    std::vector<int> indices(ItemsPerThread);
    for (int i = 0; i < ItemsPerThread; i++)
    {
      indices[i] = i;
    }

    // Sort indices based on keys
    std::stable_sort(indices.begin(), indices.end(), [&](int a, int b) {
      return keys[thread_begin + a] < keys[thread_begin + b];
    });

    // Reorder keys and values based on sorted indices
    std::vector<KeyT> sorted_keys(ItemsPerThread);
    std::vector<ValueT> sorted_values(ItemsPerThread);
    for (int i = 0; i < ItemsPerThread; i++)
    {
      sorted_keys[i] = keys[thread_begin + indices[i]];
      sorted_values[i] = values[thread_begin + indices[i]];
    }

    for (int i = 0; i < ItemsPerThread; i++)
    {
      keys[thread_begin + i] = sorted_keys[i];
      values[thread_begin + i] = sorted_values[i];
    }
  }
}

//==============================================================================
template <typename KeyT,
          typename ValueT,
          int ItemsPerThread>
bool Test()
{
  const unsigned int threads_in_block = 1024;
  const unsigned int elements = threads_in_block * ItemsPerThread;

  std::mt19937 re(42);
  std::uniform_int_distribution<unsigned int> dist(0, 255);

  bool all_pass = true;

  for (int iteration = 0; iteration < 10; iteration++)
  {
    // Generate random source data
    std::vector<std::uint8_t> data_source(elements);
    for (unsigned int i = 0; i < elements; i++)
    {
      data_source[i] = static_cast<std::uint8_t>(dist(re));
    }

    // Shuffle for keys
    std::vector<std::uint8_t> shuffled_keys = data_source;
    std::shuffle(shuffled_keys.begin(), shuffled_keys.end(), re);

    // Shuffle for values
    std::vector<std::uint8_t> shuffled_values = data_source;
    std::shuffle(shuffled_values.begin(), shuffled_values.end(), re);

    // Prepare host input
    std::vector<KeyT> h_in_keys(shuffled_keys.begin(), shuffled_keys.end());
    std::vector<ValueT> h_in_values(shuffled_values.begin(), shuffled_values.end());

    // Copy to device
    KeyT* d_in_keys = nullptr;
    KeyT* d_out_keys = nullptr;
    ValueT* d_in_values = nullptr;
    ValueT* d_out_values = nullptr;

    musaMalloc(&d_in_keys, elements * sizeof(KeyT));
    musaMalloc(&d_out_keys, elements * sizeof(KeyT));
    musaMalloc(&d_in_values, elements * sizeof(ValueT));
    musaMalloc(&d_out_values, elements * sizeof(ValueT));

    musaMemcpy(d_in_keys, h_in_keys.data(), elements * sizeof(KeyT), musaMemcpyHostToDevice);
    musaMemcpy(d_in_values, h_in_values.data(), elements * sizeof(ValueT), musaMemcpyHostToDevice);

    // Run kernel
    kernel<KeyT, ValueT, ItemsPerThread><<<1, threads_in_block>>>(
      d_in_keys, d_out_keys, d_in_values, d_out_values);

    musaDeviceSynchronize();

    // Copy result back
    std::vector<KeyT> h_out_keys(elements);
    std::vector<ValueT> h_out_values(elements);
    musaMemcpy(h_out_keys.data(), d_out_keys, elements * sizeof(KeyT), musaMemcpyDeviceToHost);
    musaMemcpy(h_out_values.data(), d_out_values, elements * sizeof(ValueT), musaMemcpyDeviceToHost);

    // Compute reference
    ReferenceSort<KeyT, ValueT, ItemsPerThread>(h_in_keys, h_in_values, threads_in_block);

    // Compare
    bool keys_match = (h_in_keys == h_out_keys);
    bool values_match = (h_in_values == h_out_values);

    if (!keys_match || !values_match)
    {
      std::cerr << "  Iteration " << iteration << " FAILED" << std::endl;
      all_pass = false;
    }

    // Cleanup
    musaFree(d_in_keys);
    musaFree(d_out_keys);
    musaFree(d_in_values);
    musaFree(d_out_values);
  }

  return all_pass;
}

//==============================================================================
template <typename KeyT,
          typename ValueT>
bool TestType()
{
  std::cout << "  Testing " << typeid(KeyT).name() << "/" << typeid(ValueT).name() << ": ";

  bool all_pass = true;
  if (!Test<KeyT, ValueT, 2>()) all_pass = false;
  if (!Test<KeyT, ValueT, 3>()) all_pass = false;
  if (!Test<KeyT, ValueT, 4>()) all_pass = false;
  if (!Test<KeyT, ValueT, 5>()) all_pass = false;
  if (!Test<KeyT, ValueT, 7>()) all_pass = false;
  if (!Test<KeyT, ValueT, 8>()) all_pass = false;
  if (!Test<KeyT, ValueT, 9>()) all_pass = false;
  if (!Test<KeyT, ValueT, 11>()) all_pass = false;

  std::cout << (all_pass ? "PASS" : "FAIL") << std::endl;
  return all_pass;
}

//==============================================================================
int main()
{
  CommandLineArgs args(0, nullptr);
  CubDebugExit(args.DeviceInit());

  std::cout << "=== Thread Sort Test (No Thrust) ===" << std::endl;

  bool all_pass = true;
  if (!TestType<std::uint32_t, std::uint32_t>()) all_pass = false;
  if (!TestType<std::uint32_t, std::uint64_t>()) all_pass = false;

  std::cout << "\n=== All tests " << (all_pass ? "PASS" : "FAIL") << " ===" << std::endl;
  return all_pass ? 0 : 1;
}
