/******************************************************************************
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

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <cub/cub.cuh>
#include <cub/util_allocator.cuh>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <random>

#include <musa_runtime.h>

#include "test_util.h"

using namespace cub;


template <typename T>
bool CheckSegmentSorted(const T* data, const int* offsets, int num_segments)
{
  for (int seg = 0; seg < num_segments; ++seg)
  {
    int begin = offsets[seg];
    int end = offsets[seg + 1];
    for (int i = begin + 1; i < end; ++i)
    {
      if (data[i] < data[i - 1])
      {
        printf("  Segment %d not sorted at index %d: %lld < %lld\n",
               seg, i, (long long)data[i], (long long)data[i - 1]);
        return false;
      }
    }
  }
  return true;
}


template <typename T>
bool CheckSegmentSortedDescending(const T* data, const int* offsets, int num_segments)
{
  for (int seg = 0; seg < num_segments; ++seg)
  {
    int begin = offsets[seg];
    int end = offsets[seg + 1];
    for (int i = begin + 1; i < end; ++i)
    {
      if (data[i] > data[i - 1])
      {
        printf("  Segment %d not sorted descending at index %d: %lld > %lld\n",
               seg, i, (long long)data[i], (long long)data[i - 1]);
        return false;
      }
    }
  }
  return true;
}


template <typename KeyT>
void TestSortKeys(int num_items, int num_segments, int segment_size, std::mt19937& rng)
{
  printf("  TestSortKeys<Key=%s> %d items, %d segments: ",
         typeid(KeyT).name(), num_items, num_segments);

  std::vector<KeyT> h_keys_in(num_items);
  std::vector<KeyT> h_keys_out(num_items);
  std::vector<int> h_offsets(num_segments + 1);

  for (int i = 0; i < num_items; ++i)
  {
    h_keys_in[i] = static_cast<KeyT>(i % segment_size);
  }
  std::shuffle(h_keys_in.begin(), h_keys_in.end(), rng);

  for (int i = 0; i <= num_segments; ++i)
  {
    h_offsets[i] = i * segment_size;
  }

  KeyT* d_keys_in = nullptr;
  KeyT* d_keys_out = nullptr;
  int* d_offsets = nullptr;
  musaMalloc(&d_keys_in, num_items * sizeof(KeyT));
  musaMalloc(&d_keys_out, num_items * sizeof(KeyT));
  musaMalloc(&d_offsets, (num_segments + 1) * sizeof(int));
  musaMemcpy(d_keys_in, h_keys_in.data(), num_items * sizeof(KeyT), musaMemcpyHostToDevice);
  musaMemcpy(d_offsets, h_offsets.data(), (num_segments + 1) * sizeof(int), musaMemcpyHostToDevice);

  void* d_temp_storage = nullptr;
  std::size_t temp_storage_bytes = 0;
  CubDebugExit(cub::DeviceSegmentedSort::SortKeys(
    d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items, num_segments,
    d_offsets, d_offsets + 1));

  musaMalloc(&d_temp_storage, temp_storage_bytes);
  CubDebugExit(cub::DeviceSegmentedSort::SortKeys(
    d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items, num_segments,
    d_offsets, d_offsets + 1));

  musaMemcpy(h_keys_out.data(), d_keys_out, num_items * sizeof(KeyT), musaMemcpyDeviceToHost);

  bool pass = CheckSegmentSorted(h_keys_out.data(), h_offsets.data(), num_segments);
  printf("%s\n", pass ? "PASS" : "FAIL");

  musaFree(d_temp_storage);
  musaFree(d_keys_in);
  musaFree(d_keys_out);
  musaFree(d_offsets);
}


template <typename KeyT>
void TestSortKeysDescending(int num_items, int num_segments, int segment_size, std::mt19937& rng)
{
  printf("  TestSortKeysDescending<Key=%s> %d items, %d segments: ",
         typeid(KeyT).name(), num_items, num_segments);

  std::vector<KeyT> h_keys_in(num_items);
  std::vector<KeyT> h_keys_out(num_items);
  std::vector<int> h_offsets(num_segments + 1);

  for (int i = 0; i < num_items; ++i)
  {
    h_keys_in[i] = static_cast<KeyT>(i % segment_size);
  }
  std::shuffle(h_keys_in.begin(), h_keys_in.end(), rng);

  for (int i = 0; i <= num_segments; ++i)
  {
    h_offsets[i] = i * segment_size;
  }

  KeyT* d_keys_in = nullptr;
  KeyT* d_keys_out = nullptr;
  int* d_offsets = nullptr;
  musaMalloc(&d_keys_in, num_items * sizeof(KeyT));
  musaMalloc(&d_keys_out, num_items * sizeof(KeyT));
  musaMalloc(&d_offsets, (num_segments + 1) * sizeof(int));
  musaMemcpy(d_keys_in, h_keys_in.data(), num_items * sizeof(KeyT), musaMemcpyHostToDevice);
  musaMemcpy(d_offsets, h_offsets.data(), (num_segments + 1) * sizeof(int), musaMemcpyHostToDevice);

  void* d_temp_storage = nullptr;
  std::size_t temp_storage_bytes = 0;
  CubDebugExit(cub::DeviceSegmentedSort::SortKeysDescending(
    d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items, num_segments,
    d_offsets, d_offsets + 1));

  musaMalloc(&d_temp_storage, temp_storage_bytes);
  CubDebugExit(cub::DeviceSegmentedSort::SortKeysDescending(
    d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items, num_segments,
    d_offsets, d_offsets + 1));

  musaMemcpy(h_keys_out.data(), d_keys_out, num_items * sizeof(KeyT), musaMemcpyDeviceToHost);

  bool pass = CheckSegmentSortedDescending(h_keys_out.data(), h_offsets.data(), num_segments);
  printf("%s\n", pass ? "PASS" : "FAIL");

  musaFree(d_temp_storage);
  musaFree(d_keys_in);
  musaFree(d_keys_out);
  musaFree(d_offsets);
}


template <typename KeyT, typename ValueT>
void TestSortPairs(int num_items, int num_segments, int segment_size, std::mt19937& rng)
{
  printf("  TestSortPairs<Key=%s, Value=%s> %d items, %d segments: ",
         typeid(KeyT).name(), typeid(ValueT).name(), num_items, num_segments);

  std::vector<KeyT> h_keys_in(num_items);
  std::vector<KeyT> h_keys_out(num_items);
  std::vector<ValueT> h_values_in(num_items);
  std::vector<ValueT> h_values_out(num_items);
  std::vector<int> h_offsets(num_segments + 1);

  for (int i = 0; i < num_items; ++i)
  {
    h_keys_in[i] = static_cast<KeyT>(i % segment_size);
    h_values_in[i] = static_cast<ValueT>(i);
  }
  std::shuffle(h_keys_in.begin(), h_keys_in.end(), rng);

  for (int i = 0; i <= num_segments; ++i)
  {
    h_offsets[i] = i * segment_size;
  }

  KeyT* d_keys_in = nullptr;
  KeyT* d_keys_out = nullptr;
  ValueT* d_values_in = nullptr;
  ValueT* d_values_out = nullptr;
  int* d_offsets = nullptr;
  musaMalloc(&d_keys_in, num_items * sizeof(KeyT));
  musaMalloc(&d_keys_out, num_items * sizeof(KeyT));
  musaMalloc(&d_values_in, num_items * sizeof(ValueT));
  musaMalloc(&d_values_out, num_items * sizeof(ValueT));
  musaMalloc(&d_offsets, (num_segments + 1) * sizeof(int));
  musaMemcpy(d_keys_in, h_keys_in.data(), num_items * sizeof(KeyT), musaMemcpyHostToDevice);
  musaMemcpy(d_values_in, h_values_in.data(), num_items * sizeof(ValueT), musaMemcpyHostToDevice);
  musaMemcpy(d_offsets, h_offsets.data(), (num_segments + 1) * sizeof(int), musaMemcpyHostToDevice);

  void* d_temp_storage = nullptr;
  std::size_t temp_storage_bytes = 0;
  CubDebugExit(cub::DeviceSegmentedSort::SortPairs(
    d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out,
    num_items, num_segments, d_offsets, d_offsets + 1));

  musaMalloc(&d_temp_storage, temp_storage_bytes);
  CubDebugExit(cub::DeviceSegmentedSort::SortPairs(
    d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out,
    num_items, num_segments, d_offsets, d_offsets + 1));

  musaMemcpy(h_keys_out.data(), d_keys_out, num_items * sizeof(KeyT), musaMemcpyDeviceToHost);

  bool pass = CheckSegmentSorted(h_keys_out.data(), h_offsets.data(), num_segments);
  printf("%s\n", pass ? "PASS" : "FAIL");

  musaFree(d_temp_storage);
  musaFree(d_keys_in);
  musaFree(d_keys_out);
  musaFree(d_values_in);
  musaFree(d_values_out);
  musaFree(d_offsets);
}


template <typename KeyT>
void TestKeyType(int segment_size, int num_segments, std::mt19937& rng)
{
  int num_items = segment_size * num_segments;

  TestSortKeys<KeyT>(num_items, num_segments, segment_size, rng);
  TestSortKeysDescending<KeyT>(num_items, num_segments, segment_size, rng);
  TestSortPairs<KeyT, std::int32_t>(num_items, num_segments, segment_size, rng);
}


int main(int argc, char** argv)
{
  CommandLineArgs args(argc, argv);

  CubDebugExit(args.DeviceInit());

  printf("=== Device Segmented Sort Test (No Thrust) ===\n\n");

  std::mt19937 rng(12345);

  printf("Testing small segments:\n");
  TestKeyType<std::int32_t>(1, 4, rng);
  TestKeyType<std::int32_t>(2, 4, rng);
  TestKeyType<std::int32_t>(4, 4, rng);
  TestKeyType<std::int32_t>(8, 4, rng);
  TestKeyType<std::int32_t>(16, 4, rng);
  TestKeyType<std::int32_t>(32, 4, rng);
  TestKeyType<std::int32_t>(64, 4, rng);
  TestKeyType<std::int32_t>(128, 4, rng);
  TestKeyType<std::int32_t>(256, 4, rng);
  TestKeyType<std::int32_t>(512, 4, rng);
  TestKeyType<std::int32_t>(1024, 4, rng);

  printf("\nTesting medium segments:\n");
  TestKeyType<std::int32_t>(1024, 16, rng);
  TestKeyType<std::int32_t>(4096, 16, rng);
  TestKeyType<std::int32_t>(8192, 16, rng);

  printf("\nTesting large segments:\n");
  TestKeyType<std::int32_t>(65536, 4, rng);

  printf("\nTesting int64:\n");
  TestKeyType<std::int64_t>(1024, 4, rng);

  printf("\nTesting uint8:\n");
  TestKeyType<std::uint8_t>(1024, 4, rng);

  printf("\nAll tests completed.\n");
  return 0;
}
