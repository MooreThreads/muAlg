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

// Ensure printing of MUSA runtime errors to console
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


struct CustomLess
{
  template <typename DataType>
  __device__ __host__ bool operator()(const DataType &lhs, const DataType &rhs) const
  {
    return lhs < rhs;
  }
};


template <typename T>
void FillSequence(std::vector<T>& data)
{
  for (std::size_t i = 0; i < data.size(); ++i)
  {
    data[i] = static_cast<T>(i);
  }
}


template <typename T>
void ShuffleData(std::vector<T>& data, std::mt19937& rng)
{
  std::shuffle(data.begin(), data.end(), rng);
}


template <typename T>
bool CheckSorted(const T* data, std::size_t num_items)
{
  for (std::size_t i = 1; i < num_items; ++i)
  {
    if (data[i] < data[i - 1])
    {
      printf("  Sort error at index %zu: %lld < %lld\n",
             i, (long long)data[i], (long long)data[i - 1]);
      return false;
    }
  }
  return true;
}


template <typename KeyT, typename ValueT>
void TestSortPairs(std::size_t num_items, std::mt19937& rng)
{
  printf("  TestSortPairs<Key=%s, Value=%s> %zu elements: ",
         typeid(KeyT).name(), typeid(ValueT).name(), num_items);

  std::vector<KeyT> h_keys(num_items);
  std::vector<ValueT> h_values(num_items);

  FillSequence(h_values);
  ShuffleData(h_values, rng);
  for (std::size_t i = 0; i < num_items; ++i)
  {
    h_keys[i] = static_cast<KeyT>(h_values[i]);
  }

  KeyT* d_keys = nullptr;
  ValueT* d_values = nullptr;
  musaMalloc(&d_keys, num_items * sizeof(KeyT));
  musaMalloc(&d_values, num_items * sizeof(ValueT));
  musaMemcpy(d_keys, h_keys.data(), num_items * sizeof(KeyT), musaMemcpyHostToDevice);
  musaMemcpy(d_values, h_values.data(), num_items * sizeof(ValueT), musaMemcpyHostToDevice);

  void* d_temp_storage = nullptr;
  std::size_t temp_storage_bytes = 0;
  CubDebugExit(cub::DeviceMergeSort::SortPairs(
    d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, CustomLess()));

  musaMalloc(&d_temp_storage, temp_storage_bytes);
  CubDebugExit(cub::DeviceMergeSort::SortPairs(
    d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, CustomLess()));

  musaMemcpy(h_keys.data(), d_keys, num_items * sizeof(KeyT), musaMemcpyDeviceToHost);
  musaMemcpy(h_values.data(), d_values, num_items * sizeof(ValueT), musaMemcpyDeviceToHost);

  bool pass = CheckSorted(h_values.data(), num_items);
  printf("%s\n", pass ? "PASS" : "FAIL");

  musaFree(d_temp_storage);
  musaFree(d_keys);
  musaFree(d_values);
}


template <typename KeyT>
void TestSortKeys(std::size_t num_items, std::mt19937& rng)
{
  printf("  TestSortKeys<Key=%s> %zu elements: ", typeid(KeyT).name(), num_items);

  std::vector<KeyT> h_keys(num_items);
  FillSequence(h_keys);
  ShuffleData(h_keys, rng);

  KeyT* d_keys = nullptr;
  musaMalloc(&d_keys, num_items * sizeof(KeyT));
  musaMemcpy(d_keys, h_keys.data(), num_items * sizeof(KeyT), musaMemcpyHostToDevice);

  void* d_temp_storage = nullptr;
  std::size_t temp_storage_bytes = 0;
  CubDebugExit(cub::DeviceMergeSort::SortKeys(
    d_temp_storage, temp_storage_bytes, d_keys, num_items, CustomLess()));

  musaMalloc(&d_temp_storage, temp_storage_bytes);
  CubDebugExit(cub::DeviceMergeSort::SortKeys(
    d_temp_storage, temp_storage_bytes, d_keys, num_items, CustomLess()));

  musaMemcpy(h_keys.data(), d_keys, num_items * sizeof(KeyT), musaMemcpyDeviceToHost);

  bool pass = CheckSorted(h_keys.data(), num_items);
  printf("%s\n", pass ? "PASS" : "FAIL");

  musaFree(d_temp_storage);
  musaFree(d_keys);
}


template <typename KeyT, typename ValueT>
void TestSortPairsCopy(std::size_t num_items, std::mt19937& rng)
{
  printf("  TestSortPairsCopy<Key=%s, Value=%s> %zu elements: ",
         typeid(KeyT).name(), typeid(ValueT).name(), num_items);

  std::vector<KeyT> h_keys_in(num_items);
  std::vector<ValueT> h_values_in(num_items);
  std::vector<KeyT> h_keys_out(num_items);
  std::vector<ValueT> h_values_out(num_items);

  FillSequence(h_values_in);
  ShuffleData(h_values_in, rng);
  for (std::size_t i = 0; i < num_items; ++i)
  {
    h_keys_in[i] = static_cast<KeyT>(h_values_in[i]);
  }

  KeyT* d_keys_in = nullptr;
  ValueT* d_values_in = nullptr;
  KeyT* d_keys_out = nullptr;
  ValueT* d_values_out = nullptr;
  musaMalloc(&d_keys_in, num_items * sizeof(KeyT));
  musaMalloc(&d_values_in, num_items * sizeof(ValueT));
  musaMalloc(&d_keys_out, num_items * sizeof(KeyT));
  musaMalloc(&d_values_out, num_items * sizeof(ValueT));
  musaMemcpy(d_keys_in, h_keys_in.data(), num_items * sizeof(KeyT), musaMemcpyHostToDevice);
  musaMemcpy(d_values_in, h_values_in.data(), num_items * sizeof(ValueT), musaMemcpyHostToDevice);

  void* d_temp_storage = nullptr;
  std::size_t temp_storage_bytes = 0;
  CubDebugExit(cub::DeviceMergeSort::SortPairsCopy(
    d_temp_storage, temp_storage_bytes,
    d_keys_in, d_values_in, d_keys_out, d_values_out,
    num_items, CustomLess()));

  musaMalloc(&d_temp_storage, temp_storage_bytes);
  CubDebugExit(cub::DeviceMergeSort::SortPairsCopy(
    d_temp_storage, temp_storage_bytes,
    d_keys_in, d_values_in, d_keys_out, d_values_out,
    num_items, CustomLess()));

  musaMemcpy(h_keys_out.data(), d_keys_out, num_items * sizeof(KeyT), musaMemcpyDeviceToHost);
  musaMemcpy(h_values_out.data(), d_values_out, num_items * sizeof(ValueT), musaMemcpyDeviceToHost);

  bool pass = CheckSorted(h_values_out.data(), num_items);
  printf("%s\n", pass ? "PASS" : "FAIL");

  musaFree(d_temp_storage);
  musaFree(d_keys_in);
  musaFree(d_values_in);
  musaFree(d_keys_out);
  musaFree(d_values_out);
}


template <typename KeyT>
void TestSortKeysCopy(std::size_t num_items, std::mt19937& rng)
{
  printf("  TestSortKeysCopy<Key=%s> %zu elements: ", typeid(KeyT).name(), num_items);

  std::vector<KeyT> h_keys_in(num_items);
  std::vector<KeyT> h_keys_out(num_items);
  FillSequence(h_keys_in);
  ShuffleData(h_keys_in, rng);

  KeyT* d_keys_in = nullptr;
  KeyT* d_keys_out = nullptr;
  musaMalloc(&d_keys_in, num_items * sizeof(KeyT));
  musaMalloc(&d_keys_out, num_items * sizeof(KeyT));
  musaMemcpy(d_keys_in, h_keys_in.data(), num_items * sizeof(KeyT), musaMemcpyHostToDevice);

  void* d_temp_storage = nullptr;
  std::size_t temp_storage_bytes = 0;
  CubDebugExit(cub::DeviceMergeSort::SortKeysCopy(
    d_temp_storage, temp_storage_bytes,
    d_keys_in, d_keys_out, num_items, CustomLess()));

  musaMalloc(&d_temp_storage, temp_storage_bytes);
  CubDebugExit(cub::DeviceMergeSort::SortKeysCopy(
    d_temp_storage, temp_storage_bytes,
    d_keys_in, d_keys_out, num_items, CustomLess()));

  musaMemcpy(h_keys_out.data(), d_keys_out, num_items * sizeof(KeyT), musaMemcpyDeviceToHost);

  bool pass = CheckSorted(h_keys_out.data(), num_items);
  printf("%s\n", pass ? "PASS" : "FAIL");

  musaFree(d_temp_storage);
  musaFree(d_keys_in);
  musaFree(d_keys_out);
}


template <typename KeyT, typename ValueT>
void TestStableSortPairs(std::size_t num_items, std::mt19937& rng)
{
  printf("  TestStableSortPairs<Key=%s, Value=%s> %zu elements: ",
         typeid(KeyT).name(), typeid(ValueT).name(), num_items);

  std::vector<KeyT> h_keys(num_items);
  std::vector<ValueT> h_values(num_items);

  FillSequence(h_values);
  ShuffleData(h_values, rng);
  for (std::size_t i = 0; i < num_items; ++i)
  {
    h_keys[i] = static_cast<KeyT>(h_values[i]);
  }

  KeyT* d_keys = nullptr;
  ValueT* d_values = nullptr;
  musaMalloc(&d_keys, num_items * sizeof(KeyT));
  musaMalloc(&d_values, num_items * sizeof(ValueT));
  musaMemcpy(d_keys, h_keys.data(), num_items * sizeof(KeyT), musaMemcpyHostToDevice);
  musaMemcpy(d_values, h_values.data(), num_items * sizeof(ValueT), musaMemcpyHostToDevice);

  void* d_temp_storage = nullptr;
  std::size_t temp_storage_bytes = 0;
  CubDebugExit(cub::DeviceMergeSort::StableSortPairs(
    d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, CustomLess()));

  musaMalloc(&d_temp_storage, temp_storage_bytes);
  CubDebugExit(cub::DeviceMergeSort::StableSortPairs(
    d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, CustomLess()));

  musaMemcpy(h_keys.data(), d_keys, num_items * sizeof(KeyT), musaMemcpyDeviceToHost);
  musaMemcpy(h_values.data(), d_values, num_items * sizeof(ValueT), musaMemcpyDeviceToHost);

  bool pass = CheckSorted(h_values.data(), num_items);
  printf("%s\n", pass ? "PASS" : "FAIL");

  musaFree(d_temp_storage);
  musaFree(d_keys);
  musaFree(d_values);
}


template <typename KeyT>
void TestStableSortKeys(std::size_t num_items, std::mt19937& rng)
{
  printf("  TestStableSortKeys<Key=%s> %zu elements: ", typeid(KeyT).name(), num_items);

  std::vector<KeyT> h_keys(num_items);
  FillSequence(h_keys);
  ShuffleData(h_keys, rng);

  KeyT* d_keys = nullptr;
  musaMalloc(&d_keys, num_items * sizeof(KeyT));
  musaMemcpy(d_keys, h_keys.data(), num_items * sizeof(KeyT), musaMemcpyHostToDevice);

  void* d_temp_storage = nullptr;
  std::size_t temp_storage_bytes = 0;
  CubDebugExit(cub::DeviceMergeSort::StableSortKeys(
    d_temp_storage, temp_storage_bytes, d_keys, num_items, CustomLess()));

  musaMalloc(&d_temp_storage, temp_storage_bytes);
  CubDebugExit(cub::DeviceMergeSort::StableSortKeys(
    d_temp_storage, temp_storage_bytes, d_keys, num_items, CustomLess()));

  musaMemcpy(h_keys.data(), d_keys, num_items * sizeof(KeyT), musaMemcpyDeviceToHost);

  bool pass = CheckSorted(h_keys.data(), num_items);
  printf("%s\n", pass ? "PASS" : "FAIL");

  musaFree(d_temp_storage);
  musaFree(d_keys);
}


template <typename ValueT>
void TestValueType(std::size_t num_items, std::mt19937& rng)
{
  printf("Testing ValueT=%s with %zu elements:\n", typeid(ValueT).name(), num_items);

  TestSortPairs<std::uint32_t, ValueT>(num_items, rng);
  TestSortPairs<std::uint64_t, ValueT>(num_items, rng);
  TestSortKeys<std::uint32_t>(num_items, rng);
  TestSortKeys<std::uint64_t>(num_items, rng);
  TestSortPairsCopy<std::uint32_t, ValueT>(num_items, rng);
  TestSortPairsCopy<std::uint64_t, ValueT>(num_items, rng);
  TestSortKeysCopy<std::uint32_t>(num_items, rng);
  TestSortKeysCopy<std::uint64_t>(num_items, rng);
  TestStableSortPairs<std::uint32_t, ValueT>(num_items, rng);
  TestStableSortPairs<std::uint64_t, ValueT>(num_items, rng);
  TestStableSortKeys<std::uint32_t>(num_items, rng);
  TestStableSortKeys<std::uint64_t>(num_items, rng);
}


int main(int argc, char** argv)
{
  CommandLineArgs args(argc, argv);

  CubDebugExit(args.DeviceInit());

  printf("=== Device Merge Sort Test (No Thrust) ===\n\n");

  std::mt19937 rng(12345);

  TestValueType<std::int32_t>(512, rng);
  TestValueType<std::int32_t>(1024, rng);
  TestValueType<std::int32_t>(4096, rng);
  TestValueType<std::int32_t>(65536, rng);
  TestValueType<std::int32_t>(1048576, rng);

  TestValueType<std::int64_t>(65536, rng);

  printf("\nAll tests completed.\n");
  return 0;
}
