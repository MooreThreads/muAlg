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

#include <cub/device/device_adjacent_difference.cuh>
#include <cub/util_allocator.cuh>

#include <cstdio>
#include <cstring>
#include <vector>

#include <musa_runtime.h>

#include "test_util.h"


using namespace cub;


template <typename T>
void GenerateTestSequence(T* data, std::size_t num_items)
{
  for (std::size_t i = 0; i < num_items; ++i)
  {
    data[i] = static_cast<T>(i);
  }
}


template <typename T>
bool CheckSubtractLeftResult(const T* data, std::size_t num_items)
{
  if (num_items <= 1)
  {
    return true;
  }
  // SubtractLeft: output[0] unchanged, output[i] = input[i] - input[i-1] = 1
  for (std::size_t i = 1; i < num_items; ++i)
  {
    T expected = static_cast<T>(1);
    if (data[i] != expected)
    {
      printf("  SubtractLeft mismatch at index %zu: got %lld, expected %lld\n",
             i, (long long)data[i], (long long)expected);
      return false;
    }
  }
  return true;
}


template <typename T>
bool CheckSubtractRightResult(const T* data, std::size_t num_items)
{
  if (num_items <= 1)
  {
    return true;
  }
  // SubtractRight: output[i] = input[i] - input[i+1] = -1 for i < num_items-1
  for (std::size_t i = 0; i < num_items - 1; ++i)
  {
    T expected = static_cast<T>(-1);
    if (data[i] != expected)
    {
      printf("  SubtractRight mismatch at index %zu: got %lld, expected %lld\n",
             i, (long long)data[i], (long long)expected);
      return false;
    }
  }
  return true;
}


template <typename T>
void TestSubtractLeft(std::size_t num_items)
{
  std::vector<T> h_data(num_items);
  GenerateTestSequence(h_data.data(), num_items);

  T* d_data = nullptr;
  musaMalloc(&d_data, num_items * sizeof(T));
  musaMemcpy(d_data, h_data.data(), num_items * sizeof(T), musaMemcpyHostToDevice);

  void* d_temp_storage = nullptr;
  std::size_t temp_storage_bytes = 0;
  CubDebugExit(cub::DeviceAdjacentDifference::SubtractLeft(
    d_temp_storage, temp_storage_bytes, d_data, num_items));

  musaMalloc(&d_temp_storage, temp_storage_bytes);
  CubDebugExit(cub::DeviceAdjacentDifference::SubtractLeft(
    d_temp_storage, temp_storage_bytes, d_data, num_items));

  musaMemcpy(h_data.data(), d_data, num_items * sizeof(T), musaMemcpyDeviceToHost);

  bool pass = CheckSubtractLeftResult(h_data.data(), num_items);
  printf("  SubtractLeft %zu elements: %s\n", num_items, pass ? "PASS" : "FAIL");

  musaFree(d_temp_storage);
  musaFree(d_data);
}


template <typename T>
void TestSubtractRight(std::size_t num_items)
{
  std::vector<T> h_data(num_items);
  GenerateTestSequence(h_data.data(), num_items);

  T* d_data = nullptr;
  musaMalloc(&d_data, num_items * sizeof(T));
  musaMemcpy(d_data, h_data.data(), num_items * sizeof(T), musaMemcpyHostToDevice);

  void* d_temp_storage = nullptr;
  std::size_t temp_storage_bytes = 0;
  CubDebugExit(cub::DeviceAdjacentDifference::SubtractRight(
    d_temp_storage, temp_storage_bytes, d_data, num_items));

  musaMalloc(&d_temp_storage, temp_storage_bytes);
  CubDebugExit(cub::DeviceAdjacentDifference::SubtractRight(
    d_temp_storage, temp_storage_bytes, d_data, num_items));

  musaMemcpy(h_data.data(), d_data, num_items * sizeof(T), musaMemcpyDeviceToHost);

  bool pass = CheckSubtractRightResult(h_data.data(), num_items);
  printf("  SubtractRight %zu elements: %s\n", num_items, pass ? "PASS" : "FAIL");

  musaFree(d_temp_storage);
  musaFree(d_data);
}


template <typename T>
void TestSubtractLeftCopy(std::size_t num_items)
{
  std::vector<T> h_input(num_items);
  GenerateTestSequence(h_input.data(), num_items);

  T* d_input = nullptr;
  T* d_output = nullptr;
  musaMalloc(&d_input, num_items * sizeof(T));
  musaMalloc(&d_output, num_items * sizeof(T));
  musaMemcpy(d_input, h_input.data(), num_items * sizeof(T), musaMemcpyHostToDevice);

  void* d_temp_storage = nullptr;
  std::size_t temp_storage_bytes = 0;
  CubDebugExit(cub::DeviceAdjacentDifference::SubtractLeftCopy(
    d_temp_storage, temp_storage_bytes, d_input, d_output, num_items));

  musaMalloc(&d_temp_storage, temp_storage_bytes);
  CubDebugExit(cub::DeviceAdjacentDifference::SubtractLeftCopy(
    d_temp_storage, temp_storage_bytes, d_input, d_output, num_items));

  std::vector<T> h_output(num_items);
  musaMemcpy(h_output.data(), d_output, num_items * sizeof(T), musaMemcpyDeviceToHost);

  bool pass = CheckSubtractLeftResult(h_output.data(), num_items);
  printf("  SubtractLeftCopy %zu elements: %s\n", num_items, pass ? "PASS" : "FAIL");

  musaFree(d_temp_storage);
  musaFree(d_input);
  musaFree(d_output);
}


template <typename T>
void TestSubtractRightCopy(std::size_t num_items)
{
  std::vector<T> h_input(num_items);
  GenerateTestSequence(h_input.data(), num_items);

  T* d_input = nullptr;
  T* d_output = nullptr;
  musaMalloc(&d_input, num_items * sizeof(T));
  musaMalloc(&d_output, num_items * sizeof(T));
  musaMemcpy(d_input, h_input.data(), num_items * sizeof(T), musaMemcpyHostToDevice);

  void* d_temp_storage = nullptr;
  std::size_t temp_storage_bytes = 0;
  CubDebugExit(cub::DeviceAdjacentDifference::SubtractRightCopy(
    d_temp_storage, temp_storage_bytes, d_input, d_output, num_items));

  musaMalloc(&d_temp_storage, temp_storage_bytes);
  CubDebugExit(cub::DeviceAdjacentDifference::SubtractRightCopy(
    d_temp_storage, temp_storage_bytes, d_input, d_output, num_items));

  std::vector<T> h_output(num_items);
  musaMemcpy(h_output.data(), d_output, num_items * sizeof(T), musaMemcpyDeviceToHost);

  bool pass = CheckSubtractRightResult(h_output.data(), num_items);
  printf("  SubtractRightCopy %zu elements: %s\n", num_items, pass ? "PASS" : "FAIL");

  musaFree(d_temp_storage);
  musaFree(d_input);
  musaFree(d_output);
}


template <typename T>
void TestType(std::size_t num_items)
{
  printf("Testing type %s with %zu elements:\n", typeid(T).name(), num_items);
  TestSubtractLeft<T>(num_items);
  TestSubtractRight<T>(num_items);
  TestSubtractLeftCopy<T>(num_items);
  TestSubtractRightCopy<T>(num_items);
}


int main(int argc, char** argv)
{
  CommandLineArgs args(argc, argv);

  CubDebugExit(args.DeviceInit());

  printf("=== Device Adjacent Difference Test (No Thrust) ===\n\n");

  TestType<std::int32_t>(0);
  TestType<std::int32_t>(1);
  TestType<std::int32_t>(2);
  TestType<std::int32_t>(4);
  TestType<std::int32_t>(32);
  TestType<std::int32_t>(64);
  TestType<std::int32_t>(128);
  TestType<std::int32_t>(256);
  TestType<std::int32_t>(512);
  TestType<std::int32_t>(1024);
  TestType<std::int32_t>(4096);
  TestType<std::int32_t>(65536);
  TestType<std::int32_t>(1048576);

  TestType<std::uint32_t>(65536);
  TestType<std::uint64_t>(65536);
  TestType<std::int64_t>(65536);

  printf("\nAll tests completed.\n");
  return 0;
}
