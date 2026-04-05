/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
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
#include <random>
#include <algorithm>

#include <musa_runtime.h>

#include "test_util.h"

using namespace cub;


template <typename T>
void InitializeUnique(int entropy_reduction, T* h_in, int num_items, int max_segment, std::mt19937& rng)
{
  std::uniform_int_distribution<int> dist(1, max_segment > 1 ? max_segment : 1);

  int key = 0;
  int i = 0;
  while (i < num_items)
  {
    int repeat = (max_segment < 0) ? num_items : (max_segment < 2) ? 1 : dist(rng);
    repeat = std::min(repeat, num_items - i);

    for (int j = 0; j < repeat; ++j)
    {
      h_in[i + j] = static_cast<T>(key);
    }
    i += repeat;
    key++;
  }
}


template <typename T>
int SolveUnique(const T* h_in, T* h_reference, int num_items)
{
  int num_selected = 0;
  if (num_items > 0)
  {
    h_reference[num_selected] = h_in[0];
    num_selected++;
  }

  for (int i = 1; i < num_items; ++i)
  {
    if (h_in[i] != h_in[i - 1])
    {
      h_reference[num_selected] = h_in[i];
      num_selected++;
    }
  }

  return num_selected;
}


template <typename T>
bool CheckUniqueResult(const T* h_out, const T* h_reference, int num_selected, int num_items)
{
  for (int i = 0; i < num_selected; ++i)
  {
    if (h_out[i] != h_reference[i])
    {
      printf("  Mismatch at index %d: got %lld, expected %lld\n",
             i, (long long)h_out[i], (long long)h_reference[i]);
      return false;
    }
  }
  return true;
}


template <typename T>
void TestUnique(int num_items, int max_segment, std::mt19937& rng)
{
  printf("  TestUnique<T=%s> %d items, max_segment=%d: ",
         typeid(T).name(), num_items, max_segment);

  std::vector<T> h_in(num_items);
  std::vector<T> h_reference(num_items);
  std::vector<T> h_out(num_items);

  InitializeUnique(0, h_in.data(), num_items, max_segment, rng);
  int num_selected = SolveUnique(h_in.data(), h_reference.data(), num_items);

  T* d_in = nullptr;
  T* d_out = nullptr;
  int* d_num_selected = nullptr;
  musaMalloc(&d_in, num_items * sizeof(T));
  musaMalloc(&d_out, num_items * sizeof(T));
  musaMalloc(&d_num_selected, sizeof(int));

  musaMemcpy(d_in, h_in.data(), num_items * sizeof(T), musaMemcpyHostToDevice);
  musaMemset(d_out, 0, num_items * sizeof(T));
  musaMemset(d_num_selected, 0, sizeof(int));

  void* d_temp_storage = nullptr;
  std::size_t temp_storage_bytes = 0;
  CubDebugExit(cub::DeviceSelect::Unique(
    d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected, num_items));

  musaMalloc(&d_temp_storage, temp_storage_bytes);
  CubDebugExit(cub::DeviceSelect::Unique(
    d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected, num_items));

  int h_num_selected = 0;
  musaMemcpy(h_out.data(), d_out, num_items * sizeof(T), musaMemcpyDeviceToHost);
  musaMemcpy(&h_num_selected, d_num_selected, sizeof(int), musaMemcpyDeviceToHost);

  bool data_ok = CheckUniqueResult(h_out.data(), h_reference.data(), num_selected, num_items);
  bool count_ok = (h_num_selected == num_selected);

  if (!count_ok)
  {
    printf("Count mismatch: got %d, expected %d. ", h_num_selected, num_selected);
  }

  printf("%s\n", (data_ok && count_ok) ? "PASS" : "FAIL");

  musaFree(d_temp_storage);
  musaFree(d_in);
  musaFree(d_out);
  musaFree(d_num_selected);
}


template <typename T>
void TestType(int num_items)
{
  std::mt19937 rng(12345);

  TestUnique<T>(num_items, 1, rng);
  TestUnique<T>(num_items, 2, rng);
  TestUnique<T>(num_items, 11, rng);
  TestUnique<T>(num_items, 121, rng);
}


int main(int argc, char** argv)
{
  CommandLineArgs args(argc, argv);

  CubDebugExit(args.DeviceInit());

  printf("=== Device Select Unique Test (No Thrust) ===\n\n");

  printf("Testing std::int32_t:\n");
  TestType<std::int32_t>(0);
  TestType<std::int32_t>(1);
  TestType<std::int32_t>(100);
  TestType<std::int32_t>(10000);
  TestType<std::int32_t>(100000);

  printf("\nTesting std::int64_t:\n");
  TestType<std::int64_t>(10000);

  printf("\nTesting std::uint8_t:\n");
  TestType<std::uint8_t>(10000);

  printf("\nAll tests completed.\n");
  return 0;
}
