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

// No-thrust version of test_warp_merge_sort.cu for MUSA platform testing

#define CUB_STDERR

#include <stdio.h>
#include <limits>
#include <typeinfo>
#include <memory>
#include <vector>
#include <algorithm>
#include <random>

#include <cub/util_allocator.cuh>
#include <cub/warp/warp_merge_sort.cuh>

#include <musa_runtime.h>

#include "test_util.h"

using namespace cub;

struct CustomType
{
  std::uint8_t key;
  std::uint64_t count;

  __device__ __host__ CustomType()
    : key(0)
    , count(0)
  {}

  __device__ __host__ CustomType(std::uint64_t value)
    : key(static_cast<std::uint8_t>(value))
    , count(value)
  {}

  __device__ __host__ void operator=(std::uint64_t value)
  {
    key = static_cast<std::uint8_t>(value);
    count = value;
  }
};


struct CustomLess
{
  template <typename DataType>
  __device__ bool operator()(DataType &lhs, DataType &rhs)
  {
    return lhs < rhs;
  }

  __device__ bool operator()(CustomType &lhs, CustomType &rhs)
  {
    return lhs.key < rhs.key;
  }
};

//==============================================================================
// Kernel for sorting data
template <
  typename DataType,
  unsigned int ThreadsInBlock,
  unsigned int ThreadsInWarp,
  unsigned int ItemsPerThread,
  bool Stable = false>
__global__ void WarpMergeSortTestKernel(unsigned int valid_segments,
                                        DataType *data,
                                        const unsigned int *segment_sizes)
{
  using WarpMergeSortT =
    cub::WarpMergeSort<DataType, ItemsPerThread, ThreadsInWarp>;

  constexpr unsigned int WarpsInBlock = ThreadsInBlock / ThreadsInWarp;
  const unsigned int segment_id = threadIdx.x / ThreadsInWarp;

  if (segment_id >= valid_segments)
  {
    return;
  }

  __shared__ typename WarpMergeSortT::TempStorage temp_storage[WarpsInBlock];
  WarpMergeSortT warp_sort(temp_storage[segment_id]);

  DataType thread_data[ItemsPerThread];

  const unsigned int thread_offset = ThreadsInWarp * ItemsPerThread * segment_id
                                   + warp_sort.get_linear_tid() * ItemsPerThread;
  const unsigned int valid_items = segment_sizes[segment_id];

  for (unsigned int item = 0; item < ItemsPerThread; item++)
  {
    const unsigned int idx = thread_offset + item;
    thread_data[item] = item < valid_items ? data[idx] : DataType();
  }
  WARP_SYNC(warp_sort.get_member_mask());

  const DataType oob_default =
    static_cast<std::uint64_t>(ThreadsInBlock * ItemsPerThread + 1);

  if (Stable)
  {
    if (valid_items == ThreadsInBlock * ItemsPerThread)
    {
      warp_sort.StableSort(thread_data, CustomLess());
    }
    else
    {
      warp_sort.StableSort(thread_data, CustomLess(), valid_items, oob_default);
    }
  }
  else
  {
    if (valid_items == ThreadsInBlock * ItemsPerThread)
    {
      warp_sort.Sort(thread_data, CustomLess());
    }
    else
    {
      warp_sort.Sort(thread_data, CustomLess(), valid_items, oob_default);
    }
  }

  for (unsigned int item = 0; item < ItemsPerThread; item++)
  {
    const unsigned int idx = thread_offset + item;

    if (item >= valid_items)
      break;

    data[idx] = thread_data[item];
  }
}

//==============================================================================
// Host function to run warp sort
template<
  typename DataType,
  unsigned int ThreadsInBlock,
  unsigned int ThreadsInWarp,
  unsigned int ItemsPerThread,
  bool Stable>
void RunWarpMergeSortTest(unsigned int valid_segments,
                          DataType *d_data,
                          const unsigned int *d_segment_sizes)
{
  WarpMergeSortTestKernel<DataType,
                          ThreadsInBlock,
                          ThreadsInWarp,
                          ItemsPerThread,
                          Stable>
    <<<1, ThreadsInBlock>>>(valid_segments, d_data, d_segment_sizes);

  CubDebugExit(musaPeekAtLastError());
  CubDebugExit(musaDeviceSynchronize());
}

//==============================================================================
// Verify sorted result
template <typename DataType, unsigned int ThreadsInWarp, unsigned int ItemsPerThread>
bool VerifyResult(const std::vector<DataType>& h_data,
                  const std::vector<unsigned int>& h_segment_sizes,
                  unsigned int valid_segments)
{
  constexpr unsigned int max_segment_size = ThreadsInWarp * ItemsPerThread;

  for (unsigned int segment_id = 0; segment_id < valid_segments; segment_id++)
  {
    unsigned int segment_size = h_segment_sizes[segment_id];

    for (unsigned int i = 0; i < segment_size; i++)
    {
      const auto actual_value = h_data[max_segment_size * segment_id + i];
      const auto expected_value = static_cast<DataType>(i);

      if (actual_value != expected_value)
      {
        return false;
      }
    }
  }

  return true;
}

//==============================================================================
// Test function
template <
  typename DataType,
  unsigned int ThreadsInBlock,
  unsigned int ThreadsInWarp,
  unsigned int ItemsPerThread,
  bool Stable>
bool Test(unsigned int valid_segments,
          std::mt19937& rng,
          const std::vector<unsigned int>& h_segment_sizes)
{
  constexpr unsigned int max_segments = ThreadsInBlock / ThreadsInWarp;
  constexpr unsigned int max_segment_size = ThreadsInWarp * ItemsPerThread;
  constexpr unsigned int total_size = max_segments * max_segment_size;

  // Prepare host data
  std::vector<DataType> h_data(total_size, DataType{});
  
  for (unsigned int segment_id = 0; segment_id < valid_segments; segment_id++)
  {
    const unsigned int segment_offset = max_segment_size * segment_id;
    const unsigned int segment_size = h_segment_sizes[segment_id];

    // Fill with sequence
    for (unsigned int i = 0; i < segment_size; i++)
    {
      h_data[segment_offset + i] = static_cast<DataType>(i);
    }

    // Shuffle the segment
    std::shuffle(h_data.begin() + segment_offset, 
                 h_data.begin() + segment_offset + segment_size, rng);
  }

  // Allocate device memory
  DataType* d_data = nullptr;
  unsigned int* d_segment_sizes = nullptr;

  musaMalloc(&d_data, total_size * sizeof(DataType));
  musaMalloc(&d_segment_sizes, h_segment_sizes.size() * sizeof(unsigned int));

  // Copy to device
  musaMemcpy(d_data, h_data.data(), total_size * sizeof(DataType), musaMemcpyHostToDevice);
  musaMemcpy(d_segment_sizes, h_segment_sizes.data(), 
             h_segment_sizes.size() * sizeof(unsigned int), musaMemcpyHostToDevice);

  // Run sort
  RunWarpMergeSortTest<DataType, ThreadsInBlock, ThreadsInWarp, ItemsPerThread, Stable>(
    valid_segments, d_data, d_segment_sizes);

  // Copy back
  musaMemcpy(h_data.data(), d_data, total_size * sizeof(DataType), musaMemcpyDeviceToHost);

  // Cleanup
  musaFree(d_data);
  musaFree(d_segment_sizes);

  // Verify
  return VerifyResult<DataType, ThreadsInWarp, ItemsPerThread>(h_data, h_segment_sizes, valid_segments);
}

//==============================================================================
// Generate segment sizes
std::vector<unsigned int> GenerateSegmentSizes(unsigned int max_segment_size, unsigned int max_segments)
{
  std::vector<unsigned int> sizes;
  for (unsigned int i = 0; i < max_segment_size; i++)
  {
    sizes.push_back(i + 1);
  }
  // Replicate for each segment
  std::vector<unsigned int> result;
  for (unsigned int seg = 0; seg < max_segments; seg++)
  {
    result.insert(result.end(), sizes.begin(), sizes.end());
  }
  return result;
}

//==============================================================================
// Test wrapper for key-only sort
template <
  typename KeyType,
  unsigned int ThreadsInBlock,
  unsigned int ThreadsInWarp,
  unsigned int ItemsPerThread,
  bool Stable>
bool TestKeyType(std::mt19937& rng)
{
  constexpr unsigned int max_segments = ThreadsInBlock / ThreadsInWarp;
  constexpr unsigned int max_segment_size = ThreadsInWarp * ItemsPerThread;

  auto h_segment_sizes = GenerateSegmentSizes(max_segment_size, max_segments);
  std::shuffle(h_segment_sizes.begin(), h_segment_sizes.end(), rng);

  bool all_pass = true;
  for (unsigned int valid_segments = 1; valid_segments < max_segments; valid_segments += 3)
  {
    // Truncate segment sizes for current valid_segments
    std::vector<unsigned int> current_sizes(h_segment_sizes.begin(), 
                                            h_segment_sizes.begin() + valid_segments * max_segment_size);
    
    if (!Test<KeyType, ThreadsInBlock, ThreadsInWarp, ItemsPerThread, Stable>(
          valid_segments, rng, current_sizes))
    {
      all_pass = false;
      break;
    }
  }

  return all_pass;
}

//==============================================================================
// Test for specific configuration
template <unsigned int ThreadsInBlock, unsigned int ThreadsInWarp, unsigned int ItemsPerThread>
bool TestConfig(std::mt19937& rng)
{
  bool all_pass = true;

  std::cout << "  Testing " << ThreadsInBlock << " threads, " << ThreadsInWarp 
            << " warp, " << ItemsPerThread << " items/thread: ";

  if (!TestKeyType<std::int32_t, ThreadsInBlock, ThreadsInWarp, ItemsPerThread, false>(rng))
    all_pass = false;
  if (!TestKeyType<std::int64_t, ThreadsInBlock, ThreadsInWarp, ItemsPerThread, false>(rng))
    all_pass = false;

  std::cout << (all_pass ? "PASS" : "FAIL") << std::endl;
  return all_pass;
}

//==============================================================================
int main(int argc, char** argv)
{
  CommandLineArgs args(argc, argv);
  CubDebugExit(args.DeviceInit());

  std::cout << "=== Warp Merge Sort Test (No Thrust) ===" << std::endl;

  std::mt19937 rng(42);
  bool all_pass = true;

  // Test with different warp sizes and items per thread
  if (!TestConfig<32, 32, 2>(rng)) all_pass = false;
  if (!TestConfig<64, 32, 2>(rng)) all_pass = false;
  if (!TestConfig<128, 32, 2>(rng)) all_pass = false;

  if (!TestConfig<32, 32, 7>(rng)) all_pass = false;
  if (!TestConfig<64, 32, 7>(rng)) all_pass = false;
  if (!TestConfig<128, 32, 7>(rng)) all_pass = false;

  std::cout << "\n=== All tests " << (all_pass ? "PASS" : "FAIL") << " ===" << std::endl;
  return all_pass ? 0 : 1;
}
