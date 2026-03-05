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

#include "test_util.h"
#include "cub/warp/warp_exchange.cuh"

#include <algorithm>

template <typename InputT,
          typename OutputT,
          int LogicalWarpThreads,
          int ItemsPerThread,
          int BlockThreads,
          typename ActionT>
__global__ void kernel(const InputT *input_data,
                       OutputT *output_data,
                       ActionT action,
                       cub::Int2Type<true> /* same_type */)
{
  using WarpExchangeT =
    cub::WarpExchange<InputT, ItemsPerThread, LogicalWarpThreads>;

  constexpr int tile_size = ItemsPerThread * LogicalWarpThreads;
  constexpr int warps_per_block = BlockThreads / LogicalWarpThreads;
  __shared__ typename WarpExchangeT::TempStorage temp_storage[warps_per_block];

  const int warp_id = threadIdx.x / LogicalWarpThreads;
  const int lane_id = threadIdx.x % LogicalWarpThreads;
  WarpExchangeT exchange(temp_storage[warp_id]);

  InputT input[ItemsPerThread];

  input_data += warp_id * tile_size;
  output_data += warp_id * tile_size;

  for (int item = 0; item < ItemsPerThread; item++)
  {
    input[item] = input_data[lane_id * ItemsPerThread + item];
  }

  action(input, input, exchange);

  for (int item = 0; item < ItemsPerThread; item++)
  {
    output_data[lane_id * ItemsPerThread + item] = input[item];
  }
}

template <typename InputT,
          typename OutputT,
          int LogicalWarpThreads,
          int ItemsPerThread,
          int BlockThreads,
          typename ActionT>
__global__ void kernel(const InputT *input_data,
                       OutputT *output_data,
                       ActionT action,
                       cub::Int2Type<false> /* different_types */)
{
  using WarpExchangeT =
  cub::WarpExchange<InputT, ItemsPerThread, LogicalWarpThreads>;

  constexpr int tile_size = ItemsPerThread * LogicalWarpThreads;
  constexpr int warps_per_block = BlockThreads / LogicalWarpThreads;
  __shared__ typename WarpExchangeT::TempStorage temp_storage[warps_per_block];

  const int warp_id = threadIdx.x / LogicalWarpThreads;
  const int lane_id = threadIdx.x % LogicalWarpThreads;
  WarpExchangeT exchange(temp_storage[warp_id]);

  InputT input[ItemsPerThread];
  OutputT output[ItemsPerThread];

  input_data += warp_id * tile_size;
  output_data += warp_id * tile_size;

  for (int item = 0; item < ItemsPerThread; item++)
  {
    input[item] = input_data[lane_id * ItemsPerThread + item];
  }

  action(input, output, exchange);

  for (int item = 0; item < ItemsPerThread; item++)
  {
    output_data[lane_id * ItemsPerThread + item] = output[item];
  }
}

struct StripedToBlocked
{
  template <typename InputT,
            typename OutputT,
            int LogicalWarpThreads,
            int ItemsPerThread,
            int ITEMS_PER_THREAD>
  __device__ void operator()(
    InputT (&input)[ITEMS_PER_THREAD],
    OutputT (&output)[ITEMS_PER_THREAD],
    cub::WarpExchange<InputT, ItemsPerThread, LogicalWarpThreads> &exchange)
  {
    exchange.StripedToBlocked(input, output);
  }
};

struct BlockedToStriped
{
  template <typename InputT,
            typename OutputT,
            int LogicalWarpThreads,
            int ItemsPerThread,
            int ITEMS_PER_THREAD>
  __device__ void operator()(
    InputT (&input)[ITEMS_PER_THREAD],
    OutputT (&output)[ITEMS_PER_THREAD],
    cub::WarpExchange<InputT, ItemsPerThread, LogicalWarpThreads> &exchange)
  {
    exchange.BlockedToStriped(input, output);
  }
};


template <typename T>
bool Compare(
  const CubDeviceVector<T> &lhs,
  const CubDeviceVector<T> &rhs)
{
  return CubCompare(lhs, rhs);
}


template <typename InputT,
          typename OutputT,
          int LogicalWarpThreads,
          int ItemsPerThread,
          int BlockThreads>
void TestStripedToBlocked(CubDeviceVector<InputT> &input,
                          CubDeviceVector<OutputT> &output)
{
  cub_fill(output.begin(), output.end(), OutputT{0});

  std::vector<InputT> h_input(input.size());
  FillStriped<LogicalWarpThreads, ItemsPerThread, BlockThreads>(
    h_input.begin());

  input = h_input;

  kernel<InputT, OutputT, LogicalWarpThreads, ItemsPerThread, BlockThreads>
    <<<1, BlockThreads>>>(input.data(),
                          output.data(),
                          StripedToBlocked{},
                          cub::Int2Type<std::is_same<InputT, OutputT>::value>{});
  musaDeviceSynchronize();

  CubDeviceVector<OutputT> expected_output(output.size());
  cub_sequence(expected_output.begin(), expected_output.end());

  AssertTrue(Compare(expected_output, output));
}

template <typename InputT,
          typename OutputT,
          int LogicalWarpThreads,
          int ItemsPerThread,
          int BlockThreads>
void TestBlockedToStriped(CubDeviceVector<InputT> &input,
                          CubDeviceVector<OutputT> &output)
{
  cub_fill(output.begin(), output.end(), OutputT{0});

  std::vector<OutputT> expected_output(input.size());
  FillStriped<LogicalWarpThreads, ItemsPerThread, BlockThreads>(
    expected_output.begin());

  cub_sequence(input.begin(), input.end());

  kernel<InputT, OutputT, LogicalWarpThreads, ItemsPerThread, BlockThreads>
    <<<1, BlockThreads>>>(input.data(),
                          output.data(),
                          BlockedToStriped{},
                          cub::Int2Type<std::is_same<InputT, OutputT>::value>{});
  musaDeviceSynchronize();

  CubDeviceVector<OutputT> d_expected_output(expected_output);
  AssertTrue(Compare(d_expected_output, output));
}

template <typename InputT,
          typename OutputT,
          int LogicalWarpThreads,
          int ItemsPerThread,
          int BlockThreads>
__global__ void scatter_kernel(const InputT *input_data,
                               OutputT *output_data,
                               cub::Int2Type<true> /* same_type */)
{
  using WarpExchangeT =
    cub::WarpExchange<InputT, ItemsPerThread, LogicalWarpThreads>;

  constexpr int tile_size       = ItemsPerThread * LogicalWarpThreads;
  constexpr int warps_per_block = BlockThreads / LogicalWarpThreads;
  __shared__ typename WarpExchangeT::TempStorage temp_storage[warps_per_block];

  const int warp_id = threadIdx.x / LogicalWarpThreads;
  const int lane_id = threadIdx.x % LogicalWarpThreads;
  WarpExchangeT exchange(temp_storage[warp_id]);

  InputT input[ItemsPerThread];

  // Reverse data
  int ranks[ItemsPerThread];

  input_data += warp_id * tile_size;
  output_data += warp_id * tile_size;

  for (int item = 0; item < ItemsPerThread; item++)
  {
    const auto item_idx = lane_id * ItemsPerThread + item;
    input[item] = input_data[item_idx];
    ranks[item] = tile_size - 1 - item_idx;
  }

  exchange.ScatterToStriped(input, ranks);

  // Striped to blocked
  for (int item = 0; item < ItemsPerThread; item++)
  {
    output_data[item * LogicalWarpThreads + lane_id] = input[item];
  }
}

template <typename InputT,
          typename OutputT,
          int LogicalWarpThreads,
          int ItemsPerThread,
          int BlockThreads>
__global__ void scatter_kernel(const InputT *input_data,
                               OutputT *output_data,
                               cub::Int2Type<false> /* different_types */)
{
  using WarpExchangeT =
    cub::WarpExchange<InputT, ItemsPerThread, LogicalWarpThreads>;

  constexpr int tile_size       = ItemsPerThread * LogicalWarpThreads;
  constexpr int warps_per_block = BlockThreads / LogicalWarpThreads;
  __shared__ typename WarpExchangeT::TempStorage temp_storage[warps_per_block];

  const int warp_id = threadIdx.x / LogicalWarpThreads;
  const int lane_id = threadIdx.x % LogicalWarpThreads;
  WarpExchangeT exchange(temp_storage[warp_id]);

  InputT input[ItemsPerThread];
  OutputT output[ItemsPerThread];

  // Reverse data
  int ranks[ItemsPerThread];

  input_data += warp_id * tile_size;
  output_data += warp_id * tile_size;

  for (int item = 0; item < ItemsPerThread; item++)
  {
    const auto item_idx = lane_id * ItemsPerThread + item;
    input[item] = input_data[item_idx];
    ranks[item] = tile_size - 1 - item_idx;
  }

  exchange.ScatterToStriped(input, output, ranks);

  // Striped to blocked
  for (int item = 0; item < ItemsPerThread; item++)
  {
    output_data[item * LogicalWarpThreads + lane_id] = output[item];
  }
}

template <typename InputT,
          typename OutputT,
          int LogicalWarpThreads,
          int ItemsPerThread,
          int BlockThreads>
void TestScatterToStriped(CubDeviceVector<InputT> &input,
                          CubDeviceVector<OutputT> &output)
{
  cub_fill(output.begin(), output.end(), OutputT{0});
  cub_sequence(input.begin(), input.end());

  scatter_kernel<InputT, OutputT, LogicalWarpThreads, ItemsPerThread, BlockThreads>
    <<<1, BlockThreads>>>(input.data(),
                          output.data(),
                          cub::Int2Type<std::is_same<InputT, OutputT>::value>{});

  // Copy input to expected_output for reversal
  std::vector<InputT> h_input = input.to_host();
  std::vector<OutputT> h_expected(h_input.size());
  for (std::size_t i = 0; i < h_input.size(); i++)
  {
    h_expected[i] = static_cast<OutputT>(h_input[i]);
  }
  CubDeviceVector<OutputT> d_expected_output(h_expected);

  constexpr int tile_size = LogicalWarpThreads * ItemsPerThread;

  for (int warp_id = 0; warp_id < BlockThreads / LogicalWarpThreads; warp_id++)
  {
    const int warp_data_begin = tile_size * warp_id;
    const int warp_data_end = warp_data_begin + tile_size;
    CubReverse(d_expected_output, warp_data_begin, warp_data_end);
  }

  AssertTrue(Compare(d_expected_output, output));
}

template <typename InputT,
          typename OutputT,
          int LogicalWarpThreads,
          int ItemsPerThread,
          int BlockThreads>
void Test()
{
  static_assert(BlockThreads % LogicalWarpThreads == 0,
                "BlockThreads must be a multiple of LogicalWarpThreads");

  const int warps_in_block = BlockThreads / LogicalWarpThreads;
  const int items_per_warp = LogicalWarpThreads * ItemsPerThread;
  const int items_per_block = items_per_warp * warps_in_block;

  CubDeviceVector<InputT> input(items_per_block);
  CubDeviceVector<OutputT> output(items_per_block);

  TestStripedToBlocked<InputT,
                       OutputT,
                       LogicalWarpThreads,
                       ItemsPerThread,
                       BlockThreads>(input, output);

  TestBlockedToStriped<InputT,
                       OutputT,
                       LogicalWarpThreads,
                       ItemsPerThread,
                       BlockThreads>(input, output);

  TestScatterToStriped<InputT,
                       OutputT,
                       LogicalWarpThreads,
                       ItemsPerThread,
                       BlockThreads>(input, output);
}

template <int WarpThreads,
          int ItemsPerThread,
          int BlockThreads>
void Test()
{
  Test<std::uint16_t, std::uint32_t, WarpThreads, ItemsPerThread, BlockThreads>();
  Test<std::uint32_t, std::uint32_t, WarpThreads, ItemsPerThread, BlockThreads>();
  Test<std::uint64_t, std::uint32_t, WarpThreads, ItemsPerThread, BlockThreads>();
}

template <int LogicalWarpThreads,
          int ItemsPerThread>
void Test()
{
  Test<LogicalWarpThreads, ItemsPerThread, 128>();
  Test<LogicalWarpThreads, ItemsPerThread, 256>();
}

template <int LogicalWarpThreads>
void Test()
{
  Test<LogicalWarpThreads, 1>();
  Test<LogicalWarpThreads, 4>();
  Test<LogicalWarpThreads, 7>();
}

int main(int argc, char** argv)
{
  CommandLineArgs args(argc, argv);

  // Initialize device
  CubDebugExit(args.DeviceInit());

  Test<4>();
  Test<16>();
  Test<32>();

  return 0;
}
