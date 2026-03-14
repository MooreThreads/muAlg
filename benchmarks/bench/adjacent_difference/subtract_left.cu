/******************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2024, Moore Threads Corporation.  All rights reserved.
 *
 * MUSA port of adjacent_difference/subtract_left benchmark.
 * Benchmarks DeviceAdjacentDifference::SubtractLeft.
 ******************************************************************************/

#include <musa_bench.cuh>
#include <generator.cuh>
#include <cub/device/device_adjacent_difference.cuh>

// Default tuning parameters (can be overridden via compile flags)
#ifndef TUNE_BASE
#define TUNE_BASE 1
#endif

#ifndef TUNE_ITEMS_PER_THREAD
#define TUNE_ITEMS_PER_THREAD 7
#endif

#ifndef TUNE_THREADS_PER_BLOCK
#define TUNE_THREADS_PER_BLOCK 256
#endif

#if !TUNE_BASE
struct policy_hub_t
{
  struct Policy350 : cub::ChainedPolicy<350, Policy350, Policy350>
  {
    using AdjacentDifferencePolicy =
      cub::AgentAdjacentDifferencePolicy<TUNE_THREADS_PER_BLOCK,
                                         TUNE_ITEMS_PER_THREAD,
                                         cub::BLOCK_LOAD_WARP_TRANSPOSE,
                                         cub::LOAD_CA,
                                         cub::BLOCK_STORE_WARP_TRANSPOSE>;
  };

  using MaxPolicy = Policy350;
};
#endif // !TUNE_BASE

template <typename T, class OffsetT>
void run_benchmark(int64_t elements)
{
  constexpr bool may_alias = false;
  constexpr bool read_left = true;

  using input_it_t = const T*;
  using output_it_t = T*;
  using difference_op_t = cub::Difference;
  using offset_t = typename cub::detail::ChooseOffsetT<OffsetT>::Type;

#if !TUNE_BASE
  using dispatch_t = cub::DispatchAdjacentDifference<input_it_t,
                                                     output_it_t,
                                                     difference_op_t,
                                                     offset_t,
                                                     may_alias,
                                                     read_left,
                                                     policy_hub_t>;
#else
  using dispatch_t = cub::DispatchAdjacentDifference<input_it_t,
                                                     output_it_t,
                                                     difference_op_t,
                                                     offset_t,
                                                     may_alias,
                                                     read_left>;
#endif

  // Setup benchmark state
  musa_bench::State state;
  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<T>(elements);

  // Allocate data
  musa_bench::device_vector<T> in(elements);
  musa_bench::device_vector<T> out(elements);

  // Generate random input data
  musa_bench::gen(musa_bench::seed_t{}, in);

  input_it_t d_in   = in.data();
  output_it_t d_out = out.data();

  // Allocate temporary storage
  std::size_t temp_storage_bytes{};
  dispatch_t::Dispatch(nullptr,
                       temp_storage_bytes,
                       d_in,
                       d_out,
                       static_cast<offset_t>(elements),
                       difference_op_t{},
                       0,
                       false /* debug_synchronous */);

  musa_bench::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  std::uint8_t* d_temp_storage = temp_storage.data();

  // Create timer
  musa_bench::Timer timer(state.stream);

  // Warmup
  for (int i = 0; i < state.warmup_iterations; i++) {
    dispatch_t::Dispatch(d_temp_storage,
                         temp_storage_bytes,
                         d_in,
                         d_out,
                         static_cast<offset_t>(elements),
                         difference_op_t{},
                         state.stream,
                         false /* debug_synchronous */);
  }
  MUSA_BENCH_CHECK(musaDeviceSynchronize());

  // Benchmark
  for (int i = 0; i < state.test_iterations; i++) {
    timer.start();
    dispatch_t::Dispatch(d_temp_storage,
                         temp_storage_bytes,
                         d_in,
                         d_out,
                         static_cast<offset_t>(elements),
                         difference_op_t{},
                         state.stream,
                         false /* debug_synchronous */);
    timer.stop();

    float ms = timer.elapsed_ms();
    state.total_time_ms += ms;
    state.min_time_ms = std::min(state.min_time_ms, (double)ms);
    state.max_time_ms = std::max(state.max_time_ms, (double)ms);
  }

  // Print results
  std::cout << "Type: " << musa_bench::type_name<T>() << std::endl;
  state.print_results();
}

int main(int argc, char **argv)
{
  // Default element count: 2^24 = 16M
  int64_t elements = 16777216;

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if ((arg == "-n" || arg == "--elements") && i + 1 < argc) {
      elements = std::stoll(argv[++i]);
    } else if (arg == "-h" || arg == "--help") {
      std::cout << "Usage: " << argv[0] << " [options]\n"
                << "Options:\n"
                << "  -n, --elements N  Number of elements (default: 16777216)\n"
                << "  -h, --help        Show this help message\n";
      return 0;
    }
  }

  std::cout << "=== Benchmark: cub::DeviceAdjacentDifference::SubtractLeftCopy ===" << std::endl;

  // Run benchmark with int32_t type and int32_t offset (most common)
  run_benchmark<int32_t, int32_t>(elements);

  return 0;
}