/******************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2024, Moore Threads Corporation.  All rights reserved.
 *
 * MUSA port of DeviceSelect::UniqueByKey benchmark.
 ******************************************************************************/

#include <musa_bench.cuh>
#include <generator.cuh>
#include <cub/device/device_select.cuh>

// For TUNE_BASE mode
#define TUNE_BASE 1

template <typename KeyT, typename ValueT, typename OffsetT>
void run_unique_by_key_benchmark(int64_t elements, int64_t max_segment_size, bool output_json)
{
  using keys_input_it_t = const KeyT *;
  using keys_output_it_t = KeyT *;
  using vals_input_it_t = const ValueT *;
  using vals_output_it_t = ValueT *;
  using num_runs_output_iterator_t = OffsetT *;
  using equality_op_t = cub::Equality;
  using offset_t = OffsetT;

  using dispatch_t = cub::DispatchUniqueByKey<keys_input_it_t,
                                               vals_input_it_t,
                                               keys_output_it_t,
                                               vals_output_it_t,
                                               num_runs_output_iterator_t,
                                               equality_op_t,
                                               offset_t>;

  // Setup benchmark state
  musa_bench::State state;
  state.benchmark_name = "cub.bench.select_unique_by_key";
  state.type_name = musa_bench::type_name<KeyT>();
  state.add_element_count(elements);

  // Allocate data
  musa_bench::device_vector<OffsetT> num_runs_out(1);
  musa_bench::device_vector<ValueT> in_vals(elements);
  musa_bench::device_vector<ValueT> out_vals(elements);
  musa_bench::device_vector<KeyT> out_keys(elements);

  // Generate key segments with uniform distribution
  const size_t min_segment_size = 1;
  musa_bench::device_vector<KeyT> in_keys =
      musa_bench::gen_uniform_key_segments<KeyT>(musa_bench::seed_t{},
                                                   static_cast<size_t>(elements),
                                                   min_segment_size,
                                                   static_cast<size_t>(max_segment_size));

  // Generate random values
  musa_bench::gen(musa_bench::seed_t{}, in_vals);

  KeyT *d_in_keys = in_keys.data();
  KeyT *d_out_keys = out_keys.data();
  ValueT *d_in_vals = in_vals.data();
  ValueT *d_out_vals = out_vals.data();
  OffsetT *d_num_runs_out = num_runs_out.data();

  // First call to get temp storage size
  std::uint8_t *d_temp_storage = nullptr;
  std::size_t temp_storage_bytes = 0;

  dispatch_t::Dispatch(d_temp_storage,
                       temp_storage_bytes,
                       d_in_keys,
                       d_in_vals,
                       d_out_keys,
                       d_out_vals,
                       d_num_runs_out,
                       equality_op_t{},
                       elements,
                       0,
                       false);

  musa_bench::device_vector<uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = temp_storage.data();

  // Run once to get num_runs
  dispatch_t::Dispatch(d_temp_storage,
                       temp_storage_bytes,
                       d_in_keys,
                       d_in_vals,
                       d_out_keys,
                       d_out_vals,
                       d_num_runs_out,
                       equality_op_t{},
                       elements,
                       0,
                       false);
  MUSA_BENCH_CHECK(musaDeviceSynchronize());

  // For throughput calculation, use elements as upper bound
  int64_t num_runs = elements; // upper bound
  state.add_global_memory_reads<KeyT>(elements);
  state.add_global_memory_reads<ValueT>(elements);
  state.add_global_memory_writes<ValueT>(num_runs);
  state.add_global_memory_writes<KeyT>(num_runs);
  state.add_global_memory_writes<OffsetT>(1);

  // Create timer
  musa_bench::Timer timer(state.stream);

  // Warmup
  for (int i = 0; i < state.warmup_iterations; i++) {
    dispatch_t::Dispatch(d_temp_storage,
                         temp_storage_bytes,
                         d_in_keys,
                         d_in_vals,
                         d_out_keys,
                         d_out_vals,
                         d_num_runs_out,
                         equality_op_t{},
                         elements,
                         state.stream,
                         false);
  }
  MUSA_BENCH_CHECK(musaDeviceSynchronize());

  // Benchmark
  for (int i = 0; i < state.test_iterations; i++) {
    timer.start();
    dispatch_t::Dispatch(d_temp_storage,
                         temp_storage_bytes,
                         d_in_keys,
                         d_in_vals,
                         d_out_keys,
                         d_out_vals,
                         d_num_runs_out,
                         equality_op_t{},
                         elements,
                         state.stream,
                         false);
    timer.stop();

    float ms = timer.elapsed_ms();
    state.total_time_ms += ms;
    state.min_time_ms = std::min(state.min_time_ms, (double)ms);
    state.max_time_ms = std::max(state.max_time_ms, (double)ms);
  }

  // Print results
  if (output_json) {
    state.print_json();
  } else {
    std::cout << "KeyType: " << musa_bench::type_name<KeyT>()
              << ", ValueType: " << musa_bench::type_name<ValueT>()
              << ", MaxSegSize: " << max_segment_size << std::endl;
    state.print_results();
  }
}

int main(int argc, char **argv)
{
  // Default element count: 2^24 = 16M
  int64_t elements = 16777216;
  int64_t max_segment_size = 4;
  bool output_json = false;

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if ((arg == "-n" || arg == "--elements") && i + 1 < argc) {
      elements = std::stoll(argv[++i]);
    } else if ((arg == "-s" || arg == "--max-seg-size") && i + 1 < argc) {
      max_segment_size = std::stoll(argv[++i]);
    } else if (arg == "--json") {
      output_json = true;
    } else if (arg == "-h" || arg == "--help") {
      std::cout << "Usage: " << argv[0] << " [options]\n"
                << "Options:\n"
                << "  -n, --elements N     Number of elements (default: 16777216)\n"
                << "  -s, --max-seg-size N Maximum segment size (default: 4)\n"
                << "  --json               Output results in JSON format\n"
                << "  -h, --help           Show this help message\n";
      return 0;
    }
  }

  if (!output_json) {
    std::cout << "=== Benchmark: cub::DeviceSelect::UniqueByKey ===" << std::endl;
  }

  // Run with int32_t key and value types (most common)
  run_unique_by_key_benchmark<int32_t, int32_t, int32_t>(elements, max_segment_size, output_json);

  return 0;
}