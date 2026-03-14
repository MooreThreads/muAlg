/******************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2024, Moore Threads Corporation.  All rights reserved.
 *
 * MUSA port of segmented_sort/power_law/keys benchmark.
 * Benchmarks DeviceSegmentedSort::SortKeys with power-law distributed segments.
 ******************************************************************************/

#include <musa_bench.cuh>
#include <generator.cuh>
#include <cub/device/device_segmented_sort.cuh>

template <typename KeyT, typename OffsetT>
void run_benchmark(int64_t elements, int64_t num_segments, bool output_json)
{
  constexpr bool is_descending   = false;
  constexpr bool is_overwrite_ok = false;

  using offset_t          = OffsetT;
  using begin_offset_it_t = const offset_t *;
  using end_offset_it_t   = const offset_t *;
  using key_t             = KeyT;
  using value_t           = cub::NullType;

  using dispatch_t = cub::DispatchSegmentedSort<is_descending,
                                                 key_t,
                                                 value_t,
                                                 offset_t,
                                                 begin_offset_it_t,
                                                 end_offset_it_t>;

  // Setup benchmark state
  musa_bench::State state;
  state.benchmark_name = "cub.bench.segmented_sort_keys_power_law";
  state.type_name = musa_bench::type_name<KeyT>();
  state.add_element_count(elements);
  state.add_global_memory_reads<key_t>(elements);
  state.add_global_memory_writes<key_t>(elements);

  // Allocate data buffers
  musa_bench::device_vector<key_t> buffer_1(elements);
  musa_bench::device_vector<key_t> buffer_2(elements);

  // Generate random input data
  musa_bench::gen(musa_bench::seed_t{}, buffer_1);

  key_t *d_buffer_1 = buffer_1.data();
  key_t *d_buffer_2 = buffer_2.data();

  cub::DoubleBuffer<key_t> d_keys(d_buffer_1, d_buffer_2);
  cub::DoubleBuffer<value_t> d_values;

  // Generate power-law distributed segment offsets
  musa_bench::device_vector<offset_t> offsets =
    musa_bench::gen_power_law_offsets<offset_t>(musa_bench::seed_t{},
                                                 static_cast<size_t>(elements),
                                                 static_cast<size_t>(num_segments));

  state.add_global_memory_reads<offset_t>(num_segments + 1);

  begin_offset_it_t d_begin_offsets = offsets.data();
  end_offset_it_t d_end_offsets     = d_begin_offsets + 1;

  // Allocate temporary storage
  std::size_t temp_storage_bytes{};
  void *d_temp_storage{};
  dispatch_t::Dispatch(d_temp_storage,
                       temp_storage_bytes,
                       d_keys,
                       d_values,
                       static_cast<offset_t>(elements),
                       static_cast<int>(num_segments),
                       d_begin_offsets,
                       d_end_offsets,
                       is_overwrite_ok,
                       0 /* stream */,
                       false /* debug_synchronous */);

  musa_bench::device_vector<uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = temp_storage.data();

  // Create timer
  musa_bench::Timer timer(state.stream);

  // Warmup
  for (int i = 0; i < state.warmup_iterations; i++) {
    cub::DoubleBuffer<key_t> keys     = d_keys;
    cub::DoubleBuffer<value_t> values = d_values;
    dispatch_t::Dispatch(d_temp_storage,
                         temp_storage_bytes,
                         keys,
                         values,
                         static_cast<offset_t>(elements),
                         static_cast<int>(num_segments),
                         d_begin_offsets,
                         d_end_offsets,
                         is_overwrite_ok,
                         state.stream,
                         false /* debug_synchronous */);
  }
  MUSA_BENCH_CHECK(musaDeviceSynchronize());

  // Benchmark
  for (int i = 0; i < state.test_iterations; i++) {
    cub::DoubleBuffer<key_t> keys     = d_keys;
    cub::DoubleBuffer<value_t> values = d_values;

    timer.start();
    dispatch_t::Dispatch(d_temp_storage,
                         temp_storage_bytes,
                         keys,
                         values,
                         static_cast<offset_t>(elements),
                         static_cast<int>(num_segments),
                         d_begin_offsets,
                         d_end_offsets,
                         is_overwrite_ok,
                         state.stream,
                         false /* debug_synchronous */);
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
    std::cout << "Type: " << musa_bench::type_name<KeyT>() << std::endl;
    std::cout << "Elements: " << elements << std::endl;
    std::cout << "Segments: " << num_segments << std::endl;
    state.print_results();
  }
}

int main(int argc, char **argv)
{
  // Default values
  int64_t elements = 1 << 24;  // 16M elements
  int64_t num_segments = 1 << 16;  // 64K segments
  bool output_json = false;

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if ((arg == "-n" || arg == "--elements") && i + 1 < argc) {
      elements = std::stoll(argv[++i]);
    } else if ((arg == "-s" || arg == "--segments") && i + 1 < argc) {
      num_segments = std::stoll(argv[++i]);
    } else if (arg == "--json") {
      output_json = true;
    } else if (arg == "-h" || arg == "--help") {
      std::cout << "Usage: " << argv[0] << " [options]\n"
                << "Options:\n"
                << "  -n, --elements N   Number of elements (default: 16777216)\n"
                << "  -s, --segments N   Number of segments (default: 65536)\n"
                << "  --json             Output results in JSON format\n"
                << "  -h, --help         Show this help message\n";
      return 0;
    }
  }

  if (!output_json) {
    std::cout << "=== Benchmark: cub::DeviceSegmentedSort::SortKeys (power-law distribution) ===" << std::endl;
  }

  // Run benchmark with int32_t type and uint32_t offset
  run_benchmark<int32_t, uint32_t>(elements, num_segments, output_json);

  return 0;
}