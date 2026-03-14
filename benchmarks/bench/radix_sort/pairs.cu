/******************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2024, Moore Threads Corporation.  All rights reserved.
 *
 * MUSA port of radix_sort/pairs benchmark.
 * Benchmarks cub::DeviceRadixSort::SortPairs
 ******************************************************************************/

#include <musa_bench.cuh>
#include <generator.cuh>
#include <cub/device/device_radix_sort.cuh>

template <typename KeyT, typename ValueT>
void run_benchmark(int64_t elements, bool output_json)
{
  using key_t = KeyT;
  using value_t = ValueT;
  using offset_t = int;

  // Setup benchmark state
  musa_bench::State state;
  state.add_element_count(elements);
  state.add_global_memory_reads<KeyT>(elements);
  state.add_global_memory_reads<ValueT>(elements);
  state.add_global_memory_writes<KeyT>(elements);
  state.add_global_memory_writes<ValueT>(elements);

  // Set benchmark identification for JSON output
  state.benchmark_name = "cub.bench.radix_sort.pairs";
  state.type_name = std::string(musa_bench::type_name<KeyT>()) + "_" + musa_bench::type_name<ValueT>();

  // Allocate double buffers for keys and values (required for radix sort)
  musa_bench::device_vector<key_t> keys_buffer_1(elements);
  musa_bench::device_vector<key_t> keys_buffer_2(elements);
  musa_bench::device_vector<value_t> values_buffer_1(elements);
  musa_bench::device_vector<value_t> values_buffer_2(elements);

  // Generate random input data
  musa_bench::gen(musa_bench::seed_t{}, keys_buffer_1);
  musa_bench::gen(musa_bench::seed_t{}, values_buffer_1);

  key_t *d_keys_buffer_1 = keys_buffer_1.data();
  key_t *d_keys_buffer_2 = keys_buffer_2.data();
  value_t *d_values_buffer_1 = values_buffer_1.data();
  value_t *d_values_buffer_2 = values_buffer_2.data();

  // Create DoubleBuffer for keys and values
  cub::DoubleBuffer<key_t> d_keys(d_keys_buffer_1, d_keys_buffer_2);
  cub::DoubleBuffer<value_t> d_values(d_values_buffer_1, d_values_buffer_2);

  // Allocate temporary storage:
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  // Sort bits from 0 to sizeof(key_t) * 8 (full key sort)
  int begin_bit = 0;
  int end_bit = sizeof(key_t) * 8;

  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values,
                                   static_cast<offset_t>(elements),
                                   begin_bit, end_bit, state.stream);

  musa_bench::device_vector<uint8_t> temp(temp_storage_bytes);
  d_temp_storage = temp.data();

  // Create timer
  musa_bench::Timer timer(state.stream);

  // Warmup
  for (int i = 0; i < state.warmup_iterations; i++) {
    cub::DoubleBuffer<key_t> keys = d_keys;
    cub::DoubleBuffer<value_t> values = d_values;
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, keys, values,
                                     static_cast<offset_t>(elements),
                                     begin_bit, end_bit, state.stream);
  }
  MUSA_BENCH_CHECK(musaDeviceSynchronize());

  // Benchmark
  for (int i = 0; i < state.test_iterations; i++) {
    cub::DoubleBuffer<key_t> keys = d_keys;
    cub::DoubleBuffer<value_t> values = d_values;
    timer.start();
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, keys, values,
                                     static_cast<offset_t>(elements),
                                     begin_bit, end_bit, state.stream);
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
    std::cout << "Key Type: " << musa_bench::type_name<KeyT>()
              << ", Value Type: " << musa_bench::type_name<ValueT>() << std::endl;
    state.print_results();
  }
}

int main(int argc, char **argv)
{
  // Default element count: 2^24 = 16M
  int64_t elements = 16777216;
  bool output_json = false;

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if ((arg == "-n" || arg == "--elements") && i + 1 < argc) {
      elements = std::stoll(argv[++i]);
    } else if (arg == "--json") {
      output_json = true;
    } else if (arg == "-h" || arg == "--help") {
      std::cout << "Usage: " << argv[0] << " [options]\n"
                << "Options:\n"
                << "  -n, --elements N  Number of elements (default: 16777216)\n"
                << "  --json            Output results in JSON format\n"
                << "  -h, --help        Show this help message\n";
      return 0;
    }
  }

  if (!output_json) {
    std::cout << "=== Benchmark: cub::DeviceRadixSort::SortPairs ===" << std::endl;
  }

  // Run benchmark with int32_t key and value type (most common)
  run_benchmark<int32_t, int32_t>(elements, output_json);

  return 0;
}