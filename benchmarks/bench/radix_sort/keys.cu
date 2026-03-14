/******************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2024, Moore Threads Corporation.  All rights reserved.
 *
 * MUSA port of radix_sort/keys benchmark.
 * Benchmarks cub::DeviceRadixSort::SortKeys
 ******************************************************************************/

#include <musa_bench.cuh>
#include <generator.cuh>
#include <cub/device/device_radix_sort.cuh>

template <typename KeyT>
void run_benchmark(int64_t elements)
{
  using key_t = KeyT;
  using offset_t = int;

  // Setup benchmark state
  musa_bench::State state;
  state.add_element_count(elements);
  state.add_global_memory_reads<KeyT>(elements);
  state.add_global_memory_writes<KeyT>(elements);

  // Allocate double buffers for keys (required for radix sort)
  musa_bench::device_vector<key_t> keys_buffer_1(elements);
  musa_bench::device_vector<key_t> keys_buffer_2(elements);

  // Generate random input data
  musa_bench::gen(musa_bench::seed_t{}, keys_buffer_1);

  key_t *d_keys_buffer_1 = keys_buffer_1.data();
  key_t *d_keys_buffer_2 = keys_buffer_2.data();

  // Create DoubleBuffer for keys
  cub::DoubleBuffer<key_t> d_keys(d_keys_buffer_1, d_keys_buffer_2);

  // Allocate temporary storage:
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  // Sort bits from 0 to sizeof(key_t) * 8 (full key sort)
  int begin_bit = 0;
  int end_bit = sizeof(key_t) * 8;

  cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys,
                                  static_cast<offset_t>(elements),
                                  begin_bit, end_bit, state.stream, false);

  musa_bench::device_vector<uint8_t> temp(temp_storage_bytes);
  d_temp_storage = temp.data();

  // Create timer
  musa_bench::Timer timer(state.stream);

  // Warmup
  for (int i = 0; i < state.warmup_iterations; i++) {
    cub::DoubleBuffer<key_t> keys = d_keys;
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, keys,
                                    static_cast<offset_t>(elements),
                                    begin_bit, end_bit, state.stream, false);
  }
  MUSA_BENCH_CHECK(musaDeviceSynchronize());

  // Benchmark
  for (int i = 0; i < state.test_iterations; i++) {
    cub::DoubleBuffer<key_t> keys = d_keys;
    timer.start();
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, keys,
                                    static_cast<offset_t>(elements),
                                    begin_bit, end_bit, state.stream, false);
    timer.stop();

    float ms = timer.elapsed_ms();
    state.total_time_ms += ms;
    state.min_time_ms = std::min(state.min_time_ms, (double)ms);
    state.max_time_ms = std::max(state.max_time_ms, (double)ms);
  }

  // Print results
  std::cout << "Type: " << musa_bench::type_name<KeyT>() << std::endl;
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

  std::cout << "=== Benchmark: cub::DeviceRadixSort::SortKeys ===" << std::endl;

  // Run benchmark with int32_t type (most common)
  run_benchmark<int32_t>(elements);

  return 0;
}