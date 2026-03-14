/******************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2024, Moore Threads Corporation.  All rights reserved.
 *
 * MUSA port of reduce/by_key benchmark.
 *
 * Benchmark for cub::DeviceReduce::ReduceByKey - reduces values by key segments.
 * This operation groups consecutive equal keys and reduces the corresponding values.
 ******************************************************************************/

#include <musa_bench.cuh>
#include <generator.cuh>
#include <cub/device/device_reduce.cuh>

#include <vector>
#include <random>
#include <algorithm>

namespace {

//=============================================================================
// Helper to generate uniform key segments on host
//=============================================================================

template <typename KeyT>
std::vector<KeyT> generate_uniform_key_segments_host(size_t total_elements,
                                                      size_t min_segment_size,
                                                      size_t max_segment_size,
                                                      unsigned long long seed)
{
  std::vector<KeyT> keys(total_elements);
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<size_t> seg_dist(min_segment_size, max_segment_size);

  KeyT current_key = KeyT(0);
  size_t idx = 0;

  while (idx < total_elements) {
    // Determine segment size
    size_t remaining = total_elements - idx;
    size_t seg_size = std::min(seg_dist(rng), remaining);
    if (remaining <= max_segment_size) {
      seg_size = remaining;  // Last segment takes all remaining
    }

    // Fill segment with current key
    for (size_t i = 0; i < seg_size && idx < total_elements; i++, idx++) {
      keys[idx] = current_key;
    }
    current_key++;
  }

  return keys;
}

} // anonymous namespace

template <typename KeyT, typename ValueT>
void run_benchmark(int64_t elements, size_t max_segment_size)
{
  using offset_t = int;

  // Setup benchmark state
  musa_bench::State state;
  state.add_element_count(elements);

  // We read keys and values, write unique keys, aggregated values, and num_runs
  state.add_global_memory_reads<KeyT>(elements);
  state.add_global_memory_reads<ValueT>(elements);

  // Allocate data
  musa_bench::device_vector<KeyT> in_keys(elements);
  musa_bench::device_vector<KeyT> out_keys(elements);
  musa_bench::device_vector<ValueT> in_vals(elements);
  musa_bench::device_vector<ValueT> out_vals(elements);
  musa_bench::device_vector<offset_t> num_runs_out(1);

  // Generate key segments on host
  size_t min_segment_size = 1;
  std::vector<KeyT> h_keys = generate_uniform_key_segments_host<KeyT>(
      elements, min_segment_size, max_segment_size, musa_bench::seed_t{}.get());

  // Copy keys to device
  MUSA_BENCH_CHECK(musaMemcpy(in_keys.data(), h_keys.data(),
                               elements * sizeof(KeyT), musaMemcpyHostToDevice));

  // Generate random values
  musa_bench::gen(musa_bench::seed_t{}, in_vals);

  // Get raw pointers
  const KeyT *d_in_keys = in_keys.data();
  KeyT *d_out_keys = out_keys.data();
  const ValueT *d_in_vals = in_vals.data();
  ValueT *d_out_vals = out_vals.data();
  offset_t *d_num_runs = num_runs_out.data();

  // Allocate temporary storage
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes,
                                  d_in_keys, d_out_keys,
                                  d_in_vals, d_out_vals,
                                  d_num_runs,
                                  cub::Sum(),
                                  static_cast<int>(elements));

  musa_bench::device_vector<uint8_t> temp(temp_storage_bytes);
  d_temp_storage = temp.data();

  // Warmup run to get num_runs
  cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes,
                                  d_in_keys, d_out_keys,
                                  d_in_vals, d_out_vals,
                                  d_num_runs,
                                  cub::Sum(),
                                  static_cast<int>(elements),
                                  state.stream);
  MUSA_BENCH_CHECK(musaDeviceSynchronize());

  // Get number of runs for memory accounting
  offset_t h_num_runs;
  MUSA_BENCH_CHECK(musaMemcpy(&h_num_runs, d_num_runs, sizeof(offset_t),
                               musaMemcpyDeviceToHost));

  // Update memory accounting
  state.add_global_memory_writes<KeyT>(h_num_runs);
  state.add_global_memory_writes<ValueT>(h_num_runs);
  state.add_global_memory_writes<offset_t>(1);

  // Create timer
  musa_bench::Timer timer(state.stream);

  // Warmup
  for (int i = 0; i < state.warmup_iterations; i++) {
    cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes,
                                    d_in_keys, d_out_keys,
                                    d_in_vals, d_out_vals,
                                    d_num_runs,
                                    cub::Sum(),
                                    static_cast<int>(elements),
                                    state.stream);
  }
  MUSA_BENCH_CHECK(musaDeviceSynchronize());

  // Benchmark
  for (int i = 0; i < state.test_iterations; i++) {
    timer.start();
    cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes,
                                    d_in_keys, d_out_keys,
                                    d_in_vals, d_out_vals,
                                    d_num_runs,
                                    cub::Sum(),
                                    static_cast<int>(elements),
                                    state.stream);
    timer.stop();

    float ms = timer.elapsed_ms();
    state.total_time_ms += ms;
    state.min_time_ms = std::min(state.min_time_ms, (double)ms);
    state.max_time_ms = std::max(state.max_time_ms, (double)ms);
  }

  // Print results
  std::cout << "KeyType: " << musa_bench::type_name<KeyT>() << std::endl;
  std::cout << "ValueType: " << musa_bench::type_name<ValueT>() << std::endl;
  std::cout << "MaxSegmentSize: " << max_segment_size << std::endl;
  std::cout << "NumSegments: " << h_num_runs << std::endl;
  state.print_results();
}

int main(int argc, char **argv)
{
  // Default element count: 2^24 = 16M
  int64_t elements = 16777216;
  size_t max_segment_size = 8;

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if ((arg == "-n" || arg == "--elements") && i + 1 < argc) {
      elements = std::stoll(argv[++i]);
    } else if ((arg == "-s" || arg == "--max-seg-size") && i + 1 < argc) {
      max_segment_size = std::stoull(argv[++i]);
    } else if (arg == "-h" || arg == "--help") {
      std::cout << "Usage: " << argv[0] << " [options]\n"
                << "Options:\n"
                << "  -n, --elements N      Number of elements (default: 16777216)\n"
                << "  -s, --max-seg-size N  Maximum segment size (default: 8)\n"
                << "  -h, --help            Show this help message\n";
      return 0;
    }
  }

  std::cout << "=== Benchmark: cub::DeviceReduce::ReduceByKey ===" << std::endl;

  // Run benchmark with int32_t key and value types (most common)
  run_benchmark<int32_t, int32_t>(elements, max_segment_size);

  return 0;
}