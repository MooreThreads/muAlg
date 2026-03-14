/******************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2024, Moore Threads Corporation.  All rights reserved.
 *
 * MUSA port of run_length_encode/non_trivial_runs benchmark.
 * Uses DeviceReduceByKey to find non-trivial runs (runs with length > 1).
 ******************************************************************************/

#include <musa_bench.cuh>
#include <generator.cuh>
#include <cub/device/device_reduce.cuh>
#include <limits>
#include <cmath>
#include <vector>

//=============================================================================
// Generate uniform key segments for RLE benchmark
//=============================================================================

// Host-side generation of uniform key segments
template <typename T>
void generate_uniform_key_segments(musa_bench::device_vector<T> &keys,
                                   std::size_t elements,
                                   std::size_t min_segment_size,
                                   std::size_t max_segment_size)
{
  // Generate on host
  std::vector<T> h_keys(elements);
  T current_key = 0;
  std::size_t i = 0;

  // Use simple uniform segment sizes
  std::size_t avg_segment_size = (min_segment_size + max_segment_size) / 2;
  if (avg_segment_size < 1) avg_segment_size = 1;

  while (i < elements) {
    // Simple segment size alternating between min and max
    std::size_t segment_size = ((current_key % 2) == 0) ? min_segment_size : max_segment_size;
    if (segment_size < 1) segment_size = 1;
    if (i + segment_size > elements) {
      segment_size = elements - i;
    }

    for (std::size_t j = 0; j < segment_size && i < elements; j++, i++) {
      h_keys[i] = current_key;
    }
    current_key++;
  }

  // Copy to device
  MUSA_BENCH_CHECK(musaMemcpy(keys.data(), h_keys.data(), elements * sizeof(T),
                               musaMemcpyHostToDevice));
}

//=============================================================================
// Benchmark implementation
//=============================================================================

template <typename T, typename OffsetT>
void run_non_trivial_runs(int64_t elements, int64_t max_segment_size, bool output_json)
{
  using offset_t = OffsetT;

  // Setup benchmark state
  musa_bench::State state;
  state.benchmark_name = "cub.bench.non_trivial_runs";
  state.type_name = musa_bench::type_name<T>();
  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  // Output: keys and aggregate values
  state.add_global_memory_writes<T>(elements);
  state.add_global_memory_writes<OffsetT>(elements);
  state.add_global_memory_writes<OffsetT>(1);

  // Allocate data
  musa_bench::device_vector<offset_t> num_runs_out(1);
  musa_bench::device_vector<offset_t> out_vals(elements);
  musa_bench::device_vector<T> out_keys(elements);
  musa_bench::device_vector<T> in_keys(elements);

  // Generate uniform key segments
  generate_uniform_key_segments(in_keys, elements, 1, max_segment_size);

  T *d_in_keys             = in_keys.data();
  T *d_out_keys            = out_keys.data();
  offset_t *d_out_vals     = out_vals.data();
  offset_t *d_num_runs_out = num_runs_out.data();

  // Constant input iterator for values (always 1)
  cub::ConstantInputIterator<offset_t, offset_t> d_in_vals(offset_t{1});

  std::uint8_t *d_temp_storage{};
  std::size_t temp_storage_bytes{};

  // Query temporary storage
  cub::DeviceReduce::ReduceByKey(d_temp_storage,
                                 temp_storage_bytes,
                                 d_in_keys,
                                 d_out_keys,
                                 d_in_vals,
                                 d_out_vals,
                                 d_num_runs_out,
                                 cub::Sum{},
                                 static_cast<int>(elements),
                                 0);

  musa_bench::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = temp_storage.data();

  // Create timer
  musa_bench::Timer timer(state.stream);

  // Warmup
  for (int i = 0; i < state.warmup_iterations; i++) {
    cub::DeviceReduce::ReduceByKey(d_temp_storage,
                                   temp_storage_bytes,
                                   d_in_keys,
                                   d_out_keys,
                                   d_in_vals,
                                   d_out_vals,
                                   d_num_runs_out,
                                   cub::Sum{},
                                   static_cast<int>(elements),
                                   state.stream);
  }
  MUSA_BENCH_CHECK(musaDeviceSynchronize());

  // Benchmark
  for (int i = 0; i < state.test_iterations; i++) {
    timer.start();
    cub::DeviceReduce::ReduceByKey(d_temp_storage,
                                   temp_storage_bytes,
                                   d_in_keys,
                                   d_out_keys,
                                   d_in_vals,
                                   d_out_vals,
                                   d_num_runs_out,
                                   cub::Sum{},
                                   static_cast<int>(elements),
                                   state.stream);
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
    std::cout << "Type: " << musa_bench::type_name<T>() << std::endl;
    std::cout << "Offset: " << musa_bench::type_name<OffsetT>() << std::endl;
    std::cout << "Elements: " << elements << std::endl;
    std::cout << "MaxSegmentSize: " << max_segment_size << std::endl;
    state.print_results();
  }
}

int main(int argc, char **argv)
{
  // Default values
  int64_t elements = 16777216;  // 2^24
  int64_t max_segment_size = 8;
  bool output_json = false;

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if ((arg == "-n" || arg == "--elements") && i + 1 < argc) {
      elements = std::stoll(argv[++i]);
    } else if ((arg == "-s" || arg == "--max-segment") && i + 1 < argc) {
      max_segment_size = std::stoll(argv[++i]);
    } else if (arg == "--json") {
      output_json = true;
    } else if (arg == "-h" || arg == "--help") {
      std::cout << "Usage: " << argv[0] << " [options]\n"
                << "Options:\n"
                << "  -n, --elements N     Number of elements (default: 16777216)\n"
                << "  -s, --max-segment N  Maximum segment size (default: 8)\n"
                << "  --json               Output results in JSON format\n"
                << "  -h, --help           Show this help message\n";
      return 0;
    }
  }

  if (!output_json) {
    std::cout << "=== Benchmark: cub::DeviceRunLengthEncode::NonTrivialRuns ===" << std::endl;
    std::cout << "(Implemented via DeviceReduceByKey)" << std::endl;
  }

  // Run benchmark with int32_t type (most common)
  run_non_trivial_runs<int32_t, int32_t>(elements, max_segment_size, output_json);

  return 0;
}