/******************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2024, Moore Threads Corporation.  All rights reserved.
 *
 * MUSA port of scan/exclusive/sum benchmark.
 ******************************************************************************/

#include <musa_bench.cuh>
#include <generator.cuh>
#include <cub/device/device_scan.cuh>

using op_t = cub::Sum;

template <typename T>
void run_benchmark(int64_t elements, bool output_json)
{
  using input_it_t  = const T *;
  using output_it_t = T *;
  using offset_t    = int;
  using init_t      = cub::detail::InputValue<T>;

  // Setup benchmark state
  musa_bench::State state;
  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<T>(elements);

  // Set benchmark identification for JSON output
  state.benchmark_name = "cub.bench.scan.exclusive.sum";
  state.type_name = musa_bench::type_name<T>();

  // Allocate data
  musa_bench::device_vector<T> in(elements);
  musa_bench::device_vector<T> out(elements);

  // Generate random input data
  musa_bench::gen(musa_bench::seed_t{}, in);

  input_it_t d_in   = in.data();
  output_it_t d_out = out.data();

  // Allocate temporary storage
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out,
                                 static_cast<offset_t>(elements));

  musa_bench::device_vector<uint8_t> temp(temp_storage_bytes);
  d_temp_storage = temp.data();

  // Create timer
  musa_bench::Timer timer(state.stream);

  // Warmup
  for (int i = 0; i < state.warmup_iterations; i++) {
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out,
                                   static_cast<offset_t>(elements), state.stream);
  }
  MUSA_BENCH_CHECK(musaDeviceSynchronize());

  // Benchmark
  for (int i = 0; i < state.test_iterations; i++) {
    timer.start();
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out,
                                   static_cast<offset_t>(elements), state.stream);
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
    std::cout << "=== Benchmark: cub::DeviceScan::ExclusiveSum ===" << std::endl;
  }

  // Run benchmark with int32_t type (most common)
  run_benchmark<int32_t>(elements, output_json);

  return 0;
}