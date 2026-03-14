/******************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2024, Moore Threads Corporation.  All rights reserved.
 *
 * MUSA port of DevicePartition::Flagged benchmark.
 ******************************************************************************/

#include <musa_bench.cuh>
#include <generator.cuh>
#include <cub/device/device_partition.cuh>

// For TUNE_BASE mode
#define TUNE_BASE 1

constexpr bool keep_rejects = true;
constexpr bool may_alias = false;

template <typename T, typename OffsetT>
void run_partition_benchmark(int64_t elements, const std::string &entropy_str, bool output_json)
{
  using input_it_t = const T *;
  using flag_it_t = const bool *;
  using output_it_t = T *;
  using num_selected_it_t = OffsetT *;
  using select_op_t = cub::NullType;
  using equality_op_t = cub::NullType;
  using offset_t = OffsetT;

  using dispatch_t = cub::DispatchSelectIf<input_it_t,
                                           flag_it_t,
                                           output_it_t,
                                           num_selected_it_t,
                                           select_op_t,
                                           equality_op_t,
                                           offset_t,
                                           keep_rejects>;

  // Parse entropy
  musa_bench::bit_entropy entropy = musa_bench::str_to_entropy(entropy_str);

  // Setup benchmark state
  musa_bench::State state;
  state.benchmark_name = "cub.bench.partition_flagged";
  state.type_name = musa_bench::type_name<T>();
  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_reads<bool>(elements);
  state.add_global_memory_writes<T>(elements);
  state.add_global_memory_writes<OffsetT>(1);

  // Allocate data
  musa_bench::device_vector<T> in(elements);
  musa_bench::device_vector<bool> flags(elements);
  musa_bench::device_vector<OffsetT> num_selected(1);
  musa_bench::device_vector<T> out(elements);

  // Generate random input data
  musa_bench::gen(musa_bench::seed_t{}, in);
  musa_bench::gen_bool(musa_bench::seed_t{1}, flags, entropy);

  input_it_t d_in = in.data();
  flag_it_t d_flags = flags.data();
  output_it_t d_out = out.data();
  num_selected_it_t d_num_selected = num_selected.data();

  // Allocate temporary storage
  std::size_t temp_size = 0;
  dispatch_t::Dispatch(nullptr,
                       temp_size,
                       d_in,
                       d_flags,
                       d_out,
                       d_num_selected,
                       select_op_t{},
                       equality_op_t{},
                       elements,
                       0,
                       false /* debug_synchronous */);

  musa_bench::device_vector<uint8_t> temp(temp_size);
  auto *temp_storage = temp.data();

  // Create timer
  musa_bench::Timer timer(state.stream);

  // Warmup
  for (int i = 0; i < state.warmup_iterations; i++) {
    dispatch_t::Dispatch(temp_storage,
                         temp_size,
                         d_in,
                         d_flags,
                         d_out,
                         d_num_selected,
                         select_op_t{},
                         equality_op_t{},
                         elements,
                         state.stream,
                         false /* debug_synchronous */);
  }
  MUSA_BENCH_CHECK(musaDeviceSynchronize());

  // Benchmark
  for (int i = 0; i < state.test_iterations; i++) {
    timer.start();
    dispatch_t::Dispatch(temp_storage,
                         temp_size,
                         d_in,
                         d_flags,
                         d_out,
                         d_num_selected,
                         select_op_t{},
                         equality_op_t{},
                         elements,
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
    std::cout << "Type: " << musa_bench::type_name<T>()
              << ", Entropy: " << entropy_str << std::endl;
    state.print_results();
  }
}

int main(int argc, char **argv)
{
  // Default element count: 2^24 = 16M
  int64_t elements = 16777216;
  std::string entropy = "0.544";
  bool output_json = false;

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if ((arg == "-n" || arg == "--elements") && i + 1 < argc) {
      elements = std::stoll(argv[++i]);
    } else if ((arg == "-e" || arg == "--entropy") && i + 1 < argc) {
      entropy = argv[++i];
    } else if (arg == "--json") {
      output_json = true;
    } else if (arg == "-h" || arg == "--help") {
      std::cout << "Usage: " << argv[0] << " [options]\n"
                << "Options:\n"
                << "  -n, --elements N  Number of elements (default: 16777216)\n"
                << "  -e, --entropy S  Bit entropy: 1.000, 0.544, 0.000 (default: 0.544)\n"
                << "  --json           Output results in JSON format\n"
                << "  -h, --help       Show this help message\n";
      return 0;
    }
  }

  if (!output_json) {
    std::cout << "=== Benchmark: cub::DevicePartition::Flagged ===" << std::endl;
  }

  // Run with int32_t type (most common)
  run_partition_benchmark<int32_t, int32_t>(elements, entropy, output_json);

  return 0;
}