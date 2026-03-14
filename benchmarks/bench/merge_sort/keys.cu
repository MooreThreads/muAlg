/******************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2024, Moore Threads Corporation.  All rights reserved.
 *
 * MUSA port of merge_sort/keys benchmark.
 * Benchmarks cub::DeviceMergeSort::SortKeys
 ******************************************************************************/

#include <musa_bench.cuh>
#include <generator.cuh>
#include <cub/device/device_merge_sort.cuh>

using value_t = cub::NullType;

template <typename KeyT, typename OffsetT>
void run_merge_sort_keys(int64_t elements, musa_bench::bit_entropy entropy, bool output_json)
{
  using key_t            = KeyT;
  using key_input_it_t   = key_t *;
  using value_input_it_t = value_t *;
  using key_it_t         = key_t *;
  using value_it_t       = value_t *;
  using offset_t         = OffsetT;
  using compare_op_t     = musa_bench::less_t;

  using dispatch_t = cub::DispatchMergeSort<key_input_it_t,
                                            value_input_it_t,
                                            key_it_t,
                                            value_it_t,
                                            offset_t,
                                            compare_op_t>;

  // Setup benchmark state
  musa_bench::State state;
  state.add_element_count(elements);
  state.add_global_memory_reads<KeyT>(elements);
  state.add_global_memory_writes<KeyT>(elements);

  // Set benchmark identification for JSON output
  state.benchmark_name = "cub.bench.merge_sort.keys";
  state.type_name = musa_bench::type_name<KeyT>();

  // Allocate data
  musa_bench::device_vector<KeyT> buffer_1(elements);
  musa_bench::device_vector<KeyT> buffer_2(elements);

  // Generate random input data
  musa_bench::gen(musa_bench::seed_t{}, buffer_1, entropy);

  key_t *d_buffer_1 = buffer_1.data();
  key_t *d_buffer_2 = buffer_2.data();

  // Allocate temporary storage:
  std::size_t temp_size{};
  dispatch_t::Dispatch(nullptr,
                       temp_size,
                       d_buffer_1,
                       nullptr,
                       d_buffer_2,
                       nullptr,
                       static_cast<offset_t>(elements),
                       compare_op_t{},
                       0 /* stream */,
                       false /* debug_synchronous */);

  musa_bench::device_vector<uint8_t> temp(temp_size);
  auto *temp_storage = temp.data();

  // Create timer
  musa_bench::Timer timer(state.stream);

  // Warmup
  for (int i = 0; i < state.warmup_iterations; i++) {
    dispatch_t::Dispatch(temp_storage,
                         temp_size,
                         d_buffer_1,
                         nullptr,
                         d_buffer_2,
                         nullptr,
                         static_cast<offset_t>(elements),
                         compare_op_t{},
                         state.stream,
                         false /* debug_synchronous */);
  }
  MUSA_BENCH_CHECK(musaDeviceSynchronize());

  // Benchmark
  for (int i = 0; i < state.test_iterations; i++) {
    timer.start();
    dispatch_t::Dispatch(temp_storage,
                         temp_size,
                         d_buffer_1,
                         nullptr,
                         d_buffer_2,
                         nullptr,
                         static_cast<offset_t>(elements),
                         compare_op_t{},
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
    state.print_results();
  }
}

void print_usage(const char *prog_name)
{
  std::cout << "Usage: " << prog_name << " [options]\n"
            << "Options:\n"
            << "  -n, --elements N    Number of elements (default: 16777216)\n"
            << "  -e, --entropy E     Bit entropy: 1.000, 0.811, 0.544, 0.337, 0.201 (default: 1.000)\n"
            << "  -t, --type T        Data type: int8, int16, int32, int64, float, double (default: int32)\n"
            << "  --json              Output results in JSON format\n"
            << "  -h, --help          Show this help message\n";
}

musa_bench::bit_entropy parse_entropy(const std::string &str)
{
  if (str == "1.000") return musa_bench::bit_entropy::_1_000;
  if (str == "0.811") return musa_bench::bit_entropy::_0_811;
  if (str == "0.544") return musa_bench::bit_entropy::_0_544;
  if (str == "0.337") return musa_bench::bit_entropy::_0_337;
  if (str == "0.201") return musa_bench::bit_entropy::_0_201;
  if (str == "0.000") return musa_bench::bit_entropy::_0_000;
  return musa_bench::bit_entropy::_1_000;
}

int main(int argc, char **argv)
{
  // Default values
  int64_t elements = 16777216;  // 2^24 = 16M
  musa_bench::bit_entropy entropy = musa_bench::bit_entropy::_1_000;
  std::string type_name = "int32";
  bool output_json = false;

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if ((arg == "-n" || arg == "--elements") && i + 1 < argc) {
      elements = std::stoll(argv[++i]);
    } else if ((arg == "-e" || arg == "--entropy") && i + 1 < argc) {
      entropy = parse_entropy(argv[++i]);
    } else if ((arg == "-t" || arg == "--type") && i + 1 < argc) {
      type_name = argv[++i];
    } else if (arg == "--json") {
      output_json = true;
    } else if (arg == "-h" || arg == "--help") {
      print_usage(argv[0]);
      return 0;
    }
  }

  if (!output_json) {
    std::cout << "=== Benchmark: cub::DeviceMergeSort::SortKeys ===" << std::endl;
    std::cout << "Elements: " << elements << std::endl;
    std::cout << "Entropy: " << musa_bench::entropy_to_probability(entropy) << std::endl;
    std::cout << std::endl;
  }

  // Run benchmark based on type
  if (type_name == "int8") {
    run_merge_sort_keys<int8_t, int32_t>(elements, entropy, output_json);
  } else if (type_name == "int16") {
    run_merge_sort_keys<int16_t, int32_t>(elements, entropy, output_json);
  } else if (type_name == "int32") {
    run_merge_sort_keys<int32_t, int32_t>(elements, entropy, output_json);
  } else if (type_name == "int64") {
    run_merge_sort_keys<int64_t, int32_t>(elements, entropy, output_json);
  } else if (type_name == "float") {
    run_merge_sort_keys<float, int32_t>(elements, entropy, output_json);
  } else if (type_name == "double") {
    run_merge_sort_keys<double, int32_t>(elements, entropy, output_json);
  } else {
    std::cerr << "Unknown type: " << type_name << std::endl;
    print_usage(argv[0]);
    return 1;
  }

  return 0;
}