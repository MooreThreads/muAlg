/******************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2024, Moore Threads Corporation.  All rights reserved.
 *
 * MUSA port of merge_sort/pairs benchmark.
 * Benchmarks cub::DeviceMergeSort::SortPairs
 ******************************************************************************/

#include <musa_bench.cuh>
#include <generator.cuh>
#include <cub/device/device_merge_sort.cuh>

template <typename KeyT, typename ValueT, typename OffsetT>
void run_merge_sort_pairs(int64_t elements, musa_bench::bit_entropy entropy, bool output_json)
{
  using key_t            = KeyT;
  using value_t          = ValueT;
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
  state.benchmark_name = "cub.bench.merge_sort_pairs";
  state.type_name = musa_bench::type_name<KeyT>();
  state.add_element_count(elements);
  state.add_global_memory_reads<KeyT>(elements);
  state.add_global_memory_reads<ValueT>(elements);
  state.add_global_memory_writes<KeyT>(elements);
  state.add_global_memory_writes<ValueT>(elements);

  // Allocate data
  musa_bench::device_vector<KeyT> keys_buffer_1(elements);
  musa_bench::device_vector<KeyT> keys_buffer_2(elements);
  musa_bench::device_vector<ValueT> values_buffer_1(elements);
  musa_bench::device_vector<ValueT> values_buffer_2(elements);

  // Generate random input data
  musa_bench::gen(musa_bench::seed_t{}, keys_buffer_1, entropy);

  key_t *d_keys_buffer_1   = keys_buffer_1.data();
  key_t *d_keys_buffer_2   = keys_buffer_2.data();
  value_t *d_values_buffer_1 = values_buffer_1.data();
  value_t *d_values_buffer_2 = values_buffer_2.data();

  // Allocate temporary storage:
  std::size_t temp_size{};
  dispatch_t::Dispatch(nullptr,
                       temp_size,
                       d_keys_buffer_1,
                       d_values_buffer_1,
                       d_keys_buffer_2,
                       d_values_buffer_2,
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
                         d_keys_buffer_1,
                         d_values_buffer_1,
                         d_keys_buffer_2,
                         d_values_buffer_2,
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
                         d_keys_buffer_1,
                         d_values_buffer_1,
                         d_keys_buffer_2,
                         d_values_buffer_2,
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
    std::cout << "KeyType: " << musa_bench::type_name<KeyT>() << std::endl;
    std::cout << "ValueType: " << musa_bench::type_name<ValueT>() << std::endl;
    state.print_results();
  }
}

void print_usage(const char *prog_name)
{
  std::cout << "Usage: " << prog_name << " [options]\n"
            << "Options:\n"
            << "  -n, --elements N    Number of elements (default: 16777216)\n"
            << "  -e, --entropy E     Bit entropy: 1.000, 0.811, 0.544, 0.337, 0.201 (default: 1.000)\n"
            << "  -k, --key-type T    Key type: int8, int16, int32, int64, float, double (default: int32)\n"
            << "  -v, --value-type T  Value type: int8, int16, int32, int64 (default: int32)\n"
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

// Helper template to run benchmark with different value types
template <typename KeyT>
void run_with_value_type(const std::string &value_type, int64_t elements, musa_bench::bit_entropy entropy, bool output_json)
{
  if (value_type == "int8") {
    run_merge_sort_pairs<KeyT, int8_t, int32_t>(elements, entropy, output_json);
  } else if (value_type == "int16") {
    run_merge_sort_pairs<KeyT, int16_t, int32_t>(elements, entropy, output_json);
  } else if (value_type == "int32") {
    run_merge_sort_pairs<KeyT, int32_t, int32_t>(elements, entropy, output_json);
  } else if (value_type == "int64") {
    run_merge_sort_pairs<KeyT, int64_t, int32_t>(elements, entropy, output_json);
  } else {
    std::cerr << "Unknown value type: " << value_type << std::endl;
  }
}

int main(int argc, char **argv)
{
  // Default values
  int64_t elements = 16777216;  // 2^24 = 16M
  musa_bench::bit_entropy entropy = musa_bench::bit_entropy::_1_000;
  std::string key_type = "int32";
  std::string value_type = "int32";
  bool output_json = false;

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if ((arg == "-n" || arg == "--elements") && i + 1 < argc) {
      elements = std::stoll(argv[++i]);
    } else if ((arg == "-e" || arg == "--entropy") && i + 1 < argc) {
      entropy = parse_entropy(argv[++i]);
    } else if ((arg == "-k" || arg == "--key-type") && i + 1 < argc) {
      key_type = argv[++i];
    } else if ((arg == "-v" || arg == "--value-type") && i + 1 < argc) {
      value_type = argv[++i];
    } else if (arg == "--json") {
      output_json = true;
    } else if (arg == "-h" || arg == "--help") {
      print_usage(argv[0]);
      return 0;
    }
  }

  if (!output_json) {
    std::cout << "=== Benchmark: cub::DeviceMergeSort::SortPairs ===" << std::endl;
    std::cout << "Elements: " << elements << std::endl;
    std::cout << "Entropy: " << musa_bench::entropy_to_probability(entropy) << std::endl;
    std::cout << std::endl;
  }

  // Run benchmark based on key type
  if (key_type == "int8") {
    run_with_value_type<int8_t>(value_type, elements, entropy, output_json);
  } else if (key_type == "int16") {
    run_with_value_type<int16_t>(value_type, elements, entropy, output_json);
  } else if (key_type == "int32") {
    run_with_value_type<int32_t>(value_type, elements, entropy, output_json);
  } else if (key_type == "int64") {
    run_with_value_type<int64_t>(value_type, elements, entropy, output_json);
  } else if (key_type == "float") {
    run_with_value_type<float>(value_type, elements, entropy, output_json);
  } else if (key_type == "double") {
    run_with_value_type<double>(value_type, elements, entropy, output_json);
  } else {
    std::cerr << "Unknown key type: " << key_type << std::endl;
    print_usage(argv[0]);
    return 1;
  }

  return 0;
}