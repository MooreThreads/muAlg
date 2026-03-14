/******************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2024, Moore Threads Corporation.  All rights reserved.
 *
 * MUSA port of scan/exclusive/by_key benchmark.
 ******************************************************************************/

#include <musa_bench.cuh>
#include <generator.cuh>
#include <cub/device/device_scan.cuh>

using op_t = cub::Sum;

//=============================================================================
// Helper kernel to generate uniform key segments
//=============================================================================

template <typename KeyT>
__global__ void generate_key_segments_kernel(KeyT *keys, int64_t n, int64_t segment_size)
{
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;
  keys[idx] = static_cast<KeyT>(idx / segment_size);
}

template <typename KeyT>
void generate_key_segments(musa_bench::device_vector<KeyT> &keys, int64_t n, int64_t segment_size)
{
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  generate_key_segments_kernel<<<blocks, threads>>>(keys.data(), n, segment_size);
  MUSA_BENCH_CHECK(musaGetLastError());
  MUSA_BENCH_CHECK(musaDeviceSynchronize());
}

//=============================================================================
// Benchmark implementation
//=============================================================================

template <typename KeyT, typename ValueT>
void run_benchmark(int64_t elements, int64_t segment_size)
{
  using init_value_t    = ValueT;
  using key_input_it_t  = const KeyT *;
  using val_input_it_t  = const ValueT *;
  using val_output_it_t = ValueT *;
  using equality_op_t   = cub::Equality;
  using offset_t        = int;

  // Setup benchmark state
  musa_bench::State state;
  state.add_element_count(elements);
  state.add_global_memory_reads<KeyT>(elements);    // Keys
  state.add_global_memory_reads<ValueT>(elements);  // Values
  state.add_global_memory_writes<ValueT>(elements); // Output values

  // Allocate data
  musa_bench::device_vector<KeyT> keys(elements);
  musa_bench::device_vector<ValueT> in_vals(elements);
  musa_bench::device_vector<ValueT> out_vals(elements);

  // Generate key segments
  generate_key_segments(keys, elements, segment_size);

  // Generate random values
  musa_bench::gen(musa_bench::seed_t{}, in_vals);

  key_input_it_t d_keys     = keys.data();
  val_input_it_t d_in_vals  = in_vals.data();
  val_output_it_t d_out_vals = out_vals.data();

  // Allocate temporary storage
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  cub::DeviceScan::ExclusiveScanByKey(d_temp_storage, temp_storage_bytes,
                                       d_keys, d_in_vals, d_out_vals,
                                       op_t{}, init_value_t{},
                                       static_cast<offset_t>(elements),
                                       equality_op_t{});

  musa_bench::device_vector<uint8_t> temp(temp_storage_bytes);
  d_temp_storage = temp.data();

  // Create timer
  musa_bench::Timer timer(state.stream);

  // Warmup
  for (int i = 0; i < state.warmup_iterations; i++) {
    cub::DeviceScan::ExclusiveScanByKey(d_temp_storage, temp_storage_bytes,
                                         d_keys, d_in_vals, d_out_vals,
                                         op_t{}, init_value_t{},
                                         static_cast<offset_t>(elements),
                                         equality_op_t{}, state.stream);
  }
  MUSA_BENCH_CHECK(musaDeviceSynchronize());

  // Benchmark
  for (int i = 0; i < state.test_iterations; i++) {
    timer.start();
    cub::DeviceScan::ExclusiveScanByKey(d_temp_storage, temp_storage_bytes,
                                         d_keys, d_in_vals, d_out_vals,
                                         op_t{}, init_value_t{},
                                         static_cast<offset_t>(elements),
                                         equality_op_t{}, state.stream);
    timer.stop();

    float ms = timer.elapsed_ms();
    state.total_time_ms += ms;
    state.min_time_ms = std::min(state.min_time_ms, (double)ms);
    state.max_time_ms = std::max(state.max_time_ms, (double)ms);
  }

  // Print results
  std::cout << "KeyType: " << musa_bench::type_name<KeyT>()
            << ", ValueType: " << musa_bench::type_name<ValueT>()
            << ", SegmentSize: " << segment_size << std::endl;
  state.print_results();
}

int main(int argc, char **argv)
{
  // Default element count: 2^24 = 16M
  int64_t elements = 16777216;
  int64_t segment_size = 5200;  // Default segment size (matches original benchmark)

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if ((arg == "-n" || arg == "--elements") && i + 1 < argc) {
      elements = std::stoll(argv[++i]);
    } else if ((arg == "-s" || arg == "--segment-size") && i + 1 < argc) {
      segment_size = std::stoll(argv[++i]);
    } else if (arg == "-h" || arg == "--help") {
      std::cout << "Usage: " << argv[0] << " [options]\n"
                << "Options:\n"
                << "  -n, --elements N     Number of elements (default: 16777216)\n"
                << "  -s, --segment-size S Segment size (default: 5200)\n"
                << "  -h, --help           Show this help message\n";
      return 0;
    }
  }

  std::cout << "=== Benchmark: cub::DeviceScan::ExclusiveScanByKey ===" << std::endl;

  // Run benchmark with int32_t key and value types (most common)
  run_benchmark<int32_t, int32_t>(elements, segment_size);

  return 0;
}