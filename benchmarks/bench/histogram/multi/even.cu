/******************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2024, Moore Threads Corporation.  All rights reserved.
 *
 * MUSA port of histogram/multi/even benchmark - Multi-channel histogram.
 ******************************************************************************/

#include <musa_bench.cuh>
#include <generator.cuh>
#include <cub/device/device_histogram.cuh>
#include <limits>
#include <cmath>

//=============================================================================
// Helper functions for histogram benchmarks
//=============================================================================

template <class SampleT, class OffsetT>
SampleT get_upper_level(OffsetT bins, OffsetT elements)
{
  if constexpr (std::is_integral_v<SampleT>)
  {
    if constexpr (sizeof(SampleT) < sizeof(OffsetT))
    {
      const SampleT max_key = std::numeric_limits<SampleT>::max();
      return static_cast<SampleT>(std::min(bins, static_cast<OffsetT>(max_key)));
    }
    else
    {
      return static_cast<SampleT>(bins);
    }
  }

  return static_cast<SampleT>(elements);
}

//=============================================================================
// Benchmark implementation - Multi-channel histogram (RGB-like)
//=============================================================================

template <typename SampleT, typename CounterT, typename OffsetT>
void run_multi_histogram_even(int64_t elements, int64_t num_bins, double entropy_prob)
{
  // Multi-channel histogram for RGB-like data:
  // 4 channels total (RGBA), but only 3 active channels (RGB)
  constexpr int num_channels        = 4;
  constexpr int num_active_channels = 3;

  using sample_iterator_t = SampleT *;

  const int num_levels_r = static_cast<int>(num_bins) + 1;
  const int num_levels_g = num_levels_r;
  const int num_levels_b = num_levels_g;

  const SampleT lower_level = 0;
  const SampleT upper_level = get_upper_level<SampleT>(num_bins, elements);

  // Setup benchmark state
  musa_bench::State state;
  state.add_element_count(elements);
  state.add_global_memory_reads<SampleT>(elements * num_active_channels);
  state.add_global_memory_writes<CounterT>(num_bins * num_active_channels);

  // Allocate data - RGBA input, RGB histograms
  musa_bench::device_vector<SampleT> input(elements * num_channels);
  musa_bench::device_vector<CounterT> hist_r(num_bins);
  musa_bench::device_vector<CounterT> hist_g(num_bins);
  musa_bench::device_vector<CounterT> hist_b(num_bins);

  // Generate random input data
  musa_bench::bit_entropy entropy = musa_bench::bit_entropy::_1_000;
  if (entropy_prob < 0.1) {
    entropy = musa_bench::bit_entropy::_0_000;
  } else if (entropy_prob < 0.6) {
    entropy = musa_bench::bit_entropy::_0_544;
  }
  musa_bench::gen(musa_bench::seed_t{}, input, entropy, lower_level, upper_level);

  SampleT *d_input        = input.data();
  CounterT *d_histogram_r = hist_r.data();
  CounterT *d_histogram_g = hist_g.data();
  CounterT *d_histogram_b = hist_b.data();

  CounterT *d_histogram[num_active_channels] = {d_histogram_r, d_histogram_g, d_histogram_b};
  int num_levels_arr[num_active_channels]   = {num_levels_r, num_levels_g, num_levels_b};
  SampleT lower_level_arr[num_active_channels] = {lower_level, lower_level, lower_level};
  SampleT upper_level_arr[num_active_channels] = {upper_level, upper_level, upper_level};

  std::uint8_t *d_temp_storage = nullptr;
  std::size_t temp_storage_bytes{};

  OffsetT num_row_pixels  = static_cast<OffsetT>(elements);
  OffsetT num_rows        = 1;
  size_t row_stride_bytes = sizeof(SampleT) * num_channels * num_row_pixels;

  // Query temporary storage
  cub::DeviceHistogram::MultiHistogramEven<num_channels, num_active_channels>(d_temp_storage,
                                            temp_storage_bytes,
                                            d_input,
                                            d_histogram,
                                            num_levels_arr,
                                            lower_level_arr,
                                            upper_level_arr,
                                            num_row_pixels,
                                            num_rows,
                                            row_stride_bytes);

  musa_bench::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = temp_storage.data();

  // Create timer
  musa_bench::Timer timer(state.stream);

  // Warmup
  for (int i = 0; i < state.warmup_iterations; i++) {
    cub::DeviceHistogram::MultiHistogramEven<num_channels, num_active_channels>(d_temp_storage,
                                              temp_storage_bytes,
                                              d_input,
                                              d_histogram,
                                              num_levels_arr,
                                              lower_level_arr,
                                              upper_level_arr,
                                              num_row_pixels,
                                              num_rows,
                                              row_stride_bytes,
                                              state.stream);
  }
  MUSA_BENCH_CHECK(musaDeviceSynchronize());

  // Benchmark
  for (int i = 0; i < state.test_iterations; i++) {
    timer.start();
    cub::DeviceHistogram::MultiHistogramEven<num_channels, num_active_channels>(d_temp_storage,
                                              temp_storage_bytes,
                                              d_input,
                                              d_histogram,
                                              num_levels_arr,
                                              lower_level_arr,
                                              upper_level_arr,
                                              num_row_pixels,
                                              num_rows,
                                              row_stride_bytes,
                                              state.stream);
    timer.stop();

    float ms = timer.elapsed_ms();
    state.total_time_ms += ms;
    state.min_time_ms = std::min(state.min_time_ms, (double)ms);
    state.max_time_ms = std::max(state.max_time_ms, (double)ms);
  }

  // Print results
  std::cout << "Type: " << musa_bench::type_name<SampleT>() << std::endl;
  std::cout << "Counter: " << musa_bench::type_name<CounterT>() << std::endl;
  std::cout << "Elements: " << elements << std::endl;
  std::cout << "Bins: " << num_bins << std::endl;
  std::cout << "Channels: " << num_active_channels << " active / " << num_channels << " total" << std::endl;
  std::cout << "Entropy: " << entropy_prob << std::endl;
  state.print_results();
}

int main(int argc, char **argv)
{
  // Default values
  int64_t elements = 16777216;  // 2^24
  int64_t num_bins = 128;
  double entropy = 1.0;

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if ((arg == "-n" || arg == "--elements") && i + 1 < argc) {
      elements = std::stoll(argv[++i]);
    } else if ((arg == "-b" || arg == "--bins") && i + 1 < argc) {
      num_bins = std::stoll(argv[++i]);
    } else if ((arg == "-e" || arg == "--entropy") && i + 1 < argc) {
      entropy = std::stod(argv[++i]);
    } else if (arg == "-h" || arg == "--help") {
      std::cout << "Usage: " << argv[0] << " [options]\n"
                << "Options:\n"
                << "  -n, --elements N  Number of elements (default: 16777216)\n"
                << "  -b, --bins N      Number of bins (default: 128)\n"
                << "  -e, --entropy F   Bit entropy 0.0-1.0 (default: 1.0)\n"
                << "  -h, --help        Show this help message\n";
      return 0;
    }
  }

  std::cout << "=== Benchmark: cub::DeviceHistogram::MultiHistogramEven ===" << std::endl;

  // Run benchmark with int32_t sample type (most common)
  run_multi_histogram_even<int32_t, int32_t, int32_t>(elements, num_bins, entropy);

  return 0;
}