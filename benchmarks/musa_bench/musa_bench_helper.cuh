/******************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2024, Moore Threads Corporation.  All rights reserved.
 *
 * MUSA benchmark helper - replacement for nvbench_helper.cuh
 * Provides compatibility layer for porting MUSA benchmarks to MUSA.
 ******************************************************************************/

#pragma once

#include <musa_bench.cuh>
#include <generator.cuh>

// Type aliases for compatibility
using complex = std::complex<float>;
using int128_t = __int128_t;
using uint128_t = __uint128_t;

// Type name macros (simplified)
#define MUSA_BENCH_DECLARE_TYPE_STRINGS(type, short_name, long_name)

namespace musa_bench {

// Type lists for benchmarks
using all_types = type_list<int8_t, int16_t, int32_t, int64_t, float, double>;

#ifdef TUNE_OffsetT
using offset_types = type_list<TUNE_OffsetT>;
#else
using offset_types = type_list<int32_t, int64_t>;
#endif

#ifdef TUNE_T
using fundamental_types = type_list<TUNE_T>;
using tuning_all_types = type_list<TUNE_T>;
#else
using fundamental_types = type_list<int8_t, int16_t, int32_t, int64_t, float, double>;
using tuning_all_types = all_types;
#endif

//=============================================================================
// Compatibility layer for nvbench API
//=============================================================================

// nvbench::state replacement
using nvbench_state = State;

// nvbench::launch replacement
using nvbench_launch = launch;

// nvbench::type_list (already defined as type_list)

// nvbench::range (already defined as range)

//=============================================================================
// Benchmark macro replacements
//=============================================================================

// NVBENCH_BENCH_TYPES replacement
// Instead of complex type parameterization, we use a simple function call
#define NVBENCH_BENCH_TYPES(func, type_axes)                                   \
  int main(int argc, char **argv) {                                            \
    (void)argc;                                                                \
    (void)argv;                                                                \
    std::cout << "=== Running benchmark: " << #func << " ===" << std::endl;   \
    run_##func##_all_types();                                                  \
    return 0;                                                                  \
  }

//=============================================================================
// Additional utility functions
//=============================================================================

// Raw pointer access (similar to thrust::raw_pointer_cast)
template <typename T>
T* raw_pointer_cast(device_vector<T>& vec) {
  return vec.data();
}

template <typename T>
const T* raw_pointer_cast(const device_vector<T>& vec) {
  return vec.data();
}

} // namespace musa_bench

// Namespace alias for compatibility
namespace nvbench = musa_bench;