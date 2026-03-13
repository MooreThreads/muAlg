/******************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2024, Moore Threads Corporation.  All rights reserved.
 *
 * Random data generation for MUSA benchmarks.
 * Replaces cuRAND with simple XORWOW-based generator.
 ******************************************************************************/

#pragma once

#include "musa_bench.cuh"
#include <musa_runtime.h>
#include <cmath>
#include <limits>
#include <cstring>

namespace musa_bench {

//=============================================================================
// XORWOW random number generator - runs on GPU
//=============================================================================

// XORWOW state for each thread
struct xorwow_state {
  uint32_t a, b, c, d, e;
  uint32_t counter;
};

// Initialize XORWOW state
__device__ __host__ inline void xorwow_init(xorwow_state *state, uint64_t seed) {
  // Simple initialization using seed
  uint32_t s = (uint32_t)(seed & 0xFFFFFFFF);
  state->a = 0x3C69BE41 ^ s;
  state->b = 0x6BBCC787 ^ (s >> 8);
  state->c = 0x4B5A3BE9 ^ (s >> 16);
  state->d = 0x6E8A4D1B ^ (s >> 24);
  state->e = 0x5F72B3C9 ^ s;
  state->counter = 0;
}

// XORWOW next random number
__device__ __host__ inline uint32_t xorwow_next(xorwow_state *state) {
  uint32_t t = state->d;
  uint32_t s = state->a;
  state->d = state->c;
  state->c = state->b;
  state->b = s;
  t ^= t >> 2;
  t ^= t << 1;
  state->a = t ^ s ^ (s << 4);
  state->counter += 362437;
  return state->a + state->counter;
}

//=============================================================================
// Device kernels for random number generation
//=============================================================================

// Generate random uint32_t
__global__ void generate_random_uint32_kernel(uint32_t *output, int64_t n,
                                              uint64_t seed) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  xorwow_state state;
  xorwow_init(&state, seed + idx);

  // Warm up the generator
  for (int i = 0; i < 20; i++) {
    xorwow_next(&state);
  }

  output[idx] = xorwow_next(&state);
}

// Generate random float [0, 1)
__global__ void generate_random_float_kernel(float *output, int64_t n,
                                             uint64_t seed, float min_val,
                                             float max_val) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  xorwow_state state;
  xorwow_init(&state, seed + idx);

  // Warm up
  for (int i = 0; i < 20; i++) {
    xorwow_next(&state);
  }

  uint32_t r = xorwow_next(&state);
  // Convert to float in [0, 1)
  float f = (float)r / (float)UINT32_MAX;
  output[idx] = min_val + f * (max_val - min_val);
}

// Generate random int64_t
__global__ void generate_random_int64_kernel(int64_t *output, int64_t n,
                                             uint64_t seed, int64_t min_val,
                                             int64_t max_val) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  xorwow_state state;
  xorwow_init(&state, seed + idx);

  // Warm up
  for (int i = 0; i < 20; i++) {
    xorwow_next(&state);
  }

  uint32_t r1 = xorwow_next(&state);
  uint32_t r2 = xorwow_next(&state);
  uint64_t r = ((uint64_t)r1 << 32) | r2;

  output[idx] = min_val + (int64_t)(r % (uint64_t)(max_val - min_val + 1));
}

// Generate random int32_t
__global__ void generate_random_int32_kernel(int32_t *output, int64_t n,
                                             uint64_t seed, int32_t min_val,
                                             int32_t max_val) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  xorwow_state state;
  xorwow_init(&state, seed + idx);

  for (int i = 0; i < 20; i++) {
    xorwow_next(&state);
  }

  uint32_t r = xorwow_next(&state);
  output[idx] = min_val + (int32_t)(r % (uint32_t)(max_val - min_val + 1));
}

// Generate random int16_t
__global__ void generate_random_int16_kernel(int16_t *output, int64_t n,
                                             uint64_t seed, int16_t min_val,
                                             int16_t max_val) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  xorwow_state state;
  xorwow_init(&state, seed + idx);

  for (int i = 0; i < 20; i++) {
    xorwow_next(&state);
  }

  uint32_t r = xorwow_next(&state);
  output[idx] = min_val + (int16_t)(r % (uint16_t)(max_val - min_val + 1));
}

// Generate random int8_t
__global__ void generate_random_int8_kernel(int8_t *output, int64_t n,
                                            uint64_t seed, int8_t min_val,
                                            int8_t max_val) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  xorwow_state state;
  xorwow_init(&state, seed + idx);

  for (int i = 0; i < 20; i++) {
    xorwow_next(&state);
  }

  uint32_t r = xorwow_next(&state);
  output[idx] = min_val + (int8_t)(r % (uint8_t)(max_val - min_val + 1));
}

// Generate random double
__global__ void generate_random_double_kernel(double *output, int64_t n,
                                              uint64_t seed, double min_val,
                                              double max_val) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  xorwow_state state;
  xorwow_init(&state, seed + idx);

  for (int i = 0; i < 20; i++) {
    xorwow_next(&state);
  }

  uint32_t r1 = xorwow_next(&state);
  uint32_t r2 = xorwow_next(&state);
  uint64_t r = ((uint64_t)r1 << 32) | r2;

  double f = (double)r / (double)UINT64_MAX;
  output[idx] = min_val + f * (max_val - min_val);
}

// Generate sorted data
template <typename T>
__global__ void generate_sorted_kernel(T *output, int64_t n, T start_val) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;
  output[idx] = start_val + (T)idx;
}

// Generate reverse sorted data
template <typename T>
__global__ void generate_reverse_kernel(T *output, int64_t n, T start_val) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;
  output[idx] = start_val - (T)idx;
}

//=============================================================================
// Host interface for random data generation
//=============================================================================

inline int get_launch_blocks(int64_t n, int threads_per_block = 256) {
  return (n + threads_per_block - 1) / threads_per_block;
}

template <typename T>
void generate_random(T *data, int64_t n, uint64_t seed,
                     bit_entropy entropy = bit_entropy::_1_000,
                     T min_val = std::numeric_limits<T>::min(),
                     T max_val = std::numeric_limits<T>::max()) {
  const int threads = 256;
  const int blocks = get_launch_blocks(n, threads);

  // Adjust seed based on entropy for reproducibility
  uint64_t adjusted_seed = seed + static_cast<uint64_t>(entropy);

  if constexpr (std::is_same_v<T, float>) {
    generate_random_float_kernel<<<blocks, threads>>>(data, n, adjusted_seed,
                                                       (float)min_val,
                                                       (float)max_val);
  } else if constexpr (std::is_same_v<T, double>) {
    generate_random_double_kernel<<<blocks, threads>>>(data, n, adjusted_seed,
                                                        (double)min_val,
                                                        (double)max_val);
  } else if constexpr (std::is_same_v<T, int64_t>) {
    generate_random_int64_kernel<<<blocks, threads>>>(data, n, adjusted_seed,
                                                       min_val, max_val);
  } else if constexpr (std::is_same_v<T, int32_t>) {
    generate_random_int32_kernel<<<blocks, threads>>>(data, n, adjusted_seed,
                                                       min_val, max_val);
  } else if constexpr (std::is_same_v<T, int16_t>) {
    generate_random_int16_kernel<<<blocks, threads>>>(data, n, adjusted_seed,
                                                       min_val, max_val);
  } else if constexpr (std::is_same_v<T, int8_t>) {
    generate_random_int8_kernel<<<blocks, threads>>>(data, n, adjusted_seed,
                                                      min_val, max_val);
  } else {
    // Generic fallback using byte-wise random generation
    int64_t bytes = n * sizeof(T);
    uint8_t *byte_ptr = reinterpret_cast<uint8_t *>(data);
    generate_random_int8_kernel<<<get_launch_blocks(bytes), threads>>>(
        (int8_t *)byte_ptr, bytes, adjusted_seed, 0, 127);
  }

  MUSA_BENCH_CHECK(musaGetLastError());
  MUSA_BENCH_CHECK(musaDeviceSynchronize());
}

// Specialized generation with bit entropy (simplified - uses different seeds)
template <typename T>
void generate(T *data, int64_t n, seed_t seed,
              bit_entropy entropy = bit_entropy::_1_000,
              T min_val = std::numeric_limits<T>::min(),
              T max_val = std::numeric_limits<T>::max()) {
  generate_random(data, n, seed.get(), entropy, min_val, max_val);
}

// Generate sorted sequence
template <typename T> void generate_sorted(T *data, int64_t n, T start_val = 0) {
  const int threads = 256;
  const int blocks = get_launch_blocks(n, threads);
  generate_sorted_kernel<<<blocks, threads>>>(data, n, start_val);
  MUSA_BENCH_CHECK(musaGetLastError());
  MUSA_BENCH_CHECK(musaDeviceSynchronize());
}

// Generate reverse sorted sequence
template <typename T>
void generate_reverse(T *data, int64_t n, T start_val = 0) {
  const int threads = 256;
  const int blocks = get_launch_blocks(n, threads);
  generate_reverse_kernel<<<blocks, threads>>>(data, n, start_val);
  MUSA_BENCH_CHECK(musaGetLastError());
  MUSA_BENCH_CHECK(musaDeviceSynchronize());
}

//=============================================================================
// Convenience wrapper for device_vector
//=============================================================================

template <typename T>
void gen(seed_t seed, device_vector<T> &data,
         bit_entropy entropy = bit_entropy::_1_000,
         T min_val = std::numeric_limits<T>::min(),
         T max_val = std::numeric_limits<T>::max()) {
  generate(data.data(), data.size(), seed, entropy, min_val, max_val);
}

//=============================================================================
// Generate power-law distributed offsets for segmented operations
//=============================================================================

// Simplified power-law offset generation using host-side generation
// and copying to device
template <typename T>
device_vector<T> gen_power_law_offsets(seed_t seed, size_t total_elements,
                                       size_t total_segments) {
  device_vector<T> offsets(total_segments + 1);

  // Generate on host for simplicity (power-law is complex)
  std::vector<T> h_offsets(total_segments + 1);
  T segment_size = total_elements / total_segments;

  // Simple uniform distribution for now (power-law requires more complex
  // implementation)
  for (size_t i = 0; i <= total_segments; i++) {
    h_offsets[i] = std::min(static_cast<T>(i * segment_size),
                            static_cast<T>(total_elements));
  }
  h_offsets[total_segments] = static_cast<T>(total_elements);

  MUSA_BENCH_CHECK(musaMemcpy(offsets.data(), h_offsets.data(),
                               (total_segments + 1) * sizeof(T),
                               musaMemcpyHostToDevice));
  return offsets;
}

template <typename T>
device_vector<T> gen_uniform_offsets(seed_t seed, T total_elements,
                                     T min_segment_size, T max_segment_size) {
  // Simplified: generate uniform segments
  T avg_segment_size = (min_segment_size + max_segment_size) / 2;
  size_t num_segments = total_elements / avg_segment_size;

  device_vector<T> offsets(num_segments + 1);

  std::vector<T> h_offsets(num_segments + 1);
  for (size_t i = 0; i <= num_segments; i++) {
    h_offsets[i] = std::min(static_cast<T>(i * avg_segment_size),
                            static_cast<T>(total_elements));
  }
  h_offsets[num_segments] = total_elements;

  MUSA_BENCH_CHECK(musaMemcpy(offsets.data(), h_offsets.data(),
                               (num_segments + 1) * sizeof(T),
                               musaMemcpyHostToDevice));
  return offsets;
}

//=============================================================================
// Type traits for accumulator types
//=============================================================================

template <typename T> struct accumulator_type { using type = T; };

template <> struct accumulator_type<int8_t> { using type = int32_t; };
template <> struct accumulator_type<uint8_t> { using type = uint32_t; };
template <> struct accumulator_type<int16_t> { using type = int32_t; };
template <> struct accumulator_type<uint16_t> { using type = uint32_t; };

template <typename T>
using accumulator_type_t = typename accumulator_type<T>::type;

//=============================================================================
// Comparison operators (for use with custom types)
//=============================================================================

struct less_t {
  template <typename DataType>
  __device__ bool operator()(const DataType &lhs, const DataType &rhs) {
    return lhs < rhs;
  }
};

struct max_t {
  template <typename DataType>
  __device__ DataType operator()(const DataType &lhs, const DataType &rhs) {
    return lhs < rhs ? rhs : lhs;
  }
};

struct min_t {
  template <typename DataType>
  __device__ DataType operator()(const DataType &lhs, const DataType &rhs) {
    return lhs < rhs ? lhs : rhs;
  }
};

} // namespace musa_bench