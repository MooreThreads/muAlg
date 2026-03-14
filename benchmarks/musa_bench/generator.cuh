/******************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2024, Moore Threads Corporation.  All rights reserved.
 *
 * Random data generation for MUSA benchmarks using muRAND.
 ******************************************************************************/

#pragma once

#include "musa_bench.cuh"
#include <musa_runtime.h>
#include <murand.h>
#include <murand_kernel.h>
#include <cmath>
#include <limits>
#include <cstring>
#include <vector>
#include <memory>

namespace musa_bench {

//=============================================================================
// muRAND generator wrapper - RAII style
//=============================================================================

class MurandGenerator {
public:
  MurandGenerator(uint64_t seed = 0) {
    murandCreateGenerator(&gen_, MURAND_RNG_PSEUDO_XORWOW);
    murandSetPseudoRandomGeneratorSeed(gen_, seed);
  }

  ~MurandGenerator() {
    if (gen_) {
      murandDestroyGenerator(gen_);
    }
  }

  MurandGenerator(const MurandGenerator&) = delete;
  MurandGenerator& operator=(const MurandGenerator&) = delete;

  MurandGenerator(MurandGenerator&& other) noexcept : gen_(other.gen_) {
    other.gen_ = nullptr;
  }

  void set_seed(uint64_t seed) {
    murandSetPseudoRandomGeneratorSeed(gen_, seed);
  }

  void set_stream(musaStream_t stream) {
    murandSetStream(gen_, (MUstream)stream);
  }

  void generate_uniform(float* d_output, size_t n) {
    murandGenerateUniform(gen_, d_output, n);
  }

  void generate_uniform(double* d_output, size_t n) {
    murandGenerateUniformDouble(gen_, d_output, n);
  }

  void generate(unsigned int* d_output, size_t n) {
    murandGenerate(gen_, d_output, n);
  }

  void generate(unsigned long long* d_output, size_t n) {
    murandGenerateLongLong(gen_, d_output, n);
  }

  murandGenerator_t handle() const { return gen_; }

private:
  murandGenerator_t gen_ = nullptr;
};

//=============================================================================
// Device kernels for scaling random values
//=============================================================================

template <typename T>
__global__ void scale_uniform_kernel(T *data, int64_t n, T min_val, T max_val) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  data[idx] = min_val + data[idx] * (max_val - min_val);
}

template <typename T>
__global__ void scale_int_kernel(unsigned int *rand_vals, T *output, int64_t n,
                                  T min_val, uint64_t range) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  output[idx] = min_val + static_cast<T>(rand_vals[idx] % range);
}

//=============================================================================
// Host interface for random data generation using muRAND
//=============================================================================

inline int get_launch_blocks(int64_t n, int threads_per_block = 256) {
  return (n + threads_per_block - 1) / threads_per_block;
}

// Generate random float using muRAND
inline void generate_random(float *d_data, int64_t n, uint64_t seed,
                            float min_val, float max_val) {
  MurandGenerator gen(seed);
  gen.generate_uniform(d_data, n);

  const int threads = 256;
  const int blocks = get_launch_blocks(n, threads);
  scale_uniform_kernel<<<blocks, threads>>>(d_data, n, min_val, max_val);
  MUSA_BENCH_CHECK(musaGetLastError());
  MUSA_BENCH_CHECK(musaDeviceSynchronize());
}

// Generate random double using muRAND
inline void generate_random(double *d_data, int64_t n, uint64_t seed,
                            double min_val, double max_val) {
  MurandGenerator gen(seed);
  gen.generate_uniform(d_data, n);

  const int threads = 256;
  const int blocks = get_launch_blocks(n, threads);
  scale_uniform_kernel<<<blocks, threads>>>(d_data, n, min_val, max_val);
  MUSA_BENCH_CHECK(musaGetLastError());
  MUSA_BENCH_CHECK(musaDeviceSynchronize());
}

// Generate random int32 using muRAND
inline void generate_random(int32_t *d_data, int64_t n, uint64_t seed,
                            int32_t min_val, int32_t max_val) {
  MurandGenerator gen(seed);

  // Temp buffer for random ints
  unsigned int* d_rand;
  MUSA_BENCH_CHECK(musaMalloc(&d_rand, n * sizeof(unsigned int)));
  gen.generate(d_rand, n);

  const int threads = 256;
  const int blocks = get_launch_blocks(n, threads);
  uint64_t range = static_cast<uint64_t>(max_val) - static_cast<uint64_t>(min_val) + 1;
  scale_int_kernel<<<blocks, threads>>>(d_rand, d_data, n, min_val, range);

  MUSA_BENCH_CHECK(musaGetLastError());
  MUSA_BENCH_CHECK(musaFree(d_rand));
  MUSA_BENCH_CHECK(musaDeviceSynchronize());
}

// Generate random int64 using muRAND
inline void generate_random(int64_t *d_data, int64_t n, uint64_t seed,
                            int64_t min_val, int64_t max_val) {
  MurandGenerator gen(seed);

  unsigned long long* d_rand;
  MUSA_BENCH_CHECK(musaMalloc(&d_rand, n * sizeof(unsigned long long)));
  gen.generate(d_rand, n);

  // Simple kernel for int64
  const int threads = 256;
  const int blocks = get_launch_blocks(n, threads);
  __int128 range = static_cast<__int128>(max_val) - static_cast<__int128>(min_val) + 1;

  auto kernel = [=] __device__(unsigned long long *rand_vals, int64_t *output, int64_t count) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    output[idx] = min_val + static_cast<int64_t>(rand_vals[idx] % static_cast<unsigned long long>(range));
  };

  MUSA_BENCH_CHECK(musaGetLastError());
  MUSA_BENCH_CHECK(musaFree(d_rand));
  MUSA_BENCH_CHECK(musaDeviceSynchronize());
}

// Generic fallback using XORWOW (for other types)
template <typename T>
void generate_random(T *d_data, int64_t n, uint64_t seed,
                     bit_entropy entropy = bit_entropy::_1_000,
                     T min_val = std::numeric_limits<T>::min(),
                     T max_val = std::numeric_limits<T>::max()) {
  // Use host-side generation for simplicity
  std::vector<uint8_t> h_data(n * sizeof(T));

  // Simple host-side XORWOW
  uint64_t s = seed + static_cast<uint64_t>(entropy);
  uint32_t a = 0x3C69BE41 ^ (uint32_t)s;
  uint32_t b = 0x6BBCC787 ^ (uint32_t)(s >> 8);
  uint32_t c = 0x4B5A3BE9 ^ (uint32_t)(s >> 16);
  uint32_t d = 0x6E8A4D1B ^ (uint32_t)(s >> 24);
  uint32_t e = 0x5F72B3C9 ^ (uint32_t)s;
  uint32_t counter = 0;

  auto xorwow_next = [&]() -> uint32_t {
    uint32_t t = d;
    uint32_t ss = a;
    d = c;
    c = b;
    b = ss;
    t ^= t >> 2;
    t ^= t << 1;
    a = t ^ ss ^ (ss << 4);
    counter += 362437;
    return a + counter;
  };

  // Warm up
  for (int i = 0; i < 20; i++) xorwow_next();

  // Generate random bytes
  uint8_t* bytes = h_data.data();
  for (size_t i = 0; i < h_data.size(); i += 4) {
    uint32_t r = xorwow_next();
    if (i + 4 <= h_data.size()) {
      memcpy(&bytes[i], &r, 4);
    } else {
      memcpy(&bytes[i], &r, h_data.size() - i);
    }
  }

  MUSA_BENCH_CHECK(musaMemcpy(d_data, h_data.data(), n * sizeof(T), musaMemcpyHostToDevice));
  MUSA_BENCH_CHECK(musaDeviceSynchronize());
}

// Specialized generation with bit entropy (for float)
inline void generate(float *data, int64_t n, seed_t seed,
                     bit_entropy entropy, float min_val, float max_val) {
  generate_random(data, n, seed.get(), min_val, max_val);
}

inline void generate(double *data, int64_t n, seed_t seed,
                     bit_entropy entropy, double min_val, double max_val) {
  generate_random(data, n, seed.get(), min_val, max_val);
}

inline void generate(int32_t *data, int64_t n, seed_t seed,
                     bit_entropy entropy, int32_t min_val, int32_t max_val) {
  generate_random(data, n, seed.get(), min_val, max_val);
}

inline void generate(int64_t *data, int64_t n, seed_t seed,
                     bit_entropy entropy, int64_t min_val, int64_t max_val) {
  generate_random(data, n, seed.get(), min_val, max_val);
}

// Generic template for other types
template <typename T>
void generate(T *data, int64_t n, seed_t seed,
              bit_entropy entropy = bit_entropy::_1_000,
              T min_val = std::numeric_limits<T>::min(),
              T max_val = std::numeric_limits<T>::max()) {
  generate_random(data, n, seed.get(), entropy, min_val, max_val);
}

//=============================================================================
// Device kernels for special patterns
//=============================================================================

template <typename T>
__global__ void generate_sorted_kernel(T *output, int64_t n, T start_val) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  output[idx] = start_val + (T)idx;
}

template <typename T>
__global__ void generate_reverse_kernel(T *output, int64_t n, T start_val) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  output[idx] = start_val - (T)idx;
}

template <typename T> void generate_sorted(T *data, int64_t n, T start_val = 0) {
  const int threads = 256;
  const int blocks = get_launch_blocks(n, threads);
  generate_sorted_kernel<<<blocks, threads>>>(data, n, start_val);
  MUSA_BENCH_CHECK(musaGetLastError());
  MUSA_BENCH_CHECK(musaDeviceSynchronize());
}

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
// Generate offsets for segmented operations
//=============================================================================

template <typename T>
device_vector<T> gen_power_law_offsets(seed_t seed, size_t total_elements,
                                       size_t total_segments) {
  device_vector<T> offsets(total_segments + 1);
  std::vector<T> h_offsets(total_segments + 1);
  T segment_size = total_elements / total_segments;

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
// Type traits
//=============================================================================

template <typename T> struct accumulator_type { using type = T; };
template <> struct accumulator_type<int8_t> { using type = int32_t; };
template <> struct accumulator_type<uint8_t> { using type = uint32_t; };
template <> struct accumulator_type<int16_t> { using type = int32_t; };
template <> struct accumulator_type<uint16_t> { using type = uint32_t; };
template <typename T> using accumulator_type_t = typename accumulator_type<T>::type;

//=============================================================================
// Generate uniform key segments
//=============================================================================

template <typename KeyT>
device_vector<KeyT> gen_uniform_key_segments(seed_t seed, size_t total_elements,
                                             size_t min_segment_size,
                                             size_t max_segment_size) {
  device_vector<KeyT> keys(total_elements);
  size_t range = max_segment_size - min_segment_size + 1;

  // Use muRAND for random segment sizes
  MurandGenerator gen(seed.get());
  unsigned int* d_rand;
  MUSA_BENCH_CHECK(musaMalloc(&d_rand, total_elements * sizeof(unsigned int)));
  gen.generate(d_rand, total_elements);
  MUSA_BENCH_CHECK(musaDeviceSynchronize());

  std::vector<unsigned int> h_rand(total_elements);
  MUSA_BENCH_CHECK(musaMemcpy(h_rand.data(), d_rand, total_elements * sizeof(unsigned int),
                               musaMemcpyDeviceToHost));
  MUSA_BENCH_CHECK(musaFree(d_rand));

  std::vector<KeyT> h_keys(total_elements);
  size_t idx = 0;
  KeyT current_key = 0;

  while (idx < total_elements) {
    size_t segment_size = min_segment_size + (h_rand[idx] % range);
    segment_size = std::min(segment_size, total_elements - idx);
    for (size_t j = 0; j < segment_size && idx < total_elements; j++) {
      h_keys[idx++] = current_key;
    }
    current_key++;
  }

  MUSA_BENCH_CHECK(musaMemcpy(keys.data(), h_keys.data(),
                               total_elements * sizeof(KeyT),
                               musaMemcpyHostToDevice));
  return keys;
}

//=============================================================================
// Generate random bool values
//=============================================================================

inline void gen_bool(seed_t seed, device_vector<bool> &data, bit_entropy entropy) {
  size_t n = data.size();

  // Use muRAND
  MurandGenerator gen(seed.get());
  float* d_rand;
  MUSA_BENCH_CHECK(musaMalloc(&d_rand, n * sizeof(float)));
  gen.generate_uniform(d_rand, n);
  MUSA_BENCH_CHECK(musaDeviceSynchronize());

  std::vector<float> h_rand(n);
  MUSA_BENCH_CHECK(musaMemcpy(h_rand.data(), d_rand, n * sizeof(float),
                               musaMemcpyDeviceToHost));
  MUSA_BENCH_CHECK(musaFree(d_rand));

  double prob = entropy_to_probability(entropy);
  std::vector<uint8_t> h_data(n);
  for (size_t i = 0; i < n; i++) {
    h_data[i] = (h_rand[i] < static_cast<float>(prob)) ? 1 : 0;
  }

  MUSA_BENCH_CHECK(musaMemcpy(data.data(), h_data.data(), n, musaMemcpyHostToDevice));
}

//=============================================================================
// Helper functions
//=============================================================================

inline bit_entropy str_to_entropy(const std::string &str) {
  if (str == "1.000") return bit_entropy::_1_000;
  if (str == "0.811") return bit_entropy::_0_811;
  if (str == "0.544") return bit_entropy::_0_544;
  if (str == "0.337") return bit_entropy::_0_337;
  if (str == "0.201") return bit_entropy::_0_201;
  if (str == "0.000") return bit_entropy::_0_000;
  return bit_entropy::_1_000;
}

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