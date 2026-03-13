/******************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2024, Moore Threads Corporation.  All rights reserved.
 *
 * MUSA benchmark framework - a lightweight replacement for nvbench.
 ******************************************************************************/

#pragma once

#include <musa_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string>
#include <vector>
#include <type_traits>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cmath>

namespace musa_bench {

//=============================================================================
// Error checking
//=============================================================================

#define MUSA_BENCH_CHECK(call)                                                  \
  do {                                                                          \
    musaError_t err = call;                                                     \
    if (err != musaSuccess) {                                                   \
      fprintf(stderr, "MUSA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              musaGetErrorString(err));                                         \
      exit(EXIT_FAILURE);                                                       \
    }                                                                           \
  } while (0)

//=============================================================================
// Timer class - uses MUSA events for accurate GPU timing
//=============================================================================

class Timer {
  musaEvent_t start_;
  musaEvent_t stop_;
  musaStream_t stream_;

public:
  Timer(musaStream_t stream = 0) : stream_(stream) {
    MUSA_BENCH_CHECK(musaEventCreate(&start_));
    MUSA_BENCH_CHECK(musaEventCreate(&stop_));
  }

  ~Timer() {
    MUSA_BENCH_CHECK(musaEventDestroy(start_));
    MUSA_BENCH_CHECK(musaEventDestroy(stop_));
  }

  void start() { MUSA_BENCH_CHECK(musaEventRecord(start_, stream_)); }

  void stop() { MUSA_BENCH_CHECK(musaEventRecord(stop_, stream_)); }

  float elapsed_ms() const {
    float ms;
    MUSA_BENCH_CHECK(musaEventSynchronize(stop_));
    MUSA_BENCH_CHECK(musaEventElapsedTime(&ms, start_, stop_));
    return ms;
  }
};

//=============================================================================
// Benchmark State
//=============================================================================

struct State {
  int64_t elements = 0;
  int warmup_iterations = 10;
  int test_iterations = 100;

  // Performance metrics
  double total_time_ms = 0.0;
  double min_time_ms = 1e30;
  double max_time_ms = 0.0;
  int64_t bytes_read = 0;
  int64_t bytes_written = 0;

  musaStream_t stream = 0;

  State() = default;

  void add_element_count(int64_t n) { elements = n; }

  template <typename T> void add_global_memory_reads(int64_t n) {
    bytes_read += n * sizeof(T);
  }

  template <typename T> void add_global_memory_writes(int64_t n) {
    bytes_written += n * sizeof(T);
  }

  double avg_time_ms() const {
    return total_time_ms / test_iterations;
  }

  double throughput_gb_s() const {
    if (avg_time_ms() == 0.0)
      return 0.0;
    double total_bytes = bytes_read + bytes_written;
    return (total_bytes / 1e9) / (avg_time_ms() / 1e3);
  }

  void print_results() const {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Elements: " << elements << "\n";
    std::cout << "Iterations: " << test_iterations << "\n";
    std::cout << "Avg Time: " << avg_time_ms() << " ms\n";
    std::cout << "Min Time: " << min_time_ms << " ms\n";
    std::cout << "Max Time: " << max_time_ms << " ms\n";
    std::cout << "Throughput: " << throughput_gb_s() << " GB/s\n";
  }
};

//=============================================================================
// Type list for parameterized benchmarks
//=============================================================================

template <typename... Ts> struct type_list {};

// Helper to get type at index
template <size_t I, typename T> struct type_at;

template <size_t I, typename Head, typename... Tail>
struct type_at<I, type_list<Head, Tail...>>
    : type_at<I - 1, type_list<Tail...>> {};

template <typename Head, typename... Tail>
struct type_at<0, type_list<Head, Tail...>> {
  using type = Head;
};

template <size_t I, typename T>
using type_at_t = typename type_at<I, T>::type;

// Helper to get type list size
template <typename T> struct type_list_size;

template <typename... Ts>
struct type_list_size<type_list<Ts...>> {
  static constexpr size_t value = sizeof...(Ts);
};

//=============================================================================
// Range generator (similar to nvbench::range)
//=============================================================================

inline std::vector<int64_t> range(int64_t start_pow2, int64_t end_pow2,
                                  int64_t step = 1) {
  std::vector<int64_t> result;
  for (int64_t p = start_pow2; p <= end_pow2; p += step) {
    result.push_back(int64_t(1) << p);
  }
  return result;
}

//=============================================================================
// Fundamental types for benchmarks
//=============================================================================

using fundamental_types =
    type_list<int8_t, int16_t, int32_t, int64_t, float, double>;

using offset_types = type_list<int32_t, int64_t>;

//=============================================================================
// Benchmark runner base class
//=============================================================================

template <typename Func> class Benchmark {
  Func func_;
  std::string name_;

public:
  Benchmark(std::string name, Func func) : name_(name), func_(func) {}

  void run() {
    std::cout << "=== Benchmark: " << name_ << " ===" << std::endl;
    func_();
    std::cout << std::endl;
  }
};

//=============================================================================
// Benchmark registration and execution
//=============================================================================

inline std::vector<std::pair<std::string, void (*)()>> &get_benchmarks() {
  static std::vector<std::pair<std::string, void (*)()>> benchmarks;
  return benchmarks;
}

inline void register_benchmark(const std::string &name, void (*func)()) {
  get_benchmarks().push_back({name, func});
}

inline int run_all_benchmarks() {
  auto &benchmarks = get_benchmarks();
  if (benchmarks.empty()) {
    std::cout << "No benchmarks registered." << std::endl;
    return 0;
  }

  for (auto &bench : benchmarks) {
    std::cout << "\n=== " << bench.first << " ===" << std::endl;
    bench.second();
  }

  return 0;
}

//=============================================================================
// Macros for benchmark registration
//=============================================================================

// Simple benchmark macro for single function
#define MUSA_BENCH(name, func)                                                 \
  int main(int argc, char **argv) {                                            \
    (void)argc;                                                                \
    (void)argv;                                                                \
    std::cout << "=== Benchmark: " << name << " ===" << std::endl;            \
    func();                                                                    \
    return 0;                                                                  \
  }

// Macro for benchmark with state parameter
#define MUSA_BENCH_STATE(name, bench_func)                                     \
  static void bench_func##_impl();                                             \
  int main(int argc, char **argv) {                                            \
    (void)argc;                                                                \
    (void)argv;                                                                \
    std::cout << "=== Benchmark: " << name << " ===" << std::endl;            \
    bench_func##_impl();                                                       \
    return 0;                                                                  \
  }                                                                            \
  static void bench_func##_impl()

//=============================================================================
// Device memory wrapper (simplified thrust::device_vector alternative)
//=============================================================================

template <typename T> class device_vector {
  T *ptr_ = nullptr;
  size_t size_ = 0;

public:
  device_vector() = default;

  explicit device_vector(size_t n) : size_(n) {
    MUSA_BENCH_CHECK(musaMalloc(&ptr_, n * sizeof(T)));
  }

  device_vector(size_t n, const T &value) : size_(n) {
    MUSA_BENCH_CHECK(musaMalloc(&ptr_, n * sizeof(T)));
    std::vector<T> host(n, value);
    MUSA_BENCH_CHECK(
        musaMemcpy(ptr_, host.data(), n * sizeof(T), musaMemcpyHostToDevice));
  }

  ~device_vector() {
    if (ptr_) {
      musaFree(ptr_);
    }
  }

  // Move constructor
  device_vector(device_vector &&other) noexcept
      : ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
  }

  // Move assignment
  device_vector &operator=(device_vector &&other) noexcept {
    if (this != &other) {
      if (ptr_)
        musaFree(ptr_);
      ptr_ = other.ptr_;
      size_ = other.size_;
      other.ptr_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  // Disable copy
  device_vector(const device_vector &) = delete;
  device_vector &operator=(const device_vector &) = delete;

  T *data() { return ptr_; }
  const T *data() const { return ptr_; }
  size_t size() const { return size_; }

  void resize(size_t n) {
    if (ptr_)
      musaFree(ptr_);
    size_ = n;
    MUSA_BENCH_CHECK(musaMalloc(&ptr_, n * sizeof(T)));
  }
};

//=============================================================================
// Seed wrapper for reproducible random generation
//=============================================================================

class seed_t {
  unsigned long long value_;

public:
  explicit seed_t(unsigned long long val = 42) : value_(val) {}
  unsigned long long get() const { return value_; }
  seed_t &operator++() {
    ++value_;
    return *this;
  }
};

//=============================================================================
// Bit entropy enum for data generation
//=============================================================================

enum class bit_entropy {
  _1_000 = 0,
  _0_811 = 1,
  _0_544 = 2,
  _0_337 = 3,
  _0_201 = 4,
  _0_000 = 4200
};

inline double entropy_to_probability(bit_entropy entropy) {
  switch (entropy) {
  case bit_entropy::_0_000:
    return 0.0;
  case bit_entropy::_0_811:
    return 0.811;
  case bit_entropy::_0_544:
    return 0.544;
  case bit_entropy::_0_337:
    return 0.337;
  case bit_entropy::_0_201:
    return 0.201;
  case bit_entropy::_1_000:
    return 1.0;
  default:
    return 0.0;
  }
}

//=============================================================================
// Type name utilities
//=============================================================================

template <typename T> const char *type_name() { return "unknown"; }

template <> inline const char *type_name<int8_t>() { return "int8"; }
template <> inline const char *type_name<int16_t>() { return "int16"; }
template <> inline const char *type_name<int32_t>() { return "int32"; }
template <> inline const char *type_name<int64_t>() { return "int64"; }
template <> inline const char *type_name<uint8_t>() { return "uint8"; }
template <> inline const char *type_name<uint16_t>() { return "uint16"; }
template <> inline const char *type_name<uint32_t>() { return "uint32"; }
template <> inline const char *type_name<uint64_t>() { return "uint64"; }
template <> inline const char *type_name<float>() { return "float"; }
template <> inline const char *type_name<double>() { return "double"; }

//=============================================================================
// Launch wrapper for stream management
//=============================================================================

struct launch {
  musaStream_t stream;

  explicit launch(musaStream_t s = 0) : stream(s) {}
  musaStream_t get_stream() const { return stream; }
};

} // namespace musa_bench

//=============================================================================
// Convenience macros for parameterized benchmarks
//=============================================================================

// Generate benchmark for a type axis
#define MUSA_BENCH_TYPE_AXIS(name, func, type_list)                            \
  template <typename T> void func##_impl(musa_bench::State &state);            \
  static void run_##func() {                                                    \
    musa_bench::State state;                                                    \
    func##_impl<musa_bench::type_at_t<0, type_list>>(state);                    \
  }                                                                             \
  int main(int argc, char **argv) {                                            \
    (void)argc;                                                                 \
    (void)argv;                                                                 \
    std::cout << "=== Benchmark: " << name << " ===" << std::endl;             \
    run_##func();                                                               \
    return 0;                                                                   \
  }