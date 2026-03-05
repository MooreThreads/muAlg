/******************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

// No-thrust version of test_device_three_way_partition.cu for MUSA platform testing

#define CUB_STDERR

#include <cub/device/device_partition.cuh>
#include "test_util.h"

#include <musa_runtime.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

using namespace cub;

//==============================================================================
// Functors
template <typename T>
struct LessThan
{
  T compare;
  explicit __host__ LessThan(T compare) : compare(compare) {}
  __device__ bool operator()(const T &a) const { return a < compare; }
};

template <typename T>
struct EqualTo
{
  T compare;
  explicit __host__ EqualTo(T compare) : compare(compare) {}
  __device__ bool operator()(const T &a) const { return a == compare; }
};

template <typename T>
struct GreaterOrEqual
{
  T compare;
  explicit __host__ GreaterOrEqual(T compare) : compare(compare) {}
  __device__ bool operator()(const T &a) const { return a >= compare; }
};

// Host versions of functors for reference computation
template <typename T>
bool host_less_than(T val, T compare) { return val < compare; }

template <typename T>
bool host_equal_to(T val, T compare) { return val == compare; }

template <typename T>
bool host_greater_or_equal(T val, T compare) { return val >= compare; }

//==============================================================================
// Reference three-way partition (host)
template <typename T>
void host_three_way_partition(const std::vector<T>& in,
                              std::vector<T>& first_part,
                              std::vector<T>& second_part,
                              std::vector<T>& unselected,
                              int& num_first,
                              int& num_second,
                              int& num_unselected,
                              T first_compare,
                              T second_compare,
                              bool use_less_for_first,
                              bool use_ge_for_second)
{
  first_part.clear();
  second_part.clear();
  unselected.clear();

  for (const auto& val : in)
  {
    bool in_first = use_less_for_first ? host_less_than(val, first_compare)
                                       : host_greater_or_equal(val, first_compare);
    bool in_second = use_ge_for_second ? host_greater_or_equal(val, second_compare)
                                       : host_less_than(val, second_compare);

    if (in_first)
    {
      first_part.push_back(val);
    }
    else if (in_second)
    {
      second_part.push_back(val);
    }
    else
    {
      unselected.push_back(val);
    }
  }

  num_first = static_cast<int>(first_part.size());
  num_second = static_cast<int>(second_part.size());
  num_unselected = static_cast<int>(unselected.size());
}

//==============================================================================
// CUB three-way partition (device)
template <typename T, typename FirstSelector, typename SecondSelector>
bool cub_three_way_partition(const std::vector<T>& h_in,
                             std::vector<T>& h_first_part,
                             std::vector<T>& h_second_part,
                             std::vector<T>& h_unselected,
                             int& num_first,
                             int& num_second,
                             int& num_unselected,
                             FirstSelector first_selector,
                             SecondSelector second_selector)
{
  const int num_items = static_cast<int>(h_in.size());

  // Handle empty input
  if (num_items == 0)
  {
    h_first_part.clear();
    h_second_part.clear();
    h_unselected.clear();
    num_first = 0;
    num_second = 0;
    num_unselected = 0;
    return true;
  }

  // Allocate device memory
  T* d_in = nullptr;
  T* d_first_out = nullptr;
  T* d_second_out = nullptr;
  T* d_unselected_out = nullptr;
  int* d_num_selected_out = nullptr;

  musaMalloc(&d_in, num_items * sizeof(T));
  musaMalloc(&d_first_out, num_items * sizeof(T));
  musaMalloc(&d_second_out, num_items * sizeof(T));
  musaMalloc(&d_unselected_out, num_items * sizeof(T));
  musaMalloc(&d_num_selected_out, 2 * sizeof(int));

  // Copy input to device
  musaMemcpy(d_in, h_in.data(), num_items * sizeof(T), musaMemcpyHostToDevice);

  // Get temp storage size
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  
  auto err = cub::DevicePartition::If(nullptr,
                                      temp_storage_bytes,
                                      d_in,
                                      d_first_out,
                                      d_second_out,
                                      d_unselected_out,
                                      d_num_selected_out,
                                      num_items,
                                      first_selector,
                                      second_selector,
                                      0,
                                      true);
  if (err != musaSuccess)
  {
    std::cerr << "DevicePartition::If size query failed: " << err << std::endl;
    musaFree(d_in);
    musaFree(d_first_out);
    musaFree(d_second_out);
    musaFree(d_unselected_out);
    musaFree(d_num_selected_out);
    return false;
  }

  // Allocate temp storage and run
  musaMalloc(&d_temp_storage, temp_storage_bytes);
  
  err = cub::DevicePartition::If(d_temp_storage,
                                 temp_storage_bytes,
                                 d_in,
                                 d_first_out,
                                 d_second_out,
                                 d_unselected_out,
                                 d_num_selected_out,
                                 num_items,
                                 first_selector,
                                 second_selector,
                                 0,
                                 true);
  if (err != musaSuccess)
  {
    std::cerr << "DevicePartition::If execution failed: " << err << std::endl;
    musaFree(d_temp_storage);
    musaFree(d_in);
    musaFree(d_first_out);
    musaFree(d_second_out);
    musaFree(d_unselected_out);
    musaFree(d_num_selected_out);
    return false;
  }

  musaDeviceSynchronize();

  // Copy results back
  int h_num_selected[2];
  musaMemcpy(h_num_selected, d_num_selected_out, 2 * sizeof(int), musaMemcpyDeviceToHost);

  num_first = h_num_selected[0];
  num_second = h_num_selected[1];
  num_unselected = num_items - num_first - num_second;

  // Validate counts
  if (num_first < 0 || num_second < 0 || num_unselected < 0 ||
      num_first > num_items || num_second > num_items || num_unselected > num_items)
  {
    std::cerr << "Invalid partition counts: " << num_first << ", " << num_second 
              << ", " << num_unselected << " (total: " << num_items << ")" << std::endl;
    musaFree(d_temp_storage);
    musaFree(d_in);
    musaFree(d_first_out);
    musaFree(d_second_out);
    musaFree(d_unselected_out);
    musaFree(d_num_selected_out);
    return false;
  }

  h_first_part.resize(num_first);
  h_second_part.resize(num_second);
  h_unselected.resize(num_unselected);

  musaMemcpy(h_first_part.data(), d_first_out, num_first * sizeof(T), musaMemcpyDeviceToHost);
  musaMemcpy(h_second_part.data(), d_second_out, num_second * sizeof(T), musaMemcpyDeviceToHost);
  musaMemcpy(h_unselected.data(), d_unselected_out, num_unselected * sizeof(T), musaMemcpyDeviceToHost);

  // Cleanup
  musaFree(d_temp_storage);
  musaFree(d_in);
  musaFree(d_first_out);
  musaFree(d_second_out);
  musaFree(d_unselected_out);
  musaFree(d_num_selected_out);

  return true;
}

//==============================================================================
// Compare results (order may differ, so just check counts and multiset equality)
template <typename T>
bool compare_partition_results(const std::vector<T>& ref_first,
                               const std::vector<T>& ref_second,
                               const std::vector<T>& ref_unselected,
                               const std::vector<T>& cub_first,
                               const std::vector<T>& cub_second,
                               const std::vector<T>& cub_unselected)
{
  // Check sizes
  if (ref_first.size() != cub_first.size()) return false;
  if (ref_second.size() != cub_second.size()) return false;
  if (ref_unselected.size() != cub_unselected.size()) return false;

  // Sort and compare (partition order may differ)
  std::vector<T> sorted_ref_first = ref_first;
  std::vector<T> sorted_cub_first = cub_first;
  std::sort(sorted_ref_first.begin(), sorted_ref_first.end());
  std::sort(sorted_cub_first.begin(), sorted_cub_first.end());
  if (sorted_ref_first != sorted_cub_first) return false;

  std::vector<T> sorted_ref_second = ref_second;
  std::vector<T> sorted_cub_second = cub_second;
  std::sort(sorted_ref_second.begin(), sorted_ref_second.end());
  std::sort(sorted_cub_second.begin(), sorted_cub_second.end());
  if (sorted_ref_second != sorted_cub_second) return false;

  std::vector<T> sorted_ref_unselected = ref_unselected;
  std::vector<T> sorted_cub_unselected = cub_unselected;
  std::sort(sorted_ref_unselected.begin(), sorted_ref_unselected.end());
  std::sort(sorted_cub_unselected.begin(), sorted_cub_unselected.end());
  if (sorted_ref_unselected != sorted_cub_unselected) return false;

  return true;
}

//==============================================================================
// Test case: Basic three-way partition
template <typename T>
bool TestBasic(int num_items, const std::string& test_name)
{
  std::cout << "  " << test_name << "(" << num_items << "): ";

  // Generate input data
  std::vector<T> h_in(num_items);
  for (int i = 0; i < num_items; ++i)
  {
    h_in[i] = static_cast<T>(i);
  }

  // Shuffle input
  std::mt19937 gen(42);
  std::shuffle(h_in.begin(), h_in.end(), gen);

  T first_compare = static_cast<T>(num_items / 3);
  T second_compare = static_cast<T>(2 * num_items / 3);

  // Compute reference
  std::vector<T> ref_first, ref_second, ref_unselected;
  int ref_num_first, ref_num_second, ref_num_unselected;
  host_three_way_partition(h_in, ref_first, ref_second, ref_unselected,
                           ref_num_first, ref_num_second, ref_num_unselected,
                           first_compare, second_compare, true, true);

  // Compute CUB result
  std::vector<T> cub_first, cub_second, cub_unselected;
  int cub_num_first, cub_num_second, cub_num_unselected;
  LessThan<T> first_selector(first_compare);
  GreaterOrEqual<T> second_selector(second_compare);

  bool success = cub_three_way_partition(h_in, cub_first, cub_second, cub_unselected,
                                         cub_num_first, cub_num_second, cub_num_unselected,
                                         first_selector, second_selector);

  if (!success)
  {
    std::cout << "FAIL (CUB error)" << std::endl;
    return false;
  }

  // Compare results
  bool pass = compare_partition_results(ref_first, ref_second, ref_unselected,
                                        cub_first, cub_second, cub_unselected);
  pass = pass && (ref_num_first == cub_num_first);
  pass = pass && (ref_num_second == cub_num_second);
  pass = pass && (ref_num_unselected == cub_num_unselected);

  std::cout << (pass ? "PASS" : "FAIL") << std::endl;
  return pass;
}

//==============================================================================
// Test case: Empty input
template <typename T>
bool TestEmpty()
{
  std::cout << "  TestEmpty: ";

  std::vector<T> h_in;
  LessThan<T> first_selector(T{0});
  GreaterOrEqual<T> second_selector(T{1});

  std::vector<T> cub_first, cub_second, cub_unselected;
  int cub_num_first, cub_num_second, cub_num_unselected;

  bool success = cub_three_way_partition(h_in, cub_first, cub_second, cub_unselected,
                                         cub_num_first, cub_num_second, cub_num_unselected,
                                         first_selector, second_selector);

  bool pass = success && cub_first.empty() && cub_second.empty() && cub_unselected.empty();
  std::cout << (pass ? "PASS" : "FAIL") << std::endl;
  return pass;
}

//==============================================================================
// Test case: Empty first part
template <typename T>
bool TestEmptyFirstPart(int num_items)
{
  std::cout << "  TestEmptyFirstPart(" << num_items << "): ";

  std::vector<T> h_in(num_items);
  for (int i = 0; i < num_items; ++i)
  {
    h_in[i] = static_cast<T>(i);
  }

  T first_compare = T{0};  // Nothing less than 0 for unsigned types
  T second_compare = static_cast<T>(num_items / 2);

  std::vector<T> ref_first, ref_second, ref_unselected;
  int ref_num_first, ref_num_second, ref_num_unselected;
  host_three_way_partition(h_in, ref_first, ref_second, ref_unselected,
                           ref_num_first, ref_num_second, ref_num_unselected,
                           first_compare, second_compare, true, true);

  std::vector<T> cub_first, cub_second, cub_unselected;
  int cub_num_first, cub_num_second, cub_num_unselected;
  LessThan<T> first_selector(first_compare);
  GreaterOrEqual<T> second_selector(second_compare);

  bool success = cub_three_way_partition(h_in, cub_first, cub_second, cub_unselected,
                                         cub_num_first, cub_num_second, cub_num_unselected,
                                         first_selector, second_selector);

  bool pass = success && (cub_num_first == 0);
  std::cout << (pass ? "PASS" : "FAIL") << std::endl;
  return pass;
}

//==============================================================================
// Test case: All in unselected
template <typename T>
bool TestUnselectedOnly(int num_items)
{
  std::cout << "  TestUnselectedOnly(" << num_items << "): ";

  std::vector<T> h_in(num_items);
  for (int i = 0; i < num_items; ++i)
  {
    h_in[i] = static_cast<T>(i);
  }

  // Both selectors match nothing for unsigned types with compare value 0
  T compare = T{0};
  LessThan<T> selector(compare);

  std::vector<T> cub_first, cub_second, cub_unselected;
  int cub_num_first, cub_num_second, cub_num_unselected;

  bool success = cub_three_way_partition(h_in, cub_first, cub_second, cub_unselected,
                                         cub_num_first, cub_num_second, cub_num_unselected,
                                         selector, selector);

  bool pass = success && (cub_num_first == 0) && (cub_num_second == 0) && 
              (cub_num_unselected == num_items);
  std::cout << (pass ? "PASS" : "FAIL") << std::endl;
  return pass;
}

//==============================================================================
// Test type
template <typename T>
bool TestType()
{
  std::cout << "Testing " << typeid(T).name() << ":" << std::endl;
  bool all_pass = true;

  if (!TestEmpty<T>()) all_pass = false;
  if (!TestBasic<T>(100, "TestBasic")) all_pass = false;
  if (!TestBasic<T>(1000, "TestBasic")) all_pass = false;
  if (!TestBasic<T>(10000, "TestBasic")) all_pass = false;
  if (!TestEmptyFirstPart<T>(100)) all_pass = false;
  if (!TestUnselectedOnly<T>(100)) all_pass = false;

  return all_pass;
}

//==============================================================================
int main(int argc, char **argv)
{
  CommandLineArgs args(argc, argv);
  CubDebugExit(args.DeviceInit());

  std::cout << "=== Device Three-Way Partition Test (No Thrust) ===" << std::endl;

  bool all_pass = true;
  if (!TestType<std::uint8_t>()) all_pass = false;
  if (!TestType<std::uint16_t>()) all_pass = false;
  if (!TestType<std::uint32_t>()) all_pass = false;
  if (!TestType<std::uint64_t>()) all_pass = false;

  std::cout << "\n=== All tests " << (all_pass ? "PASS" : "FAIL") << " ===" << std::endl;
  return all_pass ? 0 : 1;
}
