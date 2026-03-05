/******************************************************************************
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

// No-thrust version of test_device_spmv.cu for MUSA platform testing

#define CUB_STDERR

#include <cub/device/device_spmv.cuh>
#include <cub/util_debug.cuh>

#include <musa_runtime.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <type_traits>
#include <typeinfo>

#include "test_util.h"

bool g_verbose = false;

//==============================================================================
// Casts char types to int for numeric printing
template <typename T>
T print_cast(T val) { return val; }

int print_cast(char val) { return static_cast<int>(val); }
int print_cast(signed char val) { return static_cast<int>(val); }
int print_cast(unsigned char val) { return static_cast<int>(val); }

//==============================================================================
// Simple CSR matrix host implementation
template <typename ValueT>
struct HostCSRMatrix
{
  std::vector<ValueT> values;
  std::vector<int> row_offsets;
  std::vector<int> column_indices;
  int num_rows;
  int num_cols;
  int num_nonzeros;

  HostCSRMatrix(int rows, int cols)
    : row_offsets(rows + 1, 0)
    , num_rows(rows)
    , num_cols(cols)
    , num_nonzeros(0)
  {}

  void append_value(int row, int col, ValueT value)
  {
    ++num_nonzeros;
    ++row_offsets[row];
    column_indices.push_back(col);
    values.push_back(value);
  }

  void finalize()
  {
    // Convert counts to offsets via exclusive scan
    std::vector<int> counts = row_offsets;
    row_offsets[0] = 0;
    for (int i = 0; i < num_rows; ++i)
    {
      row_offsets[i + 1] = row_offsets[i] + counts[i];
    }
  }

  int get_row_offset(int row) const { return row_offsets[row]; }

  int get_row_num_nonzero(int row) const
  {
    return row_offsets[row + 1] - row_offsets[row];
  }
};

//==============================================================================
// Compare floating point with tolerance
template <typename ValueT>
bool almost_equal(ValueT v1, ValueT v2, typename std::enable_if<std::is_floating_point<ValueT>::value>::type* = nullptr)
{
  constexpr double r_tol = 1e-3;
  constexpr double a_tol = 1e-2;
  const double limit = r_tol * (std::fabs(v1) + std::fabs(v2)) + a_tol;
  return std::fabs(v1 - v2) <= limit;
}

template <typename ValueT>
bool almost_equal(ValueT v1, ValueT v2, typename std::enable_if<!std::is_floating_point<ValueT>::value>::type* = nullptr)
{
  return v1 == v2;
}

//==============================================================================
// Generate random CSR matrix
template <typename ValueT>
HostCSRMatrix<ValueT> make_random_csr_matrix(int num_rows, int num_cols, float target_fill_ratio)
{
  HostCSRMatrix<ValueT> mat(num_rows, num_cols);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  for (int row = 0; row < num_rows; ++row)
  {
    for (int col = 0; col < num_cols; ++col)
    {
      if (dist(gen) < target_fill_ratio)
      {
        ValueT value;
        if constexpr (std::is_floating_point<ValueT>::value)
        {
          std::uniform_real_distribution<ValueT> val_dist(static_cast<ValueT>(-100), static_cast<ValueT>(100));
          value = val_dist(gen);
        }
        else
        {
          std::uniform_int_distribution<int64_t> val_dist(std::numeric_limits<ValueT>::min(), std::numeric_limits<ValueT>::max());
          value = static_cast<ValueT>(val_dist(gen));
        }
        mat.append_value(row, col, value);
      }
    }
  }

  mat.finalize();
  return mat;
}

//==============================================================================
// Generate random vector
template <typename ValueT>
std::vector<ValueT> make_random_vector(int len)
{
  std::vector<ValueT> vec(len);
  std::mt19937 gen(42);
  
  for (int i = 0; i < len; ++i)
  {
    if constexpr (std::is_floating_point<ValueT>::value)
    {
      std::uniform_real_distribution<ValueT> dist(static_cast<ValueT>(-100), static_cast<ValueT>(100));
      vec[i] = dist(gen);
    }
    else
    {
      std::uniform_int_distribution<int64_t> dist(std::numeric_limits<ValueT>::min(), std::numeric_limits<ValueT>::max());
      vec[i] = static_cast<ValueT>(dist(gen));
    }
  }
  return vec;
}

//==============================================================================
// Serial y = Ax computation (reference)
template <typename ValueT>
void compute_reference(const HostCSRMatrix<ValueT>& a,
                       const std::vector<ValueT>& x,
                       std::vector<ValueT>& y)
{
  if (a.num_rows == 0 || a.num_cols == 0) return;

  for (int row = 0; row < a.num_rows; ++row)
  {
    const int row_offset = a.get_row_offset(row);
    const int row_length = a.get_row_num_nonzero(row);

    ValueT accum{};
    for (int i = 0; i < row_length; ++i)
    {
      int col = a.column_indices[row_offset + i];
      ValueT val = a.values[row_offset + i];
      accum += val * x[col];
    }
    y[row] = accum;
  }
}

//==============================================================================
// Compare results
template <typename ValueT>
bool compare_results(const std::vector<ValueT>& ref, const std::vector<ValueT>& result)
{
  if (ref.size() != result.size()) return false;

  for (size_t i = 0; i < ref.size(); ++i)
  {
    if (!almost_equal(ref[i], result[i]))
    {
      std::cerr << "Mismatch at position " << i << ": "
                << print_cast(ref[i]) << " vs "
                << print_cast(result[i]) << std::endl;
      return false;
    }
  }
  return true;
}

//==============================================================================
// Test SpMV with given parameters
template <typename ValueT>
bool test_spmv(const HostCSRMatrix<ValueT>& h_a, const std::vector<ValueT>& h_x)
{
  const int num_rows = h_a.num_rows;
  const int num_cols = h_a.num_cols;
  const int num_nonzeros = h_a.num_nonzeros;

  if (g_verbose)
  {
    std::cout << "Testing cub::DeviceSpmv::CsrMV on " << num_rows << "x" << num_cols
              << " matrix, " << num_nonzeros << " nonzeros" << std::endl;
  }

  // Compute reference solution
  std::vector<ValueT> h_y_ref(num_rows);
  compute_reference(h_a, h_x, h_y_ref);

  // Allocate device memory
  ValueT* d_values = nullptr;
  int* d_row_offsets = nullptr;
  int* d_column_indices = nullptr;
  ValueT* d_x = nullptr;
  ValueT* d_y = nullptr;

  musaMalloc(&d_values, num_nonzeros * sizeof(ValueT));
  musaMalloc(&d_row_offsets, (num_rows + 1) * sizeof(int));
  musaMalloc(&d_column_indices, num_nonzeros * sizeof(int));
  musaMalloc(&d_x, num_cols * sizeof(ValueT));
  musaMalloc(&d_y, num_rows * sizeof(ValueT));

  // Copy input data to device
  musaMemcpy(d_values, h_a.values.data(), num_nonzeros * sizeof(ValueT), musaMemcpyHostToDevice);
  musaMemcpy(d_row_offsets, h_a.row_offsets.data(), (num_rows + 1) * sizeof(int), musaMemcpyHostToDevice);
  musaMemcpy(d_column_indices, h_a.column_indices.data(), num_nonzeros * sizeof(int), musaMemcpyHostToDevice);
  musaMemcpy(d_x, h_x.data(), num_cols * sizeof(ValueT), musaMemcpyHostToDevice);

  // Compute temp storage size
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  auto err = cub::DeviceSpmv::CsrMV(d_temp_storage,
                                    temp_storage_bytes,
                                    d_values,
                                    d_row_offsets,
                                    d_column_indices,
                                    d_x,
                                    d_y,
                                    num_rows,
                                    num_cols,
                                    num_nonzeros);
  if (err != musaSuccess)
  {
    std::cerr << "CsrMV size query failed: " << err << std::endl;
    musaFree(d_values);
    musaFree(d_row_offsets);
    musaFree(d_column_indices);
    musaFree(d_x);
    musaFree(d_y);
    return false;
  }

  // Allocate temp storage and run
  musaMalloc(&d_temp_storage, temp_storage_bytes);

  err = cub::DeviceSpmv::CsrMV(d_temp_storage,
                               temp_storage_bytes,
                               d_values,
                               d_row_offsets,
                               d_column_indices,
                               d_x,
                               d_y,
                               num_rows,
                               num_cols,
                               num_nonzeros,
                               0,
                               true);
  if (err != musaSuccess)
  {
    std::cerr << "CsrMV execution failed: " << err << std::endl;
    musaFree(d_temp_storage);
    musaFree(d_values);
    musaFree(d_row_offsets);
    musaFree(d_column_indices);
    musaFree(d_x);
    musaFree(d_y);
    return false;
  }

  musaDeviceSynchronize();

  // Copy result back
  std::vector<ValueT> h_y(num_rows);
  musaMemcpy(h_y.data(), d_y, num_rows * sizeof(ValueT), musaMemcpyDeviceToHost);

  // Cleanup
  musaFree(d_temp_storage);
  musaFree(d_values);
  musaFree(d_row_offsets);
  musaFree(d_column_indices);
  musaFree(d_x);
  musaFree(d_y);

  // Compare results
  return compare_results(h_y_ref, h_y);
}

//==============================================================================
// Test doc example
template <typename ValueT>
bool test_doc_example()
{
  std::cout << "  test_doc_example<" << typeid(ValueT).name() << ">()" << std::endl;

  HostCSRMatrix<ValueT> h_a(9, 9);
  h_a.append_value(0, 1, ValueT{1});
  h_a.append_value(0, 3, ValueT{1});
  h_a.append_value(1, 0, ValueT{1});
  h_a.append_value(1, 2, ValueT{1});
  h_a.append_value(1, 4, ValueT{1});
  h_a.append_value(2, 1, ValueT{1});
  h_a.append_value(2, 5, ValueT{1});
  h_a.append_value(3, 0, ValueT{1});
  h_a.append_value(3, 4, ValueT{1});
  h_a.append_value(3, 6, ValueT{1});
  h_a.append_value(4, 1, ValueT{1});
  h_a.append_value(4, 3, ValueT{1});
  h_a.append_value(4, 5, ValueT{1});
  h_a.append_value(4, 7, ValueT{1});
  h_a.append_value(5, 2, ValueT{1});
  h_a.append_value(5, 4, ValueT{1});
  h_a.append_value(5, 8, ValueT{1});
  h_a.append_value(6, 3, ValueT{1});
  h_a.append_value(6, 7, ValueT{1});
  h_a.append_value(7, 4, ValueT{1});
  h_a.append_value(7, 6, ValueT{1});
  h_a.append_value(7, 8, ValueT{1});
  h_a.append_value(8, 5, ValueT{1});
  h_a.append_value(8, 7, ValueT{1});
  h_a.finalize();

  std::vector<ValueT> h_x(9, ValueT{1});

  return test_spmv(h_a, h_x);
}

//==============================================================================
// Test with random matrix
template <typename ValueT>
bool test_random(int rows, int cols, float fill_ratio)
{
  std::cout << "  test_random<" << typeid(ValueT).name() << ">(" 
            << rows << ", " << cols << ", " << fill_ratio << "): ";

  HostCSRMatrix<ValueT> h_a = make_random_csr_matrix<ValueT>(rows, cols, fill_ratio);
  std::vector<ValueT> h_x = make_random_vector<ValueT>(cols);

  bool pass = test_spmv(h_a, h_x);
  std::cout << (pass ? "PASS" : "FAIL") << std::endl;
  return pass;
}

//==============================================================================
// Test type
template <typename ValueT>
bool test_type()
{
  std::cout << "\nTesting " << typeid(ValueT).name() << ":" << std::endl;
  bool all_pass = true;

  // Doc example
  if (!test_doc_example<ValueT>()) all_pass = false;

  // Edge cases
  if (!test_random<ValueT>(0, 0, 1.f)) all_pass = false;
  if (!test_random<ValueT>(0, 1, 1.f)) all_pass = false;
  if (!test_random<ValueT>(1, 0, 1.f)) all_pass = false;
  if (!test_random<ValueT>(1, 1, 1.f)) all_pass = false;

  // Various sizes
  if (!test_random<ValueT>(16, 16, 0.5f)) all_pass = false;
  if (!test_random<ValueT>(100, 100, 0.1f)) all_pass = false;
  if (!test_random<ValueT>(256, 256, 0.3f)) all_pass = false;

  return all_pass;
}

//==============================================================================
int main(int argc, char** argv)
{
  CommandLineArgs args(argc, argv);
  g_verbose = args.CheckCmdLineFlag("v");

  if (args.CheckCmdLineFlag("help"))
  {
    printf("%s [--device=<device-id>] [--v]\n", argv[0]);
    exit(0);
  }

  CubDebugExit(args.DeviceInit());

  std::cout << "=== Device SpMV Test (No Thrust) ===" << std::endl;

  bool all_pass = true;
  if (!test_type<float>()) all_pass = false;
  if (!test_type<double>()) all_pass = false;
  if (!test_type<int>()) all_pass = false;

  std::cout << "\n=== All tests " << (all_pass ? "PASS" : "FAIL") << " ===" << std::endl;
  return all_pass ? 0 : 1;
}
