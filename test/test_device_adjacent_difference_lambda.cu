/****************************************************************************
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
 ****************************************************************************/

#define CUB_STDERR

#include <cub/device/device_adjacent_difference.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>

#include "test_util.h"

struct DetectWrongDifference
{
  using difference_type = void;
  using value_type = void;
  using pointer = void;
  using reference = void;
  using iterator_category = std::output_iterator_tag;

  bool *flag;

  __host__ __device__ DetectWrongDifference operator++() const
  {
    return *this;
  }

  __host__ __device__ DetectWrongDifference operator*() const
  {
    return *this;
  }

  template <typename Difference>
  __host__ __device__ DetectWrongDifference operator+(Difference) const
  {
    return *this;
  }

  template <typename Index>
  __host__ __device__ DetectWrongDifference operator[](Index) const
  {
    return *this;
  }

  __device__ void operator=(long long difference) const
  {
    if (difference != 1)
    {
      *flag = false;
    }
  }
};

template <typename InputIteratorT,
          typename OutputIteratorT,
          typename DifferenceOpT,
          typename NumItemsT>
void RunSubtractLeftCopy(InputIteratorT input,
                         OutputIteratorT output,
                         DifferenceOpT difference_op,
                         NumItemsT num_items)
{
  std::size_t temp_storage_bytes = 0;
  CubDebugExit(cub::DeviceAdjacentDifference::SubtractLeftCopy(nullptr,
                                                               temp_storage_bytes,
                                                               input,
                                                               output,
                                                               num_items,
                                                               difference_op));

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  CubDebugExit(cub::DeviceAdjacentDifference::SubtractLeftCopy(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_storage_bytes,
    input,
    output,
    num_items,
    difference_op));
}

void TestDeviceLambdaWithoutResultType()
{
  thrust::device_vector<long> input(5);
  input[0] = 1;
  input[1] = 1;
  input[2] = 2;
  input[3] = 2;
  input[4] = 5;

  thrust::device_vector<int> output(input.size(), -1);

  auto difference_op = [] __device__(long lhs, long rhs) {
    return lhs != rhs;
  };

  RunSubtractLeftCopy(
    input.begin(), output.begin(), difference_op, static_cast<int>(input.size()));

  thrust::host_vector<int> expected(5);
  expected[0] = 1;
  expected[1] = 0;
  expected[2] = 1;
  expected[3] = 0;
  expected[4] = 1;

  AssertEquals(output.size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i)
  {
    AssertEquals(output[i], expected[i]);
  }
}

void TestCapturedDeviceLambdaWithoutResultType()
{
  thrust::device_vector<int> rows_flat(6);
  rows_flat[0] = 1;
  rows_flat[1] = 2;
  rows_flat[2] = 1;
  rows_flat[3] = 2;
  rows_flat[4] = 3;
  rows_flat[5] = 4;

  thrust::device_vector<int> output(3, -1);
  auto row_ids = thrust::make_counting_iterator<long>(0);

  const int *rows_ptr = thrust::raw_pointer_cast(rows_flat.data());
  const int row_width = 2;

  auto difference_op = [=] __device__(long lhs_row, long rhs_row) {
    for (int column = 0; column < row_width; ++column)
    {
      const int lhs = rows_ptr[lhs_row * row_width + column];
      const int rhs = rows_ptr[rhs_row * row_width + column];
      if (lhs != rhs)
      {
        return 1;
      }
    }

    return 0;
  };

  RunSubtractLeftCopy(row_ids, output.begin(), difference_op, 3);

  thrust::host_vector<int> expected(3);
  expected[0] = 0;
  expected[1] = 0;
  expected[2] = 1;

  AssertEquals(output.size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i)
  {
    AssertEquals(output[i], expected[i]);
  }
}

void TestWriteOnlyOutputIteratorFallback()
{
  thrust::device_vector<bool> all_differences_correct(1, true);
  thrust::counting_iterator<long long> input(1);
  DetectWrongDifference output{
    thrust::raw_pointer_cast(all_differences_correct.data())};

  RunSubtractLeftCopy(input, output, cub::Difference{}, 128);
  AssertEquals(all_differences_correct.front(), true);
}

int main(int argc, char **argv)
{
  CommandLineArgs args(argc, argv);
  CubDebugExit(args.DeviceInit());

  TestDeviceLambdaWithoutResultType();
  TestCapturedDeviceLambdaWithoutResultType();
  TestWriteOnlyOutputIteratorFallback();

  return 0;
}
