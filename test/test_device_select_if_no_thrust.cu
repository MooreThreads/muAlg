/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
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

// No-Thrust version of test_device_select_if.cu for MUSA platform compatibility

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <cub/cub.cuh>
#include <cub/util_allocator.cuh>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>

#include <musa_runtime.h>

#include "test_util.h"

using namespace cub;


//---------------------------------------------------------------------
// Selection functors
//---------------------------------------------------------------------

// Select items less than a threshold
template <typename T>
struct LessThan
{
    T compare;

    __host__ __device__ __forceinline__
    LessThan(T compare) : compare(compare) {}

    __host__ __device__ __forceinline__
    bool operator()(const T &a) const {
        return (a < compare);
    }
};

// Select even items
template <typename T>
struct IsEven
{
    __host__ __device__ __forceinline__
    bool operator()(const T &a) const {
        return ((a & 1) == 0);
    }
};


//---------------------------------------------------------------------
// Test functions
//---------------------------------------------------------------------

template <typename T, typename SelectOp>
void TestSelectIf(int num_items, SelectOp select_op, const char* description, std::mt19937& rng)
{
    printf("  TestSelectIf<T=%s> %d items, %s: ", typeid(T).name(), num_items, description);

    if (num_items == 0) {
        // Test empty input
        void* d_temp_storage = nullptr;
        std::size_t temp_storage_bytes = 0;
        T* d_in = nullptr;
        T* d_out = nullptr;
        int* d_num_selected = nullptr;

        CubDebugExit(cub::DeviceSelect::If(
            d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected, num_items, select_op));

        printf("PASS (empty)\n");
        return;
    }

    // Initialize host data
    std::vector<T> h_in(num_items);
    std::vector<T> h_reference(num_items);
    std::vector<T> h_out(num_items);

    // Fill with random values
    std::uniform_int_distribution<int> dist(0, 127);
    for (int i = 0; i < num_items; ++i) {
        h_in[i] = static_cast<T>(dist(rng));
    }

    // Compute reference on host
    int num_selected = 0;
    for (int i = 0; i < num_items; ++i) {
        if (select_op(h_in[i])) {
            h_reference[num_selected] = h_in[i];
            num_selected++;
        }
    }

    // Allocate device memory
    T* d_in = nullptr;
    T* d_out = nullptr;
    int* d_num_selected = nullptr;
    musaMalloc(&d_in, num_items * sizeof(T));
    musaMalloc(&d_out, num_items * sizeof(T));
    musaMalloc(&d_num_selected, sizeof(int));

    musaMemcpy(d_in, h_in.data(), num_items * sizeof(T), musaMemcpyHostToDevice);
    musaMemset(d_out, 0, num_items * sizeof(T));
    musaMemset(d_num_selected, 0, sizeof(int));

    // Run DeviceSelect::If
    void* d_temp_storage = nullptr;
    std::size_t temp_storage_bytes = 0;
    CubDebugExit(cub::DeviceSelect::If(
        d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected, num_items, select_op));

    musaMalloc(&d_temp_storage, temp_storage_bytes);
    CubDebugExit(cub::DeviceSelect::If(
        d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected, num_items, select_op));

    // Copy results back
    int h_num_selected = 0;
    musaMemcpy(h_out.data(), d_out, num_items * sizeof(T), musaMemcpyDeviceToHost);
    musaMemcpy(&h_num_selected, d_num_selected, sizeof(int), musaMemcpyDeviceToHost);

    // Check results
    bool data_ok = true;
    for (int i = 0; i < num_selected; ++i) {
        if (h_out[i] != h_reference[i]) {
            printf("Mismatch at index %d: got %lld, expected %lld\n",
                   i, (long long)h_out[i], (long long)h_reference[i]);
            data_ok = false;
            break;
        }
    }
    bool count_ok = (h_num_selected == num_selected);

    if (!count_ok) {
        printf("Count mismatch: got %d, expected %d. ", h_num_selected, num_selected);
    }

    printf("%s\n", (data_ok && count_ok) ? "PASS" : "FAIL");

    // Cleanup
    musaFree(d_temp_storage);
    musaFree(d_in);
    musaFree(d_out);
    musaFree(d_num_selected);
}


template <typename T, typename SelectOp>
void TestPartitionIf(int num_items, SelectOp select_op, const char* description, std::mt19937& rng)
{
    printf("  TestPartitionIf<T=%s> %d items, %s: ", typeid(T).name(), num_items, description);

    if (num_items == 0) {
        void* d_temp_storage = nullptr;
        std::size_t temp_storage_bytes = 0;
        T* d_in = nullptr;
        T* d_out = nullptr;
        int* d_num_selected = nullptr;

        CubDebugExit(cub::DevicePartition::If(
            d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected, num_items, select_op));

        printf("PASS (empty)\n");
        return;
    }

    // Initialize host data
    std::vector<T> h_in(num_items);
    std::vector<T> h_reference(num_items);
    std::vector<T> h_out(num_items);

    // Fill with random values
    std::uniform_int_distribution<int> dist(0, 127);
    for (int i = 0; i < num_items; ++i) {
        h_in[i] = static_cast<T>(dist(rng));
    }

    // Compute reference on host (selected first, then unselected)
    int num_selected = 0;
    int num_unselected = 0;
    for (int i = 0; i < num_items; ++i) {
        if (select_op(h_in[i])) {
            num_selected++;
        } else {
            num_unselected++;
        }
    }

    // Re-iterate to build reference
    int sel_idx = 0;
    int unsel_idx = num_selected;
    for (int i = 0; i < num_items; ++i) {
        if (select_op(h_in[i])) {
            h_reference[sel_idx++] = h_in[i];
        } else {
            h_reference[unsel_idx++] = h_in[i];
        }
    }

    // Allocate device memory
    T* d_in = nullptr;
    T* d_out = nullptr;
    int* d_num_selected = nullptr;
    musaMalloc(&d_in, num_items * sizeof(T));
    musaMalloc(&d_out, num_items * sizeof(T));
    musaMalloc(&d_num_selected, sizeof(int));

    musaMemcpy(d_in, h_in.data(), num_items * sizeof(T), musaMemcpyHostToDevice);
    musaMemset(d_out, 0, num_items * sizeof(T));
    musaMemset(d_num_selected, 0, sizeof(int));

    // Run DevicePartition::If
    void* d_temp_storage = nullptr;
    std::size_t temp_storage_bytes = 0;
    CubDebugExit(cub::DevicePartition::If(
        d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected, num_items, select_op));

    musaMalloc(&d_temp_storage, temp_storage_bytes);
    CubDebugExit(cub::DevicePartition::If(
        d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected, num_items, select_op));

    // Copy results back
    int h_num_selected = 0;
    musaMemcpy(h_out.data(), d_out, num_items * sizeof(T), musaMemcpyDeviceToHost);
    musaMemcpy(&h_num_selected, d_num_selected, sizeof(int), musaMemcpyDeviceToHost);

    // Check results
    bool data_ok = true;
    for (int i = 0; i < num_items; ++i) {
        if (h_out[i] != h_reference[i]) {
            printf("Mismatch at index %d: got %lld, expected %lld\n",
                   i, (long long)h_out[i], (long long)h_reference[i]);
            data_ok = false;
            break;
        }
    }
    bool count_ok = (h_num_selected == num_selected);

    if (!count_ok) {
        printf("Count mismatch: got %d, expected %d. ", h_num_selected, num_selected);
    }

    printf("%s\n", (data_ok && count_ok) ? "PASS" : "FAIL");

    // Cleanup
    musaFree(d_temp_storage);
    musaFree(d_in);
    musaFree(d_out);
    musaFree(d_num_selected);
}


template <typename T>
void TestType(int num_items)
{
    std::mt19937 rng(12345);

    // Test with LessThan selector (select ~50% of items)
    LessThan<T> less_than_64(static_cast<T>(64));
    TestSelectIf<T>(num_items, less_than_64, "LessThan(64)", rng);
    TestPartitionIf<T>(num_items, less_than_64, "LessThan(64)", rng);

    // Test with IsEven selector
    IsEven<T> is_even;
    TestSelectIf<T>(num_items, is_even, "IsEven", rng);
    TestPartitionIf<T>(num_items, is_even, "IsEven", rng);

    // Test with edge cases
    LessThan<T> less_than_0(static_cast<T>(0));  // Select none
    TestSelectIf<T>(num_items, less_than_0, "LessThan(0)-select-none", rng);

    LessThan<T> less_than_128(static_cast<T>(128));  // Select all
    TestSelectIf<T>(num_items, less_than_128, "LessThan(128)-select-all", rng);
}


int main(int argc, char** argv)
{
    CommandLineArgs args(argc, argv);

    CubDebugExit(args.DeviceInit());

    printf("=== Device Select/Partition If Test (No Thrust) ===\n\n");

    printf("Testing std::int32_t:\n");
    TestType<std::int32_t>(0);
    TestType<std::int32_t>(1);
    TestType<std::int32_t>(100);
    TestType<std::int32_t>(10000);
    TestType<std::int32_t>(100000);

    printf("\nTesting std::int64_t:\n");
    TestType<std::int64_t>(10000);

    printf("\nTesting std::uint8_t:\n");
    TestType<std::uint8_t>(10000);

    printf("\nAll tests completed.\n");
    return 0;
}
