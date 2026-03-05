/******************************************************************************
 * Test of BlockAdjacentDifference utilities (无 thrust 依赖版本)
 ******************************************************************************/

#define CUB_STDERR

#include <cub/block/block_adjacent_difference.cuh>
#include <cub/util_allocator.cuh>

#include <limits>
#include <memory>
#include <typeinfo>

#include "test_util.h"

using namespace cub;

//---------------------------------------------------------------------
// 自定义类型
//---------------------------------------------------------------------

struct CustomType
{
  unsigned int key;
  unsigned int value;

  __device__ __host__ CustomType()
    : key(0)
    , value(0)
  {}

  __device__ __host__ CustomType(unsigned int key, unsigned int value)
    : key(key)
    , value(value)
  {}
};

__device__ __host__ bool operator==(const CustomType& lhs, const CustomType& rhs)
{
  return lhs.key == rhs.key && lhs.value == rhs.value;
}

__device__ __host__ bool operator!=(const CustomType& lhs, const CustomType& rhs)
{
  return !(lhs == rhs);
}

__device__ __host__ CustomType operator-(const CustomType& lhs, const CustomType& rhs)
{
  return CustomType{lhs.key - rhs.key, lhs.value - rhs.value};
}

struct CustomDifference
{
  template <typename DataType>
  __device__ DataType operator()(const DataType &lhs, const DataType &rhs) const
  {
    return lhs - rhs;
  }
};

//---------------------------------------------------------------------
// 测试内核 - SubtractLeft
//---------------------------------------------------------------------

template <typename DataType,
          unsigned int ThreadsInBlock,
          unsigned int ItemsPerThread>
__global__ void SubtractLeftKernel(const DataType *input,
                                   DataType *output,
                                   unsigned int valid_items,
                                   bool full_tile)
{
  using BlockAdjacentDifferenceT =
    cub::BlockAdjacentDifference<DataType, ThreadsInBlock>;

  __shared__ typename BlockAdjacentDifferenceT::TempStorage temp_storage;

  DataType thread_data[ItemsPerThread];

  const unsigned int thread_offset = threadIdx.x * ItemsPerThread;

  for (unsigned int item = 0; item < ItemsPerThread; item++)
  {
    unsigned int idx = thread_offset + item;
    if (idx < valid_items)
    {
      thread_data[item] = input[idx];
    }
    else
    {
      thread_data[item] = DataType();
    }
  }
  __syncthreads();

  if (full_tile)
  {
    BlockAdjacentDifferenceT(temp_storage).SubtractLeft(
      thread_data, thread_data, CustomDifference());
  }
  else
  {
    BlockAdjacentDifferenceT(temp_storage).SubtractLeftPartialTile(
      thread_data, thread_data, CustomDifference(), valid_items);
  }

  for (unsigned int item = 0; item < ItemsPerThread; item++)
  {
    unsigned int idx = thread_offset + item;
    if (idx < valid_items)
    {
      output[idx] = thread_data[item];
    }
  }
}

//---------------------------------------------------------------------
// 测试内核 - SubtractRight
//---------------------------------------------------------------------

template <typename DataType,
          unsigned int ThreadsInBlock,
          unsigned int ItemsPerThread>
__global__ void SubtractRightKernel(const DataType *input,
                                    DataType *output,
                                    unsigned int valid_items,
                                    bool full_tile)
{
  using BlockAdjacentDifferenceT =
    cub::BlockAdjacentDifference<DataType, ThreadsInBlock>;

  __shared__ typename BlockAdjacentDifferenceT::TempStorage temp_storage;

  DataType thread_data[ItemsPerThread];

  const unsigned int thread_offset = threadIdx.x * ItemsPerThread;

  for (unsigned int item = 0; item < ItemsPerThread; item++)
  {
    unsigned int idx = thread_offset + item;
    if (idx < valid_items)
    {
      thread_data[item] = input[idx];
    }
    else
    {
      thread_data[item] = DataType();
    }
  }
  __syncthreads();

  if (full_tile)
  {
    BlockAdjacentDifferenceT(temp_storage).SubtractRight(
      thread_data, thread_data, CustomDifference());
  }
  else
  {
    BlockAdjacentDifferenceT(temp_storage).SubtractRightPartialTile(
      thread_data, thread_data, CustomDifference(), valid_items);
  }

  for (unsigned int item = 0; item < ItemsPerThread; item++)
  {
    unsigned int idx = thread_offset + item;
    if (idx < valid_items)
    {
      output[idx] = thread_data[item];
    }
  }
}

//---------------------------------------------------------------------
// 辅助函数
//---------------------------------------------------------------------

template <typename T>
void simple_fill(T *d_data, size_t n, T value)
{
  T *h_data = new T[n];
  for (size_t i = 0; i < n; ++i)
    h_data[i] = value;
  musaMemcpy(d_data, h_data, n * sizeof(T), musaMemcpyHostToDevice);
  delete[] h_data;
}

template <typename T>
void simple_sequence(T *d_data, size_t n, T start = T(0))
{
  T *h_data = new T[n];
  for (size_t i = 0; i < n; ++i)
    h_data[i] = static_cast<T>(start + i);
  musaMemcpy(d_data, h_data, n * sizeof(T), musaMemcpyHostToDevice);
  delete[] h_data;
}

template <typename T>
void simple_copy_to_host(const T *d_data, T *h_data, size_t n)
{
  musaMemcpy(h_data, d_data, n * sizeof(T), musaMemcpyDeviceToHost);
}

//---------------------------------------------------------------------
// 测试函数
//---------------------------------------------------------------------

template <typename DataType,
          unsigned int ItemsPerThread,
          unsigned int ThreadsInBlock>
void TestSubtractLeft()
{
  constexpr unsigned int tile_size = ItemsPerThread * ThreadsInBlock;

  DataType *d_input = nullptr;
  DataType *d_output = nullptr;
  musaMalloc(&d_input, tile_size * sizeof(DataType));
  musaMalloc(&d_output, tile_size * sizeof(DataType));

  DataType *h_input = new DataType[tile_size];
  DataType *h_output = new DataType[tile_size];

  // Full tile test
  simple_sequence(d_input, tile_size, DataType(1));
  musaMemset(d_output, 0, tile_size * sizeof(DataType));

  SubtractLeftKernel<DataType, ThreadsInBlock, ItemsPerThread>
    <<<1, ThreadsInBlock>>>(d_input, d_output, tile_size, true);

  musaDeviceSynchronize();
  simple_copy_to_host(d_output, h_output, tile_size);

  // SubtractLeft: output[i] = input[i] - input[i-1]
  // 对于序列 1,2,3,...,n: output[0]=1, output[i]=1 (i>0)
  bool pass = true;
  if (h_output[0] != DataType(1)) pass = false;
  for (unsigned int i = 1; i < tile_size && pass; ++i)
  {
    if (h_output[i] != DataType(1)) pass = false;
  }

  printf("Test<%s, %u items/thread, %u threads> SubtractLeft FullTile: %s\n",
         typeid(DataType).name(), ItemsPerThread, ThreadsInBlock,
         pass ? "PASS" : "FAIL");

  // Partial tile test
  for (unsigned int num_items = tile_size; num_items > 1; num_items /= 2)
  {
    simple_sequence(d_input, num_items, DataType(1));
    musaMemset(d_output, 0, tile_size * sizeof(DataType));

    SubtractLeftKernel<DataType, ThreadsInBlock, ItemsPerThread>
      <<<1, ThreadsInBlock>>>(d_input, d_output, num_items, false);

    musaDeviceSynchronize();
    simple_copy_to_host(d_output, h_output, num_items);

    pass = true;
    if (h_output[0] != DataType(1)) pass = false;
    for (unsigned int i = 1; i < num_items && pass; ++i)
    {
      if (h_output[i] != DataType(1)) pass = false;
    }

    printf("Test<%s, %u items/thread, %u threads> SubtractLeft PartialTile(%u): %s\n",
           typeid(DataType).name(), ItemsPerThread, ThreadsInBlock, num_items,
           pass ? "PASS" : "FAIL");
  }

  musaFree(d_input);
  musaFree(d_output);
  delete[] h_input;
  delete[] h_output;
}

template <typename DataType,
          unsigned int ItemsPerThread,
          unsigned int ThreadsInBlock>
void TestSubtractRight()
{
  constexpr unsigned int tile_size = ItemsPerThread * ThreadsInBlock;

  DataType *d_input = nullptr;
  DataType *d_output = nullptr;
  musaMalloc(&d_input, tile_size * sizeof(DataType));
  musaMalloc(&d_output, tile_size * sizeof(DataType));

  DataType *h_output = new DataType[tile_size];

  // Full tile test
  simple_sequence(d_input, tile_size, DataType(1));
  musaMemset(d_output, 0, tile_size * sizeof(DataType));

  SubtractRightKernel<DataType, ThreadsInBlock, ItemsPerThread>
    <<<1, ThreadsInBlock>>>(d_input, d_output, tile_size, true);

  musaDeviceSynchronize();
  simple_copy_to_host(d_output, h_output, tile_size);

  // SubtractRight: output[i] = input[i] - input[i+1]
  // 对于序列 1,2,3,...,n: output[i]=-1 (i<n-1), output[n-1]保持不变
  bool pass = true;
  for (unsigned int i = 0; i < tile_size - 1 && pass; ++i)
  {
    if (h_output[i] != DataType(-1)) pass = false;
  }
  // 最后一个元素应该保持不变
  if (h_output[tile_size - 1] != DataType(tile_size)) pass = false;

  printf("Test<%s, %u items/thread, %u threads> SubtractRight FullTile: %s\n",
         typeid(DataType).name(), ItemsPerThread, ThreadsInBlock,
         pass ? "PASS" : "FAIL");

  // Partial tile test
  for (unsigned int num_items = tile_size; num_items > 1; num_items /= 2)
  {
    simple_sequence(d_input, num_items, DataType(1));
    musaMemset(d_output, 0, tile_size * sizeof(DataType));

    SubtractRightKernel<DataType, ThreadsInBlock, ItemsPerThread>
      <<<1, ThreadsInBlock>>>(d_input, d_output, num_items, false);

    musaDeviceSynchronize();
    simple_copy_to_host(d_output, h_output, num_items);

    pass = true;
    for (unsigned int i = 0; i < num_items - 1 && pass; ++i)
    {
      if (h_output[i] != DataType(-1)) pass = false;
    }

    printf("Test<%s, %u items/thread, %u threads> SubtractRight PartialTile(%u): %s\n",
           typeid(DataType).name(), ItemsPerThread, ThreadsInBlock, num_items,
           pass ? "PASS" : "FAIL");
  }

  musaFree(d_input);
  musaFree(d_output);
  delete[] h_output;
}

template <typename DataType,
          unsigned int ItemsPerThread,
          unsigned int ThreadsInBlock>
void Test()
{
  TestSubtractLeft<DataType, ItemsPerThread, ThreadsInBlock>();
  TestSubtractRight<DataType, ItemsPerThread, ThreadsInBlock>();
}

template <unsigned int ItemsPerThread, unsigned int ThreadsPerBlock>
void TestTypes()
{
  Test<std::uint8_t,  ItemsPerThread, ThreadsPerBlock>();
  Test<std::uint16_t, ItemsPerThread, ThreadsPerBlock>();
  Test<std::uint32_t, ItemsPerThread, ThreadsPerBlock>();
  Test<std::uint64_t, ItemsPerThread, ThreadsPerBlock>();
}

template <unsigned int ItemsPerThread>
void TestAll()
{
  TestTypes<ItemsPerThread, 32>();
  TestTypes<ItemsPerThread, 256>();
}

//---------------------------------------------------------------------
// 主函数
//---------------------------------------------------------------------

int main(int argc, char** argv)
{
  CommandLineArgs args(argc, argv);

  // Initialize device
  CubDebugExit(args.DeviceInit());

  printf("=== Block Adjacent Difference Test (No Thrust) ===\n\n");

  TestAll<1>();
  TestAll<2>();
  TestAll<10>();
  TestAll<15>();

  printf("\nAll tests completed.\n");

  return 0;
}
