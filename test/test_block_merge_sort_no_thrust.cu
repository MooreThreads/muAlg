/******************************************************************************
 * 测试 BlockMergeSort（不使用 thrust）
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <typeinfo>

#include <musa_runtime.h>

#include <cub/util_allocator.cuh>
#include <cub/block/block_merge_sort.cuh>

#define CUB_STDERR
#include "test_util.h"

using namespace cub;

//---------------------------------------------------------------------
// 辅助函数
//---------------------------------------------------------------------

/// 简单的 parallel_for
template <typename F, typename Size>
__global__ void ParallelForKernel(F f, Size num_items) {
  Size idx = static_cast<Size>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_items) {
    f(idx);
  }
}

template <typename F, typename Size>
void ParallelFor(F f, Size num_items) {
  if (num_items == 0) return;
  const int block_size = 256;
  const int num_blocks = static_cast<int>((num_items + block_size - 1) / block_size);
  ParallelForKernel<<<num_blocks, block_size>>>(f, num_items);
  CubDebugExit(musaPeekAtLastError());
  CubDebugExit(musaDeviceSynchronize());
}

//---------------------------------------------------------------------
// 自定义类型和比较器
//---------------------------------------------------------------------

struct CustomType {
  std::uint8_t key;
  std::uint64_t count;

  __device__ __host__ CustomType() : key(0), count(0) {}

  __device__ __host__ CustomType(std::uint64_t value)
      : key(static_cast<std::uint8_t>(value)), count(value) {}

  __device__ __host__ void operator=(std::uint64_t value) {
    key = static_cast<std::uint8_t>(value);
    count = value;
  }

  __device__ __host__ bool operator==(const CustomType &other) const {
    return key == other.key && count == other.count;
  }

  __device__ __host__ bool operator!=(const CustomType &other) const {
    return !(*this == other);
  }

  __device__ __host__ bool operator<(const CustomType &other) const {
    if (key == other.key) return count < other.count;
    return key < other.key;
  }
};

struct CustomLess {
  template <typename DataType>
  __device__ __host__ bool operator()(const DataType &lhs, const DataType &rhs) const {
    return lhs < rhs;
  }

  __device__ bool operator()(CustomType &lhs, CustomType &rhs) {
    return lhs.key < rhs.key;
  }
};

//---------------------------------------------------------------------
// 测试内核
//---------------------------------------------------------------------

template <typename DataType, unsigned int ThreadsInBlock, unsigned int ItemsPerThread,
          bool Stable = false>
__global__ void BlockMergeSortTestKernel(DataType *data, unsigned int valid_items) {
  using BlockMergeSort = cub::BlockMergeSort<DataType, ThreadsInBlock, ItemsPerThread>;

  __shared__ typename BlockMergeSort::TempStorage temp_storage_shuffle;

  DataType thread_data[ItemsPerThread];

  const unsigned int thread_offset = threadIdx.x * ItemsPerThread;

  for (unsigned int item = 0; item < ItemsPerThread; item++) {
    const unsigned int idx = thread_offset + item;
    thread_data[item] = idx < valid_items ? data[idx] : DataType();
  }
  __syncthreads();

  const DataType oob_default =
      static_cast<std::uint64_t>(ThreadsInBlock * ItemsPerThread + 1);

  if (Stable) {
    if (valid_items == ThreadsInBlock * ItemsPerThread) {
      BlockMergeSort(temp_storage_shuffle).StableSort(thread_data, CustomLess());
    } else {
      BlockMergeSort(temp_storage_shuffle).StableSort(thread_data, CustomLess(),
                                                       valid_items, oob_default);
    }
  } else {
    if (valid_items == ThreadsInBlock * ItemsPerThread) {
      BlockMergeSort(temp_storage_shuffle).Sort(thread_data, CustomLess());
    } else {
      BlockMergeSort(temp_storage_shuffle).Sort(thread_data, CustomLess(), valid_items,
                                                 oob_default);
    }
  }

  for (unsigned int item = 0; item < ItemsPerThread; item++) {
    const unsigned int idx = thread_offset + item;
    if (idx >= valid_items) break;
    data[idx] = thread_data[item];
  }
}

template <typename KeyType, typename ValueType, unsigned int ThreadsInBlock,
          unsigned int ItemsPerThread, bool Stable = false>
__global__ void BlockMergeSortTestKernel(KeyType *keys, ValueType *values,
                                         unsigned int valid_items) {
  using BlockMergeSort =
      cub::BlockMergeSort<KeyType, ThreadsInBlock, ItemsPerThread, ValueType>;

  __shared__ typename BlockMergeSort::TempStorage temp_storage_shuffle;

  KeyType thread_keys[ItemsPerThread];
  ValueType thread_values[ItemsPerThread];

  const unsigned int thread_offset = threadIdx.x * ItemsPerThread;

  for (unsigned int item = 0; item < ItemsPerThread; item++) {
    const unsigned int idx = thread_offset + item;
    thread_keys[item] = idx < valid_items ? keys[idx] : KeyType();
    thread_values[item] = idx < valid_items ? values[idx] : ValueType();
  }
  __syncthreads();

  const KeyType oob_default = ThreadsInBlock * ItemsPerThread + 1;

  if (Stable) {
    if (valid_items == ThreadsInBlock * ItemsPerThread) {
      BlockMergeSort(temp_storage_shuffle).StableSort(thread_keys, thread_values,
                                                       CustomLess());
    } else {
      BlockMergeSort(temp_storage_shuffle).StableSort(thread_keys, thread_values,
                                                       CustomLess(), valid_items, oob_default);
    }
  } else {
    if (valid_items == ThreadsInBlock * ItemsPerThread) {
      BlockMergeSort(temp_storage_shuffle).Sort(thread_keys, thread_values, CustomLess());
    } else {
      BlockMergeSort(temp_storage_shuffle).Sort(thread_keys, thread_values, CustomLess(),
                                                 valid_items, oob_default);
    }
  }

  for (unsigned int item = 0; item < ItemsPerThread; item++) {
    const unsigned int idx = thread_offset + item;
    if (idx >= valid_items) break;
    keys[idx] = thread_keys[item];
    values[idx] = thread_values[item];
  }
}

//---------------------------------------------------------------------
// 测试函数
//---------------------------------------------------------------------

template <typename DataType, unsigned int ItemsPerThread, unsigned int ThreadsInBlock,
          bool Stable = false>
void BlockMergeSortTest(DataType *data, unsigned int valid_items) {
  BlockMergeSortTestKernel<DataType, ThreadsInBlock, ItemsPerThread, Stable>
      <<<1, ThreadsInBlock>>>(data, valid_items);

  CubDebugExit(musaPeekAtLastError());
  CubDebugExit(musaDeviceSynchronize());
}

template <typename KeyType, typename ValueType, unsigned int ItemsPerThread,
          unsigned int ThreadsInBlock>
void BlockMergeSortTest(KeyType *keys, ValueType *values, unsigned int valid_items) {
  BlockMergeSortTestKernel<KeyType, ValueType, ThreadsInBlock, ItemsPerThread>
      <<<1, ThreadsInBlock>>>(keys, values, valid_items);

  CubDebugExit(musaPeekAtLastError());
  CubDebugExit(musaDeviceSynchronize());
}

/// 初始化数组为序列
template <typename T> void InitSequence(T *h_data, int num_items) {
  for (int i = 0; i < num_items; i++) {
    h_data[i] = static_cast<T>(i);
  }
}

/// Fisher-Yates 洗牌
template <typename T> void Shuffle(T *h_data, int num_items, unsigned int seed) {
  srand(seed);
  for (int i = num_items - 1; i > 0; i--) {
    int j = rand() % (i + 1);
    T temp = h_data[i];
    h_data[i] = h_data[j];
    h_data[j] = temp;
  }
}

/// 检查结果是否为排序后的序列 (0, 1, 2, ..., n-1)
template <typename T> bool CheckResult(T *h_data, int num_items) {
  for (int i = 0; i < num_items; i++) {
    if (h_data[i] != static_cast<T>(i)) {
      printf("Check failed at index %d: expected %d, got %d\n", i, i,
             static_cast<int>(h_data[i]));
      return false;
    }
  }
  return true;
}

//---------------------------------------------------------------------
// 测试用例
//---------------------------------------------------------------------

template <typename DataType, unsigned int ItemsPerThread, unsigned int ThreadsInBlock>
void Test(unsigned int num_items, unsigned int seed) {
  // 分配主机内存
  DataType *h_data = new DataType[num_items];

  // 初始化并洗牌
  InitSequence(h_data, num_items);
  Shuffle(h_data, num_items, seed);

  // 分配设备内存
  DataType *d_data;
  CubDebugExit(musaMalloc(&d_data, num_items * sizeof(DataType)));
  CubDebugExit(musaMemcpy(d_data, h_data, num_items * sizeof(DataType), musaMemcpyHostToDevice));

  // 执行排序
  BlockMergeSortTest<DataType, ItemsPerThread, ThreadsInBlock>(d_data, num_items);

  // 拷回主机并检查
  CubDebugExit(musaMemcpy(h_data, d_data, num_items * sizeof(DataType), musaMemcpyDeviceToHost));

  bool passed = CheckResult(h_data, num_items);
  printf("Test<%s, %d items/thread, %d threads> num_items=%d: %s\n",
         typeid(DataType).name(), ItemsPerThread, ThreadsInBlock, num_items,
         passed ? "PASS" : "FAIL");

  // 清理
  CubDebugExit(musaFree(d_data));
  delete[] h_data;

  if (!passed) exit(1);
}

template <typename KeyType, typename ValueType, unsigned int ItemsPerThread,
          unsigned int ThreadsInBlock>
void TestKeyValue(unsigned int num_items, unsigned int seed) {
  // 分配主机内存
  KeyType *h_keys = new KeyType[num_items];
  ValueType *h_values = new ValueType[num_items];

  // 初始化并洗牌
  InitSequence(h_keys, num_items);
  Shuffle(h_keys, num_items, seed);
  for (int i = 0; i < num_items; i++) {
    h_values[i] = static_cast<ValueType>(h_keys[i]); // values = keys
  }

  // 分配设备内存
  KeyType *d_keys;
  ValueType *d_values;
  CubDebugExit(musaMalloc(&d_keys, num_items * sizeof(KeyType)));
  CubDebugExit(musaMalloc(&d_values, num_items * sizeof(ValueType)));
  CubDebugExit(musaMemcpy(d_keys, h_keys, num_items * sizeof(KeyType), musaMemcpyHostToDevice));
  CubDebugExit(
      musaMemcpy(d_values, h_values, num_items * sizeof(ValueType), musaMemcpyHostToDevice));

  // 执行排序
  BlockMergeSortTest<KeyType, ValueType, ItemsPerThread, ThreadsInBlock>(d_keys, d_values,
                                                                          num_items);

  // 拷回主机并检查
  CubDebugExit(musaMemcpy(h_keys, d_keys, num_items * sizeof(KeyType), musaMemcpyDeviceToHost));
  CubDebugExit(
      musaMemcpy(h_values, d_values, num_items * sizeof(ValueType), musaMemcpyDeviceToHost));

  // 检查 values 是否也排序了（应该等于 keys）
  bool passed = true;
  for (int i = 0; i < num_items; i++) {
    if (h_keys[i] != static_cast<KeyType>(i)) {
      printf("Key check failed at index %d: expected %d, got %d\n", i, i,
             static_cast<int>(h_keys[i]));
      passed = false;
      break;
    }
    if (h_values[i] != static_cast<ValueType>(i)) {
      printf("Value check failed at index %d: expected %d, got %d\n", i, i,
             static_cast<int>(h_values[i]));
      passed = false;
      break;
    }
  }

  printf("TestKeyValue<%s, %s, %d items/thread, %d threads> num_items=%d: %s\n",
         typeid(KeyType).name(), typeid(ValueType).name(), ItemsPerThread, ThreadsInBlock,
         num_items, passed ? "PASS" : "FAIL");

  // 清理
  CubDebugExit(musaFree(d_keys));
  CubDebugExit(musaFree(d_values));
  delete[] h_keys;
  delete[] h_values;

  if (!passed) exit(1);
}

template <unsigned int ItemsPerThread, unsigned int ThreadsInBlock>
void TestAll(unsigned int seed) {
  const unsigned int max_items = ItemsPerThread * ThreadsInBlock;

  for (unsigned int num_items = max_items; num_items > 1; num_items /= 2) {
    Test<std::int32_t, ItemsPerThread, ThreadsInBlock>(num_items, seed);
    Test<std::int64_t, ItemsPerThread, ThreadsInBlock>(num_items, seed);
    TestKeyValue<std::int32_t, std::int32_t, ItemsPerThread, ThreadsInBlock>(num_items, seed);
    TestKeyValue<std::int64_t, std::int64_t, ItemsPerThread, ThreadsInBlock>(num_items, seed);
  }
}

//---------------------------------------------------------------------
// 稳定性测试
//---------------------------------------------------------------------

void TestStability() {
  constexpr unsigned int items_per_thread = 10;
  constexpr unsigned int threads_per_block = 128;
  constexpr unsigned int elements = items_per_thread * threads_per_block;
  constexpr bool stable = true;

  // 分配主机内存
  CustomType *h_data = new CustomType[elements];

  // 初始化：key = value & 0xFF (与原版一致)
  // 注意：不洗牌，原版 NVIDIA 测试就是这样做的
  for (unsigned int i = 0; i < elements; i++) {
    h_data[i] = CustomType(i);  // key = i & 0xFF, count = i
  }

  // 分配设备内存
  CustomType *d_data;
  CubDebugExit(musaMalloc(&d_data, elements * sizeof(CustomType)));
  CubDebugExit(musaMemcpy(d_data, h_data, elements * sizeof(CustomType), musaMemcpyHostToDevice));

  // 执行稳定排序
  BlockMergeSortTest<CustomType, items_per_thread, threads_per_block, stable>(d_data, elements);

  // 拷回主机
  CubDebugExit(musaMemcpy(h_data, d_data, elements * sizeof(CustomType), musaMemcpyDeviceToHost));

  // 检查稳定性：相同 key 的元素应该按 count 递增排列
  // 由于原始数据中 count 是递增的，稳定排序应该保持这个顺序
  bool passed = true;
  for (unsigned int i = 1; i < elements; i++) {
    if (h_data[i].key == h_data[i - 1].key && h_data[i].count < h_data[i - 1].count) {
      printf("Stability check failed at index %u: key=%u, count[%d]=%lu > count[%d]=%lu\n", i,
             h_data[i].key, i - 1, h_data[i - 1].count, i, h_data[i].count);
      passed = false;
      break;
    }
    if (h_data[i].key < h_data[i - 1].key) {
      printf("Sort check failed at index %u: key[%d]=%u < key[%d]=%u\n", i, i, h_data[i].key,
             i - 1, h_data[i - 1].key);
      passed = false;
      break;
    }
  }

  printf("TestStability: %s\n", passed ? "PASS" : "FAIL");

  // 清理
  CubDebugExit(musaFree(d_data));
  delete[] h_data;

  if (!passed) exit(1);
}

//---------------------------------------------------------------------
// 主函数
//---------------------------------------------------------------------

int main(int argc, char **argv) {
  CommandLineArgs args(argc, argv);

  // 初始化设备
  CubDebugExit(args.DeviceInit());

  unsigned int seed = 42;

  printf("=== Block Merge Sort Test (No Thrust) ===\n\n");

  TestAll<1, 32>(seed);
  TestAll<1, 256>(seed);
  TestAll<2, 32>(seed);
  TestAll<2, 256>(seed);
  TestAll<10, 32>(seed);
  TestAll<10, 256>(seed);
  TestAll<15, 32>(seed);
  TestAll<15, 256>(seed);

  // 512 threads per block tests
  Test<std::int32_t, 1, 512>(512, seed);
  Test<std::int64_t, 2, 512>(1024, seed);

  TestStability();

  printf("\n=== All tests PASSED ===\n");

  return 0;
}
