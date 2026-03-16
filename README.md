# 测试方法

## 快速开始

确保已安装 Python 和 Ninja，然后直接运行 `build_cub.sh`：

```bash
./build_cub.sh
```

## build_cub.sh 用法详解

### 命令

```bash
./build_cub.sh [选项] [命令]
```

**命令：**
- 无参数：执行完整流程（清理、编译、测试、生成报告）
- `build`：仅编译，不运行测试
- `clean`：仅清理 build 目录

### 选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `-j, --jobs N` | 编译并行数 | `nproc` (CPU 核心数) |
| `-T, --test-jobs N` | 测试并行数 | 1 |
| `-g, --gpus DEVICES` | 设置 MUSA_VISIBLE_DEVICES | 所有 GPU 可见 |
| `-a, --arch ARCH` | MUSA 目标架构 | `mp_31` |
| `-n, --no-clean` | 不删除 build 目录（增量编译） | 否 |
| `-E, --exclude RE` | 排除匹配正则表达式的测试 | `grid_barrier\|namespace_wrapped` |
| `-h, --help` | 显示帮助信息 | - |

### 架构选择

MUSA 架构决定了编译目标硬件：

| 架构 | 对应硬件 | PTX 版本 | 说明 |
|------|----------|----------|------|
| `mp_22` | S4000 系列 | 220 | Shared Memory 28KB，Warp 128 线程 |
| `mp_31` | S5000 系列 | 310 | Shared Memory 较大，Warp 32 线程 |

**重要：** 编译时必须指定正确的目标架构，否则运行时会报 "invalid device function" 错误。

### 环境变量

所有参数均可通过环境变量设置，命令行参数优先级更高：

| 环境变量 | 说明 |
|----------|------|
| `CUB_JOBS` | 编译并行数 |
| `CUB_TEST_JOBS` | 测试并行数 |
| `CUB_MUSA_ARCH` | MUSA 目标架构 |
| `CUB_MUSA_DEVICES` | 设置 MUSA_VISIBLE_DEVICES |
| `CUB_NO_CLEAN` | 设置为 1 不删除 build 目录 |
| `CUB_EXCLUDE_TESTS` | 排除匹配正则表达式的测试 |
| `CUB_BUILD_ONLY` | 设置为 1 仅编译不测试 |

### 示例

```bash
# 完整流程：清理、编译、测试、生成报告
./build_cub.sh

# 仅编译（不测试）
./build_cub.sh build

# 增量编译并测试
./build_cub.sh -n

# 为 S4000 (mp_22) 编译
./build_cub.sh -a mp_22 build

# 使用 8 个并行测试，指定 GPU 0-7
./build_cub.sh -T 8 -g 0,1,2,3,4,5,6,7

# 使用环境变量配置
export CUB_MUSA_ARCH=mp_22
export CUB_TEST_JOBS=4
./build_cub.sh

# 排除特定测试
./build_cub.sh -E "warp_exchange|device_radix_sort"

# 取消默认排除（运行所有测试）
./build_cub.sh -E ""
```

### 输出文件

- `test_verbose.log`：详细测试输出日志
- `test_report.md`：Markdown 格式的测试报告

如有问题，请联系 MingXu。

<hr>
<h3>About CUB</h3>

CUB provides state-of-the-art, reusable software components for every layer
of the CUDA programming model:
- [<b><em>Device-wide primitives</em></b>](https://nvlabs.github.io/cub/group___device_module.html)
  - Sort, prefix scan, reduction, histogram, etc.
  - Compatible with CUDA dynamic parallelism
- [<b><em>Block-wide "collective" primitives</em></b>](https://nvlabs.github.io/cub/group___block_module.html)
  - I/O, sort, prefix scan, reduction, histogram, etc.
  - Compatible with arbitrary thread block sizes and types
- [<b><em>Warp-wide "collective" primitives</em></b>](https://nvlabs.github.io/cub/group___warp_module.html)
  - Warp-wide prefix scan, reduction, etc.
  - Safe and architecture-specific
- [<b><em>Thread and resource utilities</em></b>](https://nvlabs.github.io/cub/group___util_io.html)
  - PTX intrinsics, device reflection, texture-caching iterators, caching memory allocators, etc.

![Orientation of collective primitives within the CUDA software stack](http://nvlabs.github.io/cub/cub_overview.png)

CUB is included in the NVIDIA HPC SDK and the CUDA Toolkit.

We recommend the [CUB Project Website](http://nvlabs.github.io/cub) for further information and examples.

<br><hr>
<h3>A Simple Example</h3>

```C++
#include <cub/cub.cuh>

// Block-sorting CUDA kernel
__global__ void BlockSortKernel(int *d_in, int *d_out)
{
     using namespace cub;

     // Specialize BlockRadixSort, BlockLoad, and BlockStore for 128 threads
     // owning 16 integer items each
     typedef BlockRadixSort<int, 128, 16>                     BlockRadixSort;
     typedef BlockLoad<int, 128, 16, BLOCK_LOAD_TRANSPOSE>   BlockLoad;
     typedef BlockStore<int, 128, 16, BLOCK_STORE_TRANSPOSE> BlockStore;

     // Allocate shared memory
     __shared__ union {
         typename BlockRadixSort::TempStorage  sort;
         typename BlockLoad::TempStorage       load;
         typename BlockStore::TempStorage      store;
     } temp_storage;

     int block_offset = blockIdx.x * (128 * 16);	  // OffsetT for this block's ment

     // Obtain a segment of 2048 consecutive keys that are blocked across threads
     int thread_keys[16];
     BlockLoad(temp_storage.load).Load(d_in + block_offset, thread_keys);
     __syncthreads();

     // Collectively sort the keys
     BlockRadixSort(temp_storage.sort).Sort(thread_keys);
     __syncthreads();

     // Store the sorted segment
     BlockStore(temp_storage.store).Store(d_out + block_offset, thread_keys);
}
```

Each thread block uses `cub::BlockRadixSort` to collectively sort
its own input segment.  The class is specialized by the
data type being sorted, by the number of threads per block, by the number of
keys per thread, and implicitly by the targeted compilation architecture.

The `cub::BlockLoad` and `cub::BlockStore` classes are similarly specialized.
Furthermore, to provide coalesced accesses to device memory, these primitives are
configured to access memory using a striped access pattern (where consecutive threads
simultaneously access consecutive items) and then <em>transpose</em> the keys into
a [<em>blocked arrangement</em>](index.html#sec4sec3) of elements across threads.

Once specialized, these classes expose opaque `TempStorage` member types.
The thread block uses these storage types to statically allocate the union of
shared memory needed by the thread block.  (Alternatively these storage types
could be aliased to global memory allocations).

<br><hr>
<h3>Supported Compilers</h3>

CUB is regularly tested using the specified versions of the following
compilers. Unsupported versions may emit deprecation warnings, which can be
silenced by defining CUB_IGNORE_DEPRECATED_COMPILER during compilation.

- NVCC 11.0+
- GCC 5+
- Clang 7+
- MSVC 2019+ (19.20/16.0/14.20)

<br><hr>
<h3>Releases</h3>

CUB is distributed with the NVIDIA HPC SDK and the CUDA Toolkit in addition
to GitHub.

See the [changelog](CHANGELOG.md) for details about specific releases.

| CUB Release               | Included In                             |
| ------------------------- | --------------------------------------- |
| 1.17.2                    | TBD                                     |
| 1.17.1                    | TBD                                     |
| 1.17.0                    | TBD                                     |
| 1.16.0                    | TBD                                     |
| 1.15.0                    | NVIDIA HPC SDK 22.1 & CUDA Toolkit 11.6 |
| 1.14.0                    | NVIDIA HPC SDK 21.9                     |
| 1.13.1                    | CUDA Toolkit 11.5                       |
| 1.13.0                    | NVIDIA HPC SDK 21.7                     |
| 1.12.1                    | CUDA Toolkit 11.4                       |
| 1.12.0                    | NVIDIA HPC SDK 21.3                     |
| 1.11.0                    | CUDA Toolkit 11.3                       |
| 1.10.0                    | NVIDIA HPC SDK 20.9 & CUDA Toolkit 11.2 |
| 1.9.10-1                  | NVIDIA HPC SDK 20.7 & CUDA Toolkit 11.1 |
| 1.9.10                    | NVIDIA HPC SDK 20.5                     |
| 1.9.9                     | CUDA Toolkit 11.0                       |
| 1.9.8-1                   | NVIDIA HPC SDK 20.3                     |
| 1.9.8                     | CUDA Toolkit 11.0 Early Access          |
| 1.9.8                     | CUDA 11.0 Early Access                  |
| 1.8.0                     |                                         |
| 1.7.5                     | Thrust 1.9.2                            |
| 1.7.4                     | Thrust 1.9.1-2                          |
| 1.7.3                     |                                         |
| 1.7.2                     |                                         |
| 1.7.1                     |                                         |
| 1.7.0                     | Thrust 1.9.0-5                          |
| 1.6.4                     |                                         |
| 1.6.3                     |                                         |
| 1.6.2 (previously 1.5.5)  |                                         |
| 1.6.1 (previously 1.5.4)  |                                         |
| 1.6.0 (previously 1.5.3)  |                                         |
| 1.5.2                     |                                         |
| 1.5.1                     |                                         |
| 1.5.0                     |                                         |
| 1.4.1                     |                                         |
| 1.4.0                     |                                         |
| 1.3.2                     |                                         |
| 1.3.1                     |                                         |
| 1.3.0                     |                                         |
| 1.2.3                     |                                         |
| 1.2.2                     |                                         |
| 1.2.0                     |                                         |
| 1.1.1                     |                                         |
| 1.0.2                     |                                         |
| 1.0.1                     |                                         |
| 0.9.4                     |                                         |
| 0.9.2                     |                                         |
| 0.9.1                     |                                         |
| 0.9.0                     |                                         |

<br><hr>
<h3>Development Process</h3>

CUB and Thrust depend on each other. It is recommended to clone Thrust
and build CUB as a component of Thrust.

CUB uses the [CMake build system](https://cmake.org/) to build unit tests,
examples, and header tests. To build CUB as a developer, the following
recipe should be followed:

```
# Clone Thrust and CUB from Github. CUB is located in Thrust's
# `dependencies/cub` submodule.
git clone --recursive https://github.com/NVIDIA/thrust.git
cd thrust

# Create build directory:
mkdir build
cd build

# Configure -- use one of the following:
cmake -DTHRUST_INCLUDE_CUB_CMAKE=ON ..   # Command line interface.
ccmake -DTHRUST_INCLUDE_CUB_CMAKE=ON ..  # ncurses GUI (Linux only)
cmake-gui  # Graphical UI, set source/build directories and options in the app

# Build:
cmake --build . -j <num jobs>   # invokes make (or ninja, etc)

# Run tests and examples:
ctest
```

By default, the C++14 standard is targeted, but this can be changed in CMake.
More information on configuring your CUB build and creating a pull request is
found in [CONTRIBUTING.md](CONTRIBUTING.md).

<br><hr>
<h3>Open Source License</h3>

CUB is available under the "New BSD" open-source license:

```
Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
   *  Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
   *  Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
   *  Neither the name of the NVIDIA CORPORATION nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
