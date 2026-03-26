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

/**
 * \file
 * PTX intrinsics
 */


#pragma once

#include "util_type.cuh"
#include "util_arch.cuh"
#include "util_namespace.cuh"
#include "util_debug.cuh"


CUB_NAMESPACE_BEGIN


/**
 * \addtogroup UtilPtx
 * @{
 */


/******************************************************************************
 * PTX helper macros
 ******************************************************************************/

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

/**
 * Register modifier for pointer-types (for inlining PTX assembly)
 */
#if defined(_WIN64) || defined(__LP64__)
    #define __CUB_LP64__ 1
    // 64-bit register modifier for inlined asm
    #define _CUB_ASM_PTR_ "l"
    #define _CUB_ASM_PTR_SIZE_ "u64"
#else
    #define __CUB_LP64__ 0
    // 32-bit register modifier for inlined asm
    #define _CUB_ASM_PTR_ "r"
    #define _CUB_ASM_PTR_SIZE_ "u32"
#endif

#endif // DOXYGEN_SHOULD_SKIP_THIS


/******************************************************************************
 * Inlined PTX intrinsics
 ******************************************************************************/

/**
 * \brief Shift-right then add.  Returns (\p x >> \p shift) + \p addend.
 */
// MUSA: 使用普通 C++ 替代 PTX vshr
__device__ __forceinline__ unsigned int SHR_ADD(
    unsigned int x,
    unsigned int shift,
    unsigned int addend)
{
    return (x >> shift) + addend;
}


/**
 * \brief Shift-left then add.  Returns (\p x << \p shift) + \p addend.
 */
// MUSA: 使用普通 C++ 替代 PTX vshl
__device__ __forceinline__ unsigned int SHL_ADD(
    unsigned int x,
    unsigned int shift,
    unsigned int addend)
{
    return (x << shift) + addend;
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

/**
 * Bitfield-extract.
 */
// MUSA: 使用普通 C++ 替代 PTX bfe
template <typename UnsignedBits, int BYTE_LEN>
__device__ __forceinline__ unsigned int BFE(
    UnsignedBits            source,
    unsigned int            bit_start,
    unsigned int            num_bits,
    Int2Type<BYTE_LEN>      /*byte_len*/)
{
    const unsigned int MASK = (1u << num_bits) - 1;
    return ((unsigned int)source >> bit_start) & MASK;
}


/**
 * Bitfield-extract for 64-bit types.
 */
template <typename UnsignedBits>
__device__ __forceinline__ unsigned int BFE(
    UnsignedBits            source,
    unsigned int            bit_start,
    unsigned int            num_bits,
    Int2Type<8>             /*byte_len*/)
{
    const unsigned long long MASK = (1ull << num_bits) - 1;
    return (source >> bit_start) & MASK;
}

#endif // DOXYGEN_SHOULD_SKIP_THIS

/**
 * \brief Bitfield-extract.  Extracts \p num_bits from \p source starting at bit-offset \p bit_start.  The input \p source may be an 8b, 16b, 32b, or 64b unsigned integer type.
 */
template <typename UnsignedBits>
__device__ __forceinline__ unsigned int BFE(
    UnsignedBits source,
    unsigned int bit_start,
    unsigned int num_bits)
{
    return BFE(source, bit_start, num_bits, Int2Type<sizeof(UnsignedBits)>());
}


/**
 * \brief Bitfield insert.  Inserts the \p num_bits least significant bits of \p y into \p x at bit-offset \p bit_start.
 */
// MUSA: 使用普通 C++ 替代 PTX bfi
__device__ __forceinline__ void BFI(
    unsigned int &ret,
    unsigned int x,
    unsigned int y,
    unsigned int bit_start,
    unsigned int num_bits)
{
    unsigned int mask = (1u << num_bits) - 1;
    ret = (x & ~(mask << bit_start)) | ((y & mask) << bit_start);
}


/**
 * \brief Three-operand add.  Returns \p x + \p y + \p z.
 */
// MUSA: 使用普通 C++ 替代 PTX vadd
__device__ __forceinline__ unsigned int IADD3(unsigned int x, unsigned int y, unsigned int z)
{
    return x + y + z;
}


/**
 * \brief Byte-permute. Pick four arbitrary bytes from two 32-bit registers, and reassemble them into a 32-bit destination register.  For SM2.0 or later.
 *
 * \par
 * The bytes in the two source registers \p a and \p b are numbered from 0 to 7:
 * {\p b, \p a} = {{b7, b6, b5, b4}, {b3, b2, b1, b0}}. For each of the four bytes
 * {b3, b2, b1, b0} selected in the return value, a 4-bit selector is defined within
 * the four lower "nibbles" of \p index: {\p index } = {n7, n6, n5, n4, n3, n2, n1, n0}
 *
 * \par Snippet
 * The code snippet below illustrates byte-permute.
 * \par
 * \code
 * #include <cub/cub.cuh>
 *
 * __global__ void ExampleKernel(...)
 * {
 *     int a        = 0x03020100;
 *     int b        = 0x07060504;
 *     int index    = 0x00007531;
 *
 *     int selected = PRMT(a, b, index);    // 0x07050301
 *
 * \endcode
 *
 */
// MUSA: 使用普通 C++ 替代 PTX prmt
__device__ __forceinline__ int PRMT(unsigned int a, unsigned int b, unsigned int index)
{
    // 从 a 和 b 中根据 index 选择 4 个字节
    // index 的每个 nibble (4位) 选择一个字节: 0-3 从 a, 4-7 从 b
    unsigned int result = 0;
    for (int i = 0; i < 4; ++i) {
        unsigned int selector = (index >> (i * 4)) & 0x7;
        unsigned int byte_val;
        if (selector < 4) {
            byte_val = (a >> (selector * 8)) & 0xFF;
        } else {
            byte_val = (b >> ((selector - 4) * 8)) & 0xFF;
        }
        result |= byte_val << (i * 8);
    }
    return static_cast<int>(result);
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

/**
 * Sync-threads barrier.
 */
// MUSA: 使用 __syncthreads 替代 bar.sync
__device__ __forceinline__ void BAR(int count)
{
    __syncthreads();
}

/**
 * CTA barrier
 */
__device__  __forceinline__ void CTA_SYNC()
{
    __syncthreads();
}


/**
 * CTA barrier with predicate
 */
__device__  __forceinline__ int CTA_SYNC_AND(int p)
{
    return __syncthreads_and(p);
}


/**
 * CTA barrier with predicate
 */
__device__  __forceinline__ int CTA_SYNC_OR(int p)
{
    return __syncthreads_or(p);
}


/**
 * Warp barrier
 */
__device__  __forceinline__ void WARP_SYNC(unsigned int member_mask)
{
#ifdef __MUSA_ARCH__
    __syncwarp(member_mask);
#elif defined(CUB_USE_COOPERATIVE_GROUPS)
    __syncwarp(member_mask);
#endif
}


/**
 * Warp any
 */
__device__  __forceinline__ int WARP_ANY(int predicate, unsigned int member_mask)
{
#ifdef __MUSA_ARCH__
    return __any_sync(member_mask, predicate);
#elif defined(CUB_USE_COOPERATIVE_GROUPS)
    return __any_sync(member_mask, predicate);
#else
    return ::__any(predicate);
#endif
}


/**
 * Warp all
 */
__device__  __forceinline__ int WARP_ALL(int predicate, unsigned int member_mask)
{
#ifdef __MUSA_ARCH__
    return __all_sync(member_mask, predicate);
#elif defined(CUB_USE_COOPERATIVE_GROUPS)
    return __all_sync(member_mask, predicate);
#else
    return ::__all(predicate);
#endif
}


/**
 * Warp ballot
 */
__device__  __forceinline__ int WARP_BALLOT(int predicate, unsigned int member_mask)
{
#ifdef __MUSA_ARCH__
    return __ballot_sync(member_mask, predicate);
#elif defined(CUB_USE_COOPERATIVE_GROUPS)
    return __ballot_sync(member_mask, predicate);
#else
    return __ballot(predicate);
#endif
}


/**
 * Warp synchronous shfl_up
 */
__device__ __forceinline__
unsigned int SHFL_UP_SYNC(unsigned int word, int src_offset, int flags, unsigned int member_mask)
{
#ifdef __MUSA_ARCH__
    return __shfl_up_sync(member_mask, word, src_offset, flags);
#elif defined(CUB_USE_COOPERATIVE_GROUPS)
    return __shfl_up_sync(member_mask, word, src_offset, flags);
#else
    return __shfl_up(word, src_offset, flags);
#endif
}

/**
 * Warp synchronous shfl_down
 */
__device__ __forceinline__
unsigned int SHFL_DOWN_SYNC(unsigned int word, int src_offset, int flags, unsigned int member_mask)
{
#ifdef __MUSA_ARCH__
    return __shfl_down_sync(member_mask, word, src_offset, flags);
#elif defined(CUB_USE_COOPERATIVE_GROUPS)
    return __shfl_down_sync(member_mask, word, src_offset, flags);
#else
    return __shfl_down(word, src_offset, flags);
#endif
}

/**
 * Warp synchronous shfl_idx
 */
__device__ __forceinline__
unsigned int SHFL_IDX_SYNC(unsigned int word, int src_lane, int flags, unsigned int member_mask)
{
#ifdef __MUSA_ARCH__
    return __shfl_sync(member_mask, word, src_lane, flags);
#elif defined(CUB_USE_COOPERATIVE_GROUPS)
    return __shfl_sync(member_mask, word, src_lane, flags);
#else
    return __shfl(word, src_lane, flags);
#endif
}

/**
 * Warp synchronous shfl_idx
 */
__device__ __forceinline__
unsigned int SHFL_IDX_SYNC(unsigned int word, int src_lane, unsigned int member_mask)
{
#ifdef __MUSA_ARCH__
  return __shfl_sync(member_mask, word, src_lane);
#elif defined(CUB_USE_COOPERATIVE_GROUPS)
  return __shfl_sync(member_mask, word, src_lane);
#else
  return __shfl(word, src_lane);
#endif
}

/**
 * Floating point multiply. (Mantissa LSB rounds towards zero.)
 */
// MUSA: 使用普通浮点乘法 (舍入模式可能不同)
__device__ __forceinline__ float FMUL_RZ(float a, float b)
{
    return a * b;
}


/**
 * Floating point multiply-add. (Mantissa LSB rounds towards zero.)
 */
// MUSA: 使用普通浮点乘加 (舍入模式可能不同)
__device__ __forceinline__ float FFMA_RZ(float a, float b, float c)
{
    return a * b + c;
}

#endif // DOXYGEN_SHOULD_SKIP_THIS

/**
 * \brief Terminates the calling thread
 */
// MUSA: 使用 void 返回替代 exit
__device__ __forceinline__ void ThreadExit() {
    return;
}


/**
 * \brief  Abort execution and generate an interrupt to the host CPU
 */
__device__ __forceinline__ void ThreadTrap() {
    __trap();
}


/**
 * \brief Returns the row-major linear thread identifier for a multidimensional thread block
 */
__device__ __forceinline__ int RowMajorTid(int block_dim_x, int block_dim_y, int block_dim_z)
{
    return ((block_dim_z == 1) ? 0 : (threadIdx.z * block_dim_x * block_dim_y)) +
            ((block_dim_y == 1) ? 0 : (threadIdx.y * block_dim_x)) +
            threadIdx.x;
}


/**
 * \brief Returns the warp lane ID of the calling thread
 */
__device__ __forceinline__ unsigned int LaneId()
{
    return __get_laneid() % 32;
}


/**
 * \brief Returns the warp ID of the calling thread.  Warp ID is guaranteed to be unique among warps, but may not correspond to a zero-based ranking within the thread block.
 */
// MUSA: 使用 threadIdx 计算 warp ID (可能与硬件 warpid 不同)
__device__ __forceinline__ unsigned int WarpId()
{
    return (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) / 32;
}

/**
 * @brief Returns the warp mask for a warp of @p LOGICAL_WARP_THREADS threads
 *
 * @par
 * If the number of threads assigned to the virtual warp is not a power of two,
 * it's assumed that only one virtual warp exists.
 *
 * @tparam LOGICAL_WARP_THREADS <b>[optional]</b> The number of threads per
 *                              "logical" warp (may be less than the number of
 *                              hardware warp threads).
 * @param warp_id Id of virtual warp within architectural warp
 */
template <int LOGICAL_WARP_THREADS,
          int PTX_ARCH = CUB_PTX_ARCH>
__host__ __device__ __forceinline__
unsigned int WarpMask(unsigned int warp_id)
{
  constexpr bool is_pow_of_two = PowerOfTwo<LOGICAL_WARP_THREADS>::VALUE;
  constexpr bool is_arch_warp = LOGICAL_WARP_THREADS ==
                                CUB_WARP_THREADS(PTX_ARCH);

  unsigned int member_mask =
    0xFFFFFFFFu >> (CUB_WARP_THREADS(PTX_ARCH) - LOGICAL_WARP_THREADS);

  if (is_pow_of_two && !is_arch_warp)
  {
    member_mask <<= warp_id * LOGICAL_WARP_THREADS;
  }

  return member_mask;
}

/**
 * \brief Returns the warp lane mask of all lanes less than the calling thread
 */
// MUSA: 使用内置函数获取 lane mask (小于当前 lane)
__device__ __forceinline__ unsigned int LaneMaskLt()
{
    return __get_lanemask_lt();
}

/**
 * \brief Returns the warp lane mask of all lanes less than or equal to the calling thread
 */
// MUSA: 使用位运算计算 lane mask (小于等于当前 lane)
__device__ __forceinline__ unsigned int LaneMaskLe()
{
    return (1u << (__get_laneid() % 32 + 1)) - 1;
}

/**
 * \brief Returns the warp lane mask of all lanes greater than the calling thread
 */
// MUSA: 使用位运算计算 lane mask (大于当前 lane)
__device__ __forceinline__ unsigned int LaneMaskGt()
{
    return ~((1u << (__get_laneid() % 32 + 1)) - 1);
}

/**
 * \brief Returns the warp lane mask of all lanes greater than or equal to the calling thread
 */
// MUSA: 使用位运算计算 lane mask (大于等于当前 lane)
__device__ __forceinline__ unsigned int LaneMaskGe()
{
    return ~((1u << (__get_laneid() % 32)) - 1);
}

/** @} */       // end group UtilPtx




/**
 * \brief Shuffle-up for any data type.  Each <em>warp-lane<sub>i</sub></em> obtains the value \p input contributed by <em>warp-lane</em><sub><em>i</em>-<tt>src_offset</tt></sub>.  For thread lanes \e i < src_offset, the thread's own \p input is returned to the thread. ![](shfl_up_logo.png)
 * \ingroup WarpModule
 *
 * \tparam LOGICAL_WARP_THREADS     The number of threads per "logical" warp.  Must be a power-of-two <= 32.
 * \tparam T                        <b>[inferred]</b> The input/output element type
 *
 * \par
 * - Available only for SM3.0 or newer
 *
 * \par Snippet
 * The code snippet below illustrates each thread obtaining a \p double value from the
 * predecessor of its predecessor.
 * \par
 * \code
 * #include <cub/cub.cuh>   // or equivalently <cub/util_ptx.cuh>
 *
 * __global__ void ExampleKernel(...)
 * {
 *     // Obtain one input item per thread
 *     double thread_data = ...
 *
 *     // Obtain item from two ranks below
 *     double peer_data = ShuffleUp<32>(thread_data, 2, 0, 0xffffffff);
 *
 * \endcode
 * \par
 * Suppose the set of input \p thread_data across the first warp of threads is <tt>{1.0, 2.0, 3.0, 4.0, 5.0, ..., 32.0}</tt>.
 * The corresponding output \p peer_data will be <tt>{1.0, 2.0, 1.0, 2.0, 3.0, ..., 30.0}</tt>.
 *
 */
template <
    int LOGICAL_WARP_THREADS,   ///< Number of threads per logical warp
    typename T>
__device__ __forceinline__ T ShuffleUp(
    T               input,              ///< [in] The value to broadcast
    int             src_offset,         ///< [in] The relative down-offset of the peer to read from
    int             first_thread,       ///< [in] Index of first lane in logical warp (typically 0)
    unsigned int    member_mask)        ///< [in] 32-bit mask of participating warp lanes
{
    /// The 5-bit SHFL mask for logically splitting warps into sub-segments starts 8-bits up
    enum {
        SHFL_C = (32 - LOGICAL_WARP_THREADS) << 8
    };

    typedef typename UnitWord<T>::ShuffleWord ShuffleWord;

    const int       WORDS           = (sizeof(T) + sizeof(ShuffleWord) - 1) / sizeof(ShuffleWord);

    T               output;
    ShuffleWord     *output_alias   = reinterpret_cast<ShuffleWord *>(&output);
    ShuffleWord     *input_alias    = reinterpret_cast<ShuffleWord *>(&input);

    unsigned int shuffle_word;
#ifdef __MUSA_ARCH__
    // MUSA: Use LOGICAL_WARP_THREADS as width parameter, not first_thread
    shuffle_word = __shfl_up_sync(member_mask, (unsigned int)input_alias[0], src_offset, LOGICAL_WARP_THREADS);
#else
    shuffle_word = SHFL_UP_SYNC((unsigned int)input_alias[0], src_offset, first_thread | SHFL_C, member_mask);
#endif
    output_alias[0] = shuffle_word;

    #pragma unroll
    for (int WORD = 1; WORD < WORDS; ++WORD)
    {
#ifdef __MUSA_ARCH__
        // MUSA: Use LOGICAL_WARP_THREADS as width parameter, not first_thread
        shuffle_word = __shfl_up_sync(member_mask, (unsigned int)input_alias[WORD], src_offset, LOGICAL_WARP_THREADS);
#else
        shuffle_word       = SHFL_UP_SYNC((unsigned int)input_alias[WORD], src_offset, first_thread | SHFL_C, member_mask);
#endif
        output_alias[WORD] = shuffle_word;
    }

    return output;
}


/**
 * \brief Shuffle-down for any data type.  Each <em>warp-lane<sub>i</sub></em> obtains the value \p input contributed by <em>warp-lane</em><sub><em>i</em>+<tt>src_offset</tt></sub>.  For thread lanes \e i >= WARP_THREADS, the thread's own \p input is returned to the thread.  ![](shfl_down_logo.png)
 * \ingroup WarpModule
 *
 * \tparam LOGICAL_WARP_THREADS     The number of threads per "logical" warp.  Must be a power-of-two <= 32.
 * \tparam T                        <b>[inferred]</b> The input/output element type
 *
 * \par
 * - Available only for SM3.0 or newer
 *
 * \par Snippet
 * The code snippet below illustrates each thread obtaining a \p double value from the
 * successor of its successor.
 * \par
 * \code
 * #include <cub/cub.cuh>   // or equivalently <cub/util_ptx.cuh>
 *
 * __global__ void ExampleKernel(...)
 * {
 *     // Obtain one input item per thread
 *     double thread_data = ...
 *
 *     // Obtain item from two ranks below
 *     double peer_data = ShuffleDown<32>(thread_data, 2, 31, 0xffffffff);
 *
 * \endcode
 * \par
 * Suppose the set of input \p thread_data across the first warp of threads is <tt>{1.0, 2.0, 3.0, 4.0, 5.0, ..., 32.0}</tt>.
 * The corresponding output \p peer_data will be <tt>{3.0, 4.0, 5.0, 6.0, 7.0, ..., 32.0}</tt>.
 *
 */
template <
    int LOGICAL_WARP_THREADS,   ///< Number of threads per logical warp
    typename T>
__device__ __forceinline__ T ShuffleDown(
    T               input,              ///< [in] The value to broadcast
    int             src_offset,         ///< [in] The relative up-offset of the peer to read from
    int             last_thread,        ///< [in] Index of last thread in logical warp (typically 31 for a 32-thread warp)
    unsigned int    member_mask)        ///< [in] 32-bit mask of participating warp lanes
{
    /// The 5-bit SHFL mask for logically splitting warps into sub-segments starts 8-bits up
    enum {
        SHFL_C = (32 - LOGICAL_WARP_THREADS) << 8
    };

    typedef typename UnitWord<T>::ShuffleWord ShuffleWord;

    const int       WORDS           = (sizeof(T) + sizeof(ShuffleWord) - 1) / sizeof(ShuffleWord);

    T               output;
    ShuffleWord     *output_alias   = reinterpret_cast<ShuffleWord *>(&output);
    ShuffleWord     *input_alias    = reinterpret_cast<ShuffleWord *>(&input);

    unsigned int shuffle_word;
#ifdef __MUSA_ARCH__
    // MUSA: Use LOGICAL_WARP_THREADS as width parameter, not last_thread
    shuffle_word    = __shfl_down_sync(member_mask, (unsigned int)input_alias[0], src_offset, LOGICAL_WARP_THREADS);
#else
    shuffle_word    = SHFL_DOWN_SYNC((unsigned int)input_alias[0], src_offset, last_thread | SHFL_C, member_mask);
#endif
    output_alias[0] = shuffle_word;

    #pragma unroll
    for (int WORD = 1; WORD < WORDS; ++WORD)
    {
#ifdef __MUSA_ARCH__
        // MUSA: Use LOGICAL_WARP_THREADS as width parameter, not last_thread
        shuffle_word = __shfl_down_sync(member_mask, (unsigned int)input_alias[WORD], src_offset, LOGICAL_WARP_THREADS);
#else
        shuffle_word       = SHFL_DOWN_SYNC((unsigned int)input_alias[WORD], src_offset, last_thread | SHFL_C, member_mask);
#endif
        output_alias[WORD] = shuffle_word;
    }

    return output;
}


/**
 * \brief Shuffle-broadcast for any data type.  Each <em>warp-lane<sub>i</sub></em> obtains the value \p input
 * contributed by <em>warp-lane</em><sub><tt>src_lane</tt></sub>.  For \p src_lane < 0 or \p src_lane >= WARP_THREADS,
 * then the thread's own \p input is returned to the thread. ![](shfl_broadcast_logo.png)
 *
 * \tparam LOGICAL_WARP_THREADS     The number of threads per "logical" warp.  Must be a power-of-two <= 32.
 * \tparam T                        <b>[inferred]</b> The input/output element type
 *
 * \ingroup WarpModule
 *
 * \par
 * - Available only for SM3.0 or newer
 *
 * \par Snippet
 * The code snippet below illustrates each thread obtaining a \p double value from <em>warp-lane</em><sub>0</sub>.
 *
 * \par
 * \code
 * #include <cub/cub.cuh>   // or equivalently <cub/util_ptx.cuh>
 *
 * __global__ void ExampleKernel(...)
 * {
 *     // Obtain one input item per thread
 *     double thread_data = ...
 *
 *     // Obtain item from thread 0
 *     double peer_data = ShuffleIndex<32>(thread_data, 0, 0xffffffff);
 *
 * \endcode
 * \par
 * Suppose the set of input \p thread_data across the first warp of threads is <tt>{1.0, 2.0, 3.0, 4.0, 5.0, ..., 32.0}</tt>.
 * The corresponding output \p peer_data will be <tt>{1.0, 1.0, 1.0, 1.0, 1.0, ..., 1.0}</tt>.
 *
 */
template <
    int LOGICAL_WARP_THREADS,   ///< Number of threads per logical warp
    typename T>
__device__ __forceinline__ T ShuffleIndex(
    T               input,                  ///< [in] The value to broadcast
    int             src_lane,               ///< [in] Which warp lane is to do the broadcasting
    unsigned int    member_mask)            ///< [in] 32-bit mask of participating warp lanes
{
    /// The 5-bit SHFL mask for logically splitting warps into sub-segments starts 8-bits up
    enum {
        SHFL_C = ((32 - LOGICAL_WARP_THREADS) << 8) | (LOGICAL_WARP_THREADS - 1)
    };

    typedef typename UnitWord<T>::ShuffleWord ShuffleWord;

    const int       WORDS           = (sizeof(T) + sizeof(ShuffleWord) - 1) / sizeof(ShuffleWord);

    T               output;
    ShuffleWord     *output_alias   = reinterpret_cast<ShuffleWord *>(&output);
    ShuffleWord     *input_alias    = reinterpret_cast<ShuffleWord *>(&input);

    unsigned int shuffle_word;
#ifdef __MUSA_ARCH__
    // MUSA: Use LOGICAL_WARP_THREADS as width parameter for ShuffleIndex
    shuffle_word = __shfl_sync(member_mask, (unsigned int)input_alias[0], src_lane, LOGICAL_WARP_THREADS);
#else
    shuffle_word = SHFL_IDX_SYNC((unsigned int)input_alias[0],
                                 src_lane,
                                 SHFL_C,
                                 member_mask);
#endif

    output_alias[0] = shuffle_word;

    #pragma unroll
    for (int WORD = 1; WORD < WORDS; ++WORD)
    {
#ifdef __MUSA_ARCH__
        // MUSA: Use LOGICAL_WARP_THREADS as width parameter for ShuffleIndex
        shuffle_word = __shfl_sync(member_mask, (unsigned int)input_alias[WORD], src_lane, LOGICAL_WARP_THREADS);
#else
        shuffle_word = SHFL_IDX_SYNC((unsigned int)input_alias[WORD],
                                     src_lane,
                                     SHFL_C,
                                     member_mask);
#endif

        output_alias[WORD] = shuffle_word;
    }

    return output;
}



/**
 * Compute a 32b mask of threads having the same least-significant
 * LABEL_BITS of \p label as the calling thread.
 */
template <int LABEL_BITS>
inline __device__ unsigned int MatchAny(unsigned int label)
{
    unsigned int retval = 0xffffffff;

    // Extract masks of common threads for each bit
    #pragma unroll
    for (int BIT = 0; BIT < LABEL_BITS; ++BIT)
    {
        unsigned int current_bit = 1 << BIT;
        bool bit_set = (label & current_bit) != 0;
#ifdef CUB_USE_COOPERATIVE_GROUPS
        unsigned int mask = __ballot_sync(0xffffffff, bit_set);
#else
        unsigned int mask = __ballot(bit_set);
#endif
        // If current thread's bit is not set, invert the mask
        if (!bit_set)
        {
            mask = ~mask;
        }

        // Remove peers who differ
        retval = (BIT == 0) ? mask : retval & mask;
    }

    return retval;
}

CUB_NAMESPACE_END
