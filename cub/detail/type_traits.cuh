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
 * Wrappers and extensions around <type_traits> utilities.
 */

#pragma once

#include "../util_cpp_dialect.cuh"
#include "../util_namespace.cuh"

#include <type_traits>

// MUSA compiler may not have std::void_t in some configurations
#if __cplusplus < 201703L
namespace std {
template <typename... Ts>
using void_t = void;
}
#endif

CUB_NAMESPACE_BEGIN
namespace detail {

template <typename T, typename = void>
struct result_type_impl
{
  using type = void;
};

template <typename T>
struct result_type_impl<T, std::void_t<typename T::result_type>>
{
  using type = typename T::result_type;
};

template <typename T>
using result_type_t = typename result_type_impl<T>::type;

template <typename T, typename = void>
struct has_result_type : std::false_type {};

template <typename T>
struct has_result_type<T, std::void_t<typename T::result_type>> : std::true_type {};

template <typename Invokable, typename... Args>
struct invoke_result_impl
{
#if CUB_CPP_DIALECT < 2017
  using type = typename std::result_of<Invokable(Args...)>::type;
#else // 2017+
  using type = std::invoke_result_t<Invokable, Args...>;
#endif
};

template <typename Invokable, typename... Args>
using invoke_result_t = typename invoke_result_impl<Invokable, Args...>::type;

template <typename Invokable, typename InputT, bool = has_result_type<Invokable>::value>
struct adjacent_difference_output_impl
{
  using type = result_type_t<Invokable>;
};

template <typename Invokable, typename InputT>
struct adjacent_difference_output_impl<Invokable, InputT, false>
{
#if CUB_CPP_DIALECT < 2017
  using type = typename std::result_of<Invokable(InputT, InputT)>::type;
#else // 2017+
  using type = std::invoke_result_t<Invokable, InputT, InputT>;
#endif
};

template <typename Invokable, typename InputT>
using adjacent_difference_output_t = typename adjacent_difference_output_impl<Invokable, InputT>::type;


} // namespace detail
CUB_NAMESPACE_END
