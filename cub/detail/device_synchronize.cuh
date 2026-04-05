/*
 *  Copyright 2021 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <cub/detail/exec_check_disable.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_namespace.cuh>

#include <musa_runtime_api.h>

CUB_NAMESPACE_BEGIN

namespace detail
{

/**
 * Call `cudaDeviceSynchronize()` using the proper API for the current CUB and
 * MUSA configuration.
 */
CUB_EXEC_CHECK_DISABLE
CUB_RUNTIME_FUNCTION inline musaError_t device_synchronize()
{
  musaError_t result = musaErrorUnknown;

  if (CUB_IS_HOST_CODE)
  {
#if CUB_INCLUDE_HOST_CODE
    result = musaDeviceSynchronize();
#endif
  }
  else
  {
    // Device code with the MUSA runtime.
#if defined(CUB_INCLUDE_DEVICE_CODE) && defined(CUB_RUNTIME_ENABLED)
    result = musaDeviceSynchronize();
#else // Device code without the MUSA runtime.
    // Device side MUSA API calls are not supported in this configuration.
    result = musaErrorInvalidConfiguration;
#endif
  }

  return result;
}

} // namespace detail

CUB_NAMESPACE_END
