/******************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2024, Moore Threads Corporation.  All rights reserved.
 *
 * MUSA port of look_back_helper.cuh - simplified version without tuning.
 ******************************************************************************/

#pragma once

// For TUNE_BASE builds, this header is not needed
// For tuned builds, we use a simple delay constructor

#if !TUNE_BASE
#include <musa_bench.cuh>
#include <cub/agent/single_pass_scan_operators.cuh>

// Default delay constructor for MUSA benchmarks
// Using no_delay_constructor as the default
using delay_constructor_t = cub::detail::no_delay_constructor_t<0>;

#endif // !TUNE_BASE