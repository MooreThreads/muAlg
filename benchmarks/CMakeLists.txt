# CUB Benchmarks for MUSA
# Based on NVIDIA CUB benchmarks, adapted for MUSA platform

cmake_minimum_required(VERSION 3.15)
project(CUBBenchmarks)

# MUSA Configuration
list(APPEND CMAKE_MODULE_PATH /usr/local/musa/cmake)
find_package(MUSA REQUIRED)

# muRAND Configuration
find_library(MURAND_LIBRARY murand HINTS /usr/local/musa/lib)
if(NOT MURAND_LIBRARY)
  message(FATAL_ERROR "muRAND library not found. Please install muRAND.")
endif()
message(STATUS "Found muRAND: ${MURAND_LIBRARY}")

# Build type must be Release for accurate benchmarking
if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
  message(WARNING "CUB benchmarks should be built in Release mode for accurate results.")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# MUSA Architecture Configuration
set(MUSA_SUPPORTED_ARCHS mp_22 mp_31)
set(MUSA_ARCH_LIST "mp_31" CACHE STRING "MUSA target architectures")

# Validate and build flags
set(MUSA_ARCH_FLAGS "")
foreach(arch IN LISTS MUSA_ARCH_LIST)
  if(NOT arch IN_LIST MUSA_SUPPORTED_ARCHS)
    message(FATAL_ERROR "Unsupported MUSA architecture: ${arch}. Supported: ${MUSA_SUPPORTED_ARCHS}")
  endif()
  string(APPEND MUSA_ARCH_FLAGS "--offload-arch=${arch} ")
endforeach()
string(STRIP "${MUSA_ARCH_FLAGS}" MUSA_ARCH_FLAGS)
set(MUSA_MCC_FLAGS ${MUSA_ARCH_FLAGS} -std=c++17 -O3)

message(STATUS "Building CUB benchmarks for MUSA")
message(STATUS "MUSA Architectures: ${MUSA_ARCH_LIST}")

# Set output directories
set(CUB_EXECUTABLE_OUTPUT_DIR "${CMAKE_BINARY_DIR}/bin")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)  # cub/ parent
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/musa_bench)
include_directories(/usr/local/musa/include)  # For Thrust headers

#=============================================================================
# musa_bench library
#=============================================================================

add_library(musa_bench INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/musa_bench/musa_bench.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/musa_bench/generator.cuh
)
target_include_directories(musa_bench INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/..
)
target_link_libraries(musa_bench INTERFACE ${MURAND_LIBRARY})

#=============================================================================
# Benchmark discovery and registration
#=============================================================================

set(benches_meta_target cub.all.benches)
add_custom_target(${benches_meta_target})

# Function to get all benchmark subdirectories
function(get_recursive_subdirs subdirs)
  set(dirs)
  file(GLOB_RECURSE contents
    CONFIGURE_DEPENDS
    LIST_DIRECTORIES ON
    "${CMAKE_CURRENT_LIST_DIR}/bench/*"
  )

  foreach(test_dir IN LISTS contents)
    if(IS_DIRECTORY "${test_dir}")
      list(APPEND dirs "${test_dir}")
    endif()
  endforeach()

  set(${subdirs} "${dirs}" PARENT_SCOPE)
endfunction()

# Function to add a single benchmark
function(add_bench target_name bench_name bench_src)
  set(bench_target ${bench_name})
  set(${target_name} ${bench_target} PARENT_SCOPE)

  musa_add_executable(${bench_target} "${bench_src}")
  target_link_libraries(${bench_target} musa_bench)
  target_include_directories(${bench_target} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/musa_bench
  )
  set_target_properties(${bench_target}
    PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CUB_EXECUTABLE_OUTPUT_DIR}"
  )
endfunction()

# Function to add benchmarks from a directory
function(add_bench_dir bench_dir)
  file(GLOB bench_srcs CONFIGURE_DEPENDS "${bench_dir}/*.cu")
  file(RELATIVE_PATH bench_prefix "${CMAKE_CURRENT_SOURCE_DIR}" "${bench_dir}")
  file(TO_CMAKE_PATH "${bench_prefix}" bench_prefix)
  string(REPLACE "/" "." bench_prefix "${bench_prefix}")

  foreach(bench_src IN LISTS bench_srcs)
    get_filename_component(bench_name "${bench_src}" NAME_WLE)
    string(PREPEND bench_name "cub.${bench_prefix}.")

    # Skip files that end with .cuh (header files)
    if(NOT bench_src MATCHES "\\.cuh$")
      # Add base benchmark (without tuning)
      set(base_bench_name "${bench_name}.base")
      add_bench(base_bench_target ${base_bench_name} "${bench_src}")
      add_dependencies(${benches_meta_target} ${base_bench_target})
      target_compile_definitions(${base_bench_target} PRIVATE TUNE_BASE=1)
    endif()
  endforeach()
endfunction()

#=============================================================================
# Discover and build all benchmarks
#=============================================================================

get_recursive_subdirs(subdirs)

foreach(subdir IN LISTS subdirs)
  add_bench_dir("${subdir}")
endforeach()

#=============================================================================
# Summary
#=============================================================================

message(STATUS "CUB Benchmarks configured successfully")
message(STATUS "Build with: cmake --build . --target cub.all.benches")
message(STATUS "Run benchmarks from: ${CUB_EXECUTABLE_OUTPUT_DIR}/")