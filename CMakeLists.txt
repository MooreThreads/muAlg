# 3.15 is the minimum.
# 3.17 for NVC++.
# 3.18.3 for C++17 + CUDA.
cmake_minimum_required(VERSION 3.15)

# Remove this when we use the new CUDA_ARCHITECTURES properties.
if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
  cmake_policy(SET CMP0104 OLD)
endif()

# CXX is only needed for AppendOptionIfAvailable.
project(CUB NONE)
# MUSA Configuration
list(APPEND CMAKE_MODULE_PATH /usr/local/musa/cmake)
find_package(MUSA REQUIRED)

# MUSA Architecture Configuration
set(MUSA_SUPPORTED_ARCHS mp_22 mp_31)
set(MUSA_ARCH_LIST "mp_31" CACHE STRING "MUSA target architectures (semicolon-separated: 'mp_22', 'mp_31', or 'mp_22;mp_31')")

# Validate and build flags
set(MUSA_ARCH_FLAGS "")
foreach(arch IN LISTS MUSA_ARCH_LIST)
    if(NOT arch IN_LIST MUSA_SUPPORTED_ARCHS)
        message(FATAL_ERROR "Unsupported MUSA architecture: ${arch}. Supported: ${MUSA_SUPPORTED_ARCHS}")
    endif()
    string(APPEND MUSA_ARCH_FLAGS "--offload-arch=${arch} ")
endforeach()
string(STRIP "${MUSA_ARCH_FLAGS}" MUSA_ARCH_FLAGS)
set(MUSA_MCC_FLAGS ${MUSA_ARCH_FLAGS} -std=c++17)

message(STATUS "MUSA Architectures: ${MUSA_ARCH_LIST}")
message(STATUS "MUSA Flags: ${MUSA_MCC_FLAGS}")


# Determine whether CUB is the top-level project or included into
# another project via add_subdirectory().
if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_LIST_DIR}")
  set(CUB_TOPLEVEL_PROJECT ON)
else()
  set(CUB_TOPLEVEL_PROJECT OFF)
endif()

# This must be done before any languages are enabled:
if (CUB_TOPLEVEL_PROJECT)
  include(cmake/CubCompilerHacks.cmake)
endif()

# This must appear after our Compiler Hacks or else CMake will delete the cache
# and reconfigure from scratch.
# This must also appear before the installation rules, as it is required by the
# GNUInstallDirs CMake module.
enable_language(CXX)

# Require C++17 for std::invoke_result_t support
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable C++17 dialect for MUSA (use std::invoke_result_t instead of std::result_of)
set(CUB_ENABLE_DIALECT_CPP17 ON CACHE BOOL "Enable C++17 dialect" FORCE)
set(CUB_ENABLE_DIALECT_CPP14 OFF CACHE BOOL "Disable C++14 dialect" FORCE)

# Thrust has its own copy of CUB install rules to handle packaging usecases
# where we want to install CUB headers but aren't actually building anything.
# In these cases the add_subdirectory(dependencies/cub) line in Thrust won't get
# called so we can't rely on CUB providing its own rules.
if (NOT CUB_IN_THRUST)
  option(CUB_ENABLE_INSTALL_RULES "Enable installation of CUB" ${CUB_TOPLEVEL_PROJECT})
  if (CUB_ENABLE_INSTALL_RULES)
    include(cmake/CubInstallRules.cmake)
  endif()
endif()

# Support adding CUB to a parent project via add_subdirectory.
# See examples/cmake/add_subdir/CMakeLists.txt for details.
if (NOT CUB_TOPLEVEL_PROJECT AND NOT CUB_IN_THRUST)
  include(cmake/CubAddSubdir.cmake)
  return()
endif()

option(CUB_ENABLE_HEADER_TESTING "Test that all public headers compile." OFF)
option(CUB_ENABLE_TESTING "Build CUB testing suite." ON)
option(CUB_ENABLE_EXAMPLES "Build CUB examples." ON)
# This is needed for NVCXX QA, which requires a static set of executable names.
# Only a single dialect may be enabled when this is off.
option(CUB_ENABLE_CPP_DIALECT_IN_NAMES
  "Include C++ dialect information in target/object/etc names."
  ON
)
mark_as_advanced(CUB_ENABLE_CPP_DIALECT_IN_NAMES)

# This option is only used when CUB is built stand-alone; otherwise the Thrust
# option has the same effect.
if (NOT CUB_IN_THRUST)
  option(CUB_IGNORE_DEPRECATED_API
    "Suppress warnings about deprecated Thrust/CUB API."
    OFF
  )
endif()

# Check if we're actually building anything before continuing. If not, no need
# to search for deps, etc. This is a common approach for packagers that just
# need the install rules. See GH issue NVIDIA/thrust#1211.
if (NOT (CUB_ENABLE_HEADER_TESTING OR
         CUB_ENABLE_TESTING OR
         CUB_ENABLE_EXAMPLES))
  return()
endif()

include(cmake/AppendOptionIfAvailable.cmake)
include(cmake/CubBuildCompilerTargets.cmake)
include(cmake/CubBuildTargetList.cmake)
include(cmake/CubMusaConfig.cmake)
include(cmake/CubUtilities.cmake)

if ("" STREQUAL "${CMAKE_BUILD_TYPE}")
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the type of build." FORCE)

  set_property(
    CACHE CMAKE_BUILD_TYPE
    PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel
  )
endif ()

set(CMAKE_CXX_EXTENSIONS OFF)

# Where to put build outputs. Use CMAKE_BINARY_DIR so they'll show up alongside
# Thrust targets when building as part of Thrust.
set(CUB_LIBRARY_OUTPUT_DIR "${CMAKE_BINARY_DIR}/lib")
set(CUB_EXECUTABLE_OUTPUT_DIR "${CMAKE_BINARY_DIR}/bin")

cub_build_target_list()

if (CUB_ENABLE_HEADER_TESTING)
  include(cmake/CubHeaderTesting.cmake)
endif()

# Both testing and examples use ctest
if (CUB_ENABLE_TESTING OR CUB_ENABLE_EXAMPLES)
  include(CTest)
  enable_testing()
endif()

if (CUB_ENABLE_TESTING)
  add_subdirectory(test)
endif()

if (CUB_ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()
