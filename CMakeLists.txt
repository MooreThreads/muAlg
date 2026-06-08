# 3.15 is the minimum.
# 3.17 for NVC++.
# 3.18.3 for C++17 + CUDA.
cmake_minimum_required(VERSION 3.15)

# Remove this when we use the new CUDA_ARCHITECTURES properties.
if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
  cmake_policy(SET CMP0104 OLD)
endif()

set( CMAKE_INSTALL_PREFIX "/usr/local/musa" CACHE PATH
    "Install path prefix, prepended onto install directories" )
set ( VERSION_STRING "1.17.2" )
# CXX is selected after SimpleMUSA configures the mcc compiler.
project(CUB VERSION ${VERSION_STRING} LANGUAGES NONE)
# MUSA Configuration
include(cmake/SimpleMUSA.cmake)

# MUSA Architecture Configuration
set(MUSA_SUPPORTED_ARCHS mp_21 mp_22 mp_31)
set(MUSA_ARCH_LIST "mp_31" CACHE STRING "MUSA target architectures (semicolon-separated: 'mp_21', 'mp_22', 'mp_31', or 'mp_21;mp_22;mp_31')")

# Validate and build flags
set(MUSA_ARCH_FLAGS "")
foreach(arch IN LISTS MUSA_ARCH_LIST)
    if(NOT arch IN_LIST MUSA_SUPPORTED_ARCHS)
        message(FATAL_ERROR "Unsupported MUSA architecture: ${arch}. Supported: ${MUSA_SUPPORTED_ARCHS}")
    endif()
    string(APPEND MUSA_ARCH_FLAGS "--offload-arch=${arch} ")
endforeach()
string(STRIP "${MUSA_ARCH_FLAGS}" MUSA_ARCH_FLAGS)
set(CUB_MUSA_CPP_STANDARD "17" CACHE STRING "C++ standard used for MUSA builds")
set_property(CACHE CUB_MUSA_CPP_STANDARD PROPERTY STRINGS 11 14 17 20)
set(MUSA_MCC_FLAGS ${MUSA_ARCH_FLAGS} -std=c++${CUB_MUSA_CPP_STANDARD})

# Determine CUB_MUSA_ARCH for host code Policy selection
# When compiling for a single architecture, use that architecture's version number
# mp_21 -> 210, mp_22 -> 220, mp_31 -> 310
list(LENGTH MUSA_ARCH_LIST MUSA_ARCH_COUNT)
if(MUSA_ARCH_COUNT EQUAL 1)
    list(GET MUSA_ARCH_LIST 0 SINGLE_ARCH)
    if(SINGLE_ARCH STREQUAL "mp_21")
        set(CUB_MUSA_ARCH_VALUE 210)
    elseif(SINGLE_ARCH STREQUAL "mp_22")
        set(CUB_MUSA_ARCH_VALUE 220)
    elseif(SINGLE_ARCH STREQUAL "mp_31")
        set(CUB_MUSA_ARCH_VALUE 310)
    else()
        set(CUB_MUSA_ARCH_VALUE 210)  # Default fallback to lowest
    endif()
else()
    # Multi-arch: use the highest architecture for host code
    set(CUB_MUSA_ARCH_VALUE 310)
endif()

message(STATUS "MUSA Architectures: ${MUSA_ARCH_LIST}")
message(STATUS "MUSA Flags: ${MUSA_MCC_FLAGS}")
message(STATUS "CUB_MUSA_ARCH: ${CUB_MUSA_ARCH_VALUE}")

# Add CUB_MUSA_ARCH definition for host code Policy selection
add_compile_definitions(CUB_MUSA_ARCH=${CUB_MUSA_ARCH_VALUE})

# Add MUSA_ARCH_LIST definition for namespace magic (must match CUB_MUSA_ARCH)
add_compile_definitions(MUSA_ARCH_LIST=${CUB_MUSA_ARCH_VALUE})


# Determine whether CUB is the top-level project or included into
# another project via add_subdirectory().
if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_LIST_DIR}")
  set(CUB_TOPLEVEL_PROJECT ON)
else()
  set(CUB_TOPLEVEL_PROJECT OFF)
endif()

# This must be done before any languages are enabled:
if (CUB_TOPLEVEL_PROJECT)
  include(./module_version/generate_version_query.cmake)
  add_version_query(${VERSION_STRING} ${VERSION_STRING} muAlg)
  include(cmake/CubCompilerHacks.cmake)
endif()

# This must appear after our Compiler Hacks or else CMake will delete the cache
# and reconfigure from scratch.
# This must also appear before the installation rules, as it is required by the
# GNUInstallDirs CMake module.
enable_language(CXX)

# Set default installation prefix for MUSA
if (NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local/musa" CACHE PATH "Installation prefix" FORCE)
endif()

# Require the configured C++ dialect for generated CUB targets.
set(CMAKE_CXX_STANDARD ${CUB_MUSA_CPP_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable a single MUSA dialect by default.
foreach(dialect IN ITEMS 11 14 17 20)
  set(enable_dialect OFF)
  if ("${dialect}" EQUAL "${CUB_MUSA_CPP_STANDARD}")
    set(enable_dialect ON)
  endif()
  set(CUB_ENABLE_DIALECT_CPP${dialect}
    ${enable_dialect}
    CACHE BOOL "Enable C++${dialect} dialect"
    FORCE
  )
endforeach()

# Thrust has its own copy of CUB install rules to handle packaging usecases
# where we want to install CUB headers but aren't actually building anything.
# In these cases the add_subdirectory(dependencies/cub) line in Thrust won't get
# called so we can't rely on CUB providing its own rules.
if (NOT CUB_IN_THRUST)
  option(CUB_ENABLE_INSTALL_RULES "Enable installation of CUB" ON)
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
option(CUB_ENABLE_TESTING "Build CUB testing suite." OFF)
option(CUB_ENABLE_EXAMPLES "Build CUB examples." OFF)
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
