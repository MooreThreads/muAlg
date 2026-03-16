#
# MUSA Architecture options for CUB
#

# MUSA supported architectures: mp_22 (S4000) and mp_31 (S5000)
set(all_archs 22 31)
set(arch_message "CUB: Explicitly enabled MUSA architectures:")

# Thrust sets up the architecture flags already. Just reuse them if possible.
if (CUB_IN_THRUST)
  # Configure to use all flags from thrust:
  set(CMAKE_MUSA_FLAGS "${THRUST_MUSA_FLAGS_BASE}")

  # Update the enabled architectures list from thrust
  foreach (arch IN LISTS all_archs)
    if (THRUST_ENABLE_COMPUTE_${arch})
      set(CUB_ENABLE_COMPUTE_${arch} True)
      string(APPEND arch_message " mp_${arch}")
    else()
      set(CUB_ENABLE_COMPUTE_${arch} False)
    endif()
  endforeach()

  # Otherwise create cache options and build the flags ourselves:
else() # NOT CUB_IN_THRUST

  # Find the highest arch:
  list(SORT all_archs)
  list(LENGTH all_archs max_idx)
  math(EXPR max_idx "${max_idx} - 1")
  list(GET all_archs ${max_idx} highest_arch)

  option(CUB_DISABLE_ARCH_BY_DEFAULT
    "If ON, then all compute architectures are disabled on the initial CMake run."
    OFF
  )

  set(option_init ON)
  if (CUB_DISABLE_ARCH_BY_DEFAULT)
    set(option_init OFF)
  endif()

  set(arch_flags)
  set(num_archs_enabled 0)
  foreach (arch IN LISTS all_archs)
    option(CUB_ENABLE_COMPUTE_${arch}
      "Enable code generation for mp_${arch}."
      ${option_init}
    )

    if (CUB_ENABLE_COMPUTE_${arch})
      math(EXPR num_archs_enabled "${num_archs_enabled} + 1")
      # MUSA uses --offload-arch=mp_XX format
      string(APPEND arch_flags " --offload-arch=mp_${arch}")
      string(APPEND arch_message " mp_${arch}")
    endif()
  endforeach()

  # Append to MUSA flags
  string(APPEND CMAKE_MUSA_FLAGS "${arch_flags}")
endif()

message(STATUS ${arch_message})

#
# RDC options:
#

# RDC is off by default for MUSA
option(CUB_ENABLE_TESTS_WITH_RDC
  "Build all CUB tests with RDC; tests that require RDC are not affected by this option."
  OFF
)

option(CUB_ENABLE_EXAMPLES_WITH_RDC
  "Build all CUB examples with RDC; examples which require RDC are not affected by this option."
  OFF
)