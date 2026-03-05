# For every public header, build a translation unit containing `#include <header>`
# to let the compiler try to figure out warnings in that header if it is not otherwise
# included in tests, and also to verify if the headers are modular enough.
# .inl files are not globbed for, because they are not supposed to be used as public
# entrypoints.

# Meta target for all configs' header builds:
add_custom_target(cub.all.headers)

file(GLOB_RECURSE headers
  RELATIVE "${CUB_SOURCE_DIR}/cub"
  CONFIGURE_DEPENDS
  cub/*.cuh
)

set(headertest_srcs)
foreach (header IN LISTS headers)
  set(headertest_src "${CMAKE_CURRENT_BINARY_DIR}/headers/${header}.cu")
  configure_file("${CUB_SOURCE_DIR}/cmake/header_test.in" "${headertest_src}")
  list(APPEND headertest_srcs "${headertest_src}")
endforeach()

foreach(cub_target IN LISTS CUB_TARGETS)
  cub_get_target_property(config_prefix ${cub_target} PREFIX)

  set(headertest_target ${config_prefix}.headers)
  musa_add_library(${headertest_target} OBJECT ${headertest_srcs})
  target_link_libraries(${headertest_target} ${cub_target})
  # Wrap Thrust/CUB in a custom namespace to check proper use of ns macros:
  set_target_properties(${headertest_target} PROPERTIES
    COMPILE_DEFINITIONS "THRUST_WRAPPED_NAMESPACE=wrapped_thrust;CUB_WRAPPED_NAMESPACE=wrapped_cub"
  )
  cub_clone_target_properties(${headertest_target} ${cub_target})

  add_dependencies(cub.all.headers ${headertest_target})
  add_dependencies(${config_prefix}.all ${headertest_target})
endforeach()
