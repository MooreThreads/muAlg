# Parse version information from version.h:
unset(_CUB_VERSION_INCLUDE_DIR CACHE) # Clear old result to force search

# Find CMAKE_INSTALL_INCLUDEDIR=include directory"
set(from_install_prefix "lib/cmake/cub")

# Transform to a list of directories, replace each directoy with "../"
# and convert back to a string
string(REGEX REPLACE "/" ";" from_install_prefix "${from_install_prefix}")
list(TRANSFORM from_install_prefix REPLACE ".+" "../")
list(JOIN from_install_prefix "" from_install_prefix)

# First try the install prefix path
find_path(_CUB_VERSION_INCLUDE_DIR cub/version.cuh
  NO_DEFAULT_PATH # Only search explicit paths below:
  PATHS
    "${CMAKE_CURRENT_LIST_DIR}/${from_install_prefix}/include"
    "${CMAKE_CURRENT_LIST_DIR}/../../"  # For building from source tree
)

# If not found, try relative path from source tree
if(NOT _CUB_VERSION_INCLUDE_DIR)
  if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/../../cub/version.cuh")
    set(_CUB_VERSION_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../")
  endif()
endif()

set_property(CACHE _CUB_VERSION_INCLUDE_DIR PROPERTY TYPE INTERNAL)
