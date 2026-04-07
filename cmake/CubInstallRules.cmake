# Thrust manages its own copy of these rules. Update ThrustInstallRules.cmake
# if modifying this file.
if (CUB_IN_THRUST)
  return()
endif()

# Bring in CMAKE_INSTALL_LIBDIR
include(GNUInstallDirs)

# MUSA toolkits place CMake package files under ${prefix}/lib, not lib64.
set(cub_cmake_install_dir "lib/cmake/cub")

# CUB is a header library; no need to build anything before installing:
set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY TRUE)

install(DIRECTORY "${CUB_SOURCE_DIR}/cub"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  FILES_MATCHING
    PATTERN "*.cuh"
)
if (EXISTS "${CUB_SOURCE_DIR}/cub/statement")
  install(FILES "${CUB_SOURCE_DIR}/cub/statement"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/cub")
endif()

install(DIRECTORY "${CUB_SOURCE_DIR}/cub/cmake/"
  DESTINATION "${cub_cmake_install_dir}"
  PATTERN cub-header-search EXCLUDE
)
# Need to configure a file to store the infix specified in
# CMAKE_INSTALL_INCLUDEDIR since it can be defined by the user
set(install_location "${cub_cmake_install_dir}")
configure_file("${CUB_SOURCE_DIR}/cub/cmake/cub-header-search.cmake.in"
  "${CUB_BINARY_DIR}/cub/cmake/cub-header-search.cmake"
  @ONLY)
install(FILES "${CUB_BINARY_DIR}/cub/cmake/cub-header-search.cmake"
  DESTINATION "${install_location}")
