get_filename_component(THIS_DIR ${CMAKE_CURRENT_LIST_DIR} ABSOLUTE)

# add_version_query(<status>)
# status - version status, e.g.: "rtm" "pre-release" "release" "develop"
# ARGV[1] : version_string
# ARGV[2] : module_name
function (add_version_query status)
  set(MODULE_VERSION_STATUS ${status})
  if(ARGV1)
    set(MODULE_VERSION_STRING "${ARGV1}")
    string(REGEX MATCHALL "[0-9]+" VERSION_PARTS "${MODULE_VERSION_STRING}")
    list(LENGTH VERSION_PARTS VERSION_PARTS_NUM)
    if (VERSION_PARTS_NUM GREATER_EQUAL 1)
      list(GET VERSION_PARTS 0 MODULE_VERSION_MAJOR)
    endif ()
    if (VERSION_PARTS_NUM GREATER_EQUAL 2)
      list(GET VERSION_PARTS 1 MODULE_VERSION_MINOR)
    endif ()
    if (VERSION_PARTS_NUM GREATER_EQUAL 3)
      list(GET VERSION_PARTS 2 MODULE_VERSION_PATCH)
    endif ()
    if (VERSION_PARTS_NUM GREATER_EQUAL 4)
      list(GET VERSION_PARTS 3 MODULE_VERSION_TWEAK)
    endif ()
  else()
    set(MODULE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
    set(MODULE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
    set(MODULE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
    set(MODULE_VERSION_TWEAK ${PROJECT_VERSION_TWEAK})
    if(NOT MODULE_VERSION_MAJOR)
      set(MODULE_VERSION_MAJOR "0")
    endif()
    if(NOT MODULE_VERSION_MINOR)
      set(MODULE_VERSION_MINOR "0")
    endif()
    if(NOT MODULE_VERSION_PATCH)
      set(MODULE_VERSION_PATCH "0")
    endif()
    if(NOT MODULE_VERSION_TWEAK)
      set(MODULE_VERSION_TWEAK "0")
    endif()
    set(MODULE_VERSION_STRING "${MODULE_VERSION_MAJOR}.${MODULE_VERSION_MINOR}.${MODULE_VERSION_PATCH}")
  endif()

  if(ARGV2)
    set(MODULE_NAME "${ARGV2}")
  else()
    set(MODULE_NAME ${PROJECT_NAME})
  endif()

  set(CONFIG_TMP_DIR "${CMAKE_CURRENT_BINARY_DIR}/${MODULE_NAME}_version_config")

  string(TIMESTAMP TIMESTAMP_STRING "%Y-%m-%d %H:%M:%S")
  find_package(Git)
  message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
  message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
  message(STATUS "CMAKE_CURRENT_LIST_DIR: ${CMAKE_CURRENT_LIST_DIR}")
  message(STATUS "THIS_DIR: ${THIS_DIR}")
  message(STATUS "MODULE_NAME: ${MODULE_NAME}")
  message(STATUS "MODULE_VERSION_MAJOR: ${MODULE_VERSION_MAJOR}")
  message(STATUS "MODULE_VERSION_MINOR: ${MODULE_VERSION_MINOR}")
  message(STATUS "MODULE_VERSION_PATCH: ${MODULE_VERSION_PATCH}")
  message(STATUS "MODULE_VERSION_STRING: ${MODULE_VERSION_STRING}")
  if (Git_FOUND)
    execute_process(COMMAND
      "${GIT_EXECUTABLE}" rev-parse --is-inside-work-tree
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      OUTPUT_VARIABLE IS_INSIDE_WORK_TREE
      ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    if (IS_INSIDE_WORK_TREE)
      execute_process(COMMAND
        "${GIT_EXECUTABLE}" rev-parse --abbrev-ref HEAD
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE GIT_BRANCH
        ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
      message(STATUS "GIT_BRANCH: ${GIT_BRANCH}")
      execute_process(COMMAND
        ${GIT_EXECUTABLE} describe --tags --exact-match HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_TAG
        ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
      if (NOT GIT_TAG)
        set(GIT_TAG "No tag")
      endif()
      message(STATUS "GIT_TAG: ${GIT_TAG}")
      execute_process(COMMAND
        # "${GIT_EXECUTABLE}" rev-parse --short HEAD
        "${GIT_EXECUTABLE}" rev-parse HEAD
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE GIT_COMMIT_ID
        ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
      message(STATUS "GIT_COMMIT_ID: ${GIT_COMMIT_ID}")
      execute_process(COMMAND
        "${GIT_EXECUTABLE}" log -1 --format=%cd --date=iso
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE GIT_COMMIT_DATE
        ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
      message(STATUS "GIT_COMMIT_DATE: ${GIT_COMMIT_DATE}")
    else ()
      set(GIT_BRANCH "none")
      set(GIT_TAG "No tag")
      set(GIT_COMMIT_ID "not a git repository")
      set(GIT_COMMIT_DATE ${TIMESTAMP_STRING})
    endif ()
  else ()
    set(GIT_BRANCH "none")
    set(GIT_TAG "No tag")
    set(GIT_COMMIT_ID "not a git repository")
    set(GIT_COMMIT_DATE ${TIMESTAMP_STRING})
  endif ()

  if (MUSA_ARCHS)
    set(MODULE_BUILD_ARCHS "${MUSA_ARCHS}")
  else ()
    set(MODULE_BUILD_ARCHS "Not Set")
  endif ()
  message(STATUS "MODULE_BUILD_ARCHS: ${MODULE_BUILD_ARCHS}")

  set(SOURCE_FILENAME ${MODULE_NAME}_version.cpp)
  configure_file(${THIS_DIR}/version_query.cpp ${CONFIG_TMP_DIR}/${SOURCE_FILENAME} @ONLY)
  configure_file(${THIS_DIR}/Json.hpp ${CONFIG_TMP_DIR}/Json.hpp @ONLY)

  set(target_name ${MODULE_NAME}_version)
  if(NOT TARGET ${target_name})
    add_executable(${target_name} ${CONFIG_TMP_DIR}/${SOURCE_FILENAME})
    target_link_options(${target_name} PRIVATE "LINKER:--strip-all")
    if (MSVC)
      set_target_properties(${target_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY bin
        COMPILE_OPTIONS ""
        )
    else ()
      set_target_properties(${target_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY bin
        COMPILE_OPTIONS "-Wno-error"
        )
      target_compile_options(${target_name} PRIVATE -fPIE -fstack-protector-strong)
      target_link_options(${target_name} PRIVATE -Wl,-z,relro,-z,now -pie)
    endif()
    message(STATUS VERSION_WHITOUT_GIT: ${VERSION_WHITOUT_GIT})
    if(VERSION_WHITOUT_GIT EQUAL 1)
      target_compile_definitions(${target_name} PRIVATE VERSION_WHITOUT_GIT)
    endif()
    install(TARGETS ${target_name})
  endif()

endfunction()
