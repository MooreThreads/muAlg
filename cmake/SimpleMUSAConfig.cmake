# SimpleMUSAConfig.cmake
# Simplified MUSA configuration without find_package(MUSA)
# Uses CMake native commands to find MUSA toolkit

# Find mcc compiler
find_program(MCC_EXECUTABLE
    NAMES mcc
    PATHS /usr/local/musa/bin
    ENV MUSA_PATH
    ENV MUSA_BIN_PATH
    PATH_SUFFIXES bin
)

if(NOT MCC_EXECUTABLE)
    message(FATAL_ERROR "MUSA compiler (mcc) not found. Please install MUSA toolkit or set MUSA_PATH environment variable.")
endif()

message(STATUS "Found MUSA compiler: ${MCC_EXECUTABLE}")

# Find MUSA include directory
find_path(MUSA_INCLUDE_DIR
    NAMES musa_runtime.h
    PATHS /usr/local/musa/include
    ENV MUSA_PATH
    ENV MUSA_INC_PATH
    PATH_SUFFIXES include
)

if(NOT MUSA_INCLUDE_DIR)
    message(FATAL_ERROR "MUSA include directory not found.")
endif()

message(STATUS "Found MUSA include: ${MUSA_INCLUDE_DIR}")

# Find MUSA runtime library
find_library(MUSA_LIBRARY
    NAMES musart musa
    PATHS /usr/local/musa/lib
    ENV MUSA_PATH
    ENV MUSA_LIB_PATH
    PATH_SUFFIXES lib lib64
)

if(NOT MUSA_LIBRARY)
    message(FATAL_ERROR "MUSA runtime library not found.")
endif()

message(STATUS "Found MUSA library: ${MUSA_LIBRARY}")

# Set MUSA toolkit root directory
get_filename_component(MUSA_TOOLKIT_ROOT_DIR "${MCC_EXECUTABLE}" DIRECTORY)
get_filename_component(MUSA_TOOLKIT_ROOT_DIR "${MUSA_TOOLKIT_ROOT_DIR}" DIRECTORY)

# Set MUSA link libraries keyword (can be empty or PRIVATE/PUBLIC/INTERFACE)
set(MUSA_LINK_LIBRARIES_KEYWORD "")

# Function to add MUSA include directories once per directory
function(musa_add_musa_include_once)
    get_directory_property(_include_directories INCLUDE_DIRECTORIES)
    set(_add TRUE)
    if(_include_directories)
        foreach(dir ${_include_directories})
            if("${dir}" STREQUAL "${MUSA_INCLUDE_DIR}")
                set(_add FALSE)
            endif()
        endforeach()
    endif()
    if(_add)
        include_directories(${MUSA_INCLUDE_DIR})
    endif()
endfunction()

# Function to compile .cu files using custom command
function(musa_add_executable target_name)
    # Separate sources and options
    set(sources)
    set(options)
    set(cmake_options)
    set(found_options FALSE)

    foreach(arg ${ARGN})
        if("x${arg}" STREQUAL "xOPTIONS")
            set(found_options TRUE)
        elseif("x${arg}" STREQUAL "xWIN32" OR
               "x${arg}" STREQUAL "xMACOSX_BUNDLE" OR
               "x${arg}" STREQUAL "xEXCLUDE_FROM_ALL")
            list(APPEND cmake_options ${arg})
        else()
            if(found_options)
                list(APPEND options ${arg})
            else()
                list(APPEND sources ${arg})
            endif()
        endif()
    endforeach()

    # Add MUSA include directories
    musa_add_musa_include_once()

    # Process each source file
    set(generated_objects)
    foreach(src ${sources})
        get_filename_component(src_ext ${src} EXT)
        get_filename_component(src_name ${src} NAME_WE)

        # Only process .cu files
        if("${src_ext}" STREQUAL ".cu")
            set(obj_file "${CMAKE_CURRENT_BINARY_DIR}/${src_name}.o")

            # Add custom command to compile .cu to .o
            add_custom_command(
                OUTPUT ${obj_file}
                COMMAND ${MCC_EXECUTABLE}
                    -x musa
                    -c ${CMAKE_CURRENT_SOURCE_DIR}/${src}
                    -o ${obj_file}
                    -I${MUSA_INCLUDE_DIR}
                    ${MUSA_MCC_FLAGS}
                    ${options}
                DEPENDS ${src}
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                COMMENT "Building MUSA object ${src_name}.o from ${src}"
                VERBATIM
            )

            list(APPEND generated_objects ${obj_file})
        else()
            # Non-.cu files are passed as-is
            list(APPEND generated_objects ${src})
        endif()
    endforeach()

    # Create executable
    add_executable(${target_name} ${cmake_options} ${generated_objects})

    # Link MUSA runtime
    target_link_libraries(${target_name} ${MUSA_LINK_LIBRARIES_KEYWORD} ${MUSA_LIBRARY})

    # Set linker language to CXX
    set_target_properties(${target_name} PROPERTIES LINKER_LANGUAGE CXX)
endfunction()