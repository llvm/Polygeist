# Check if Polymer is built as a standalone out-of-(llvm/mlir)-tree project.
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)

  project(polymer LANGUAGES CXX C)
  cmake_minimum_required(VERSION 3.10)

  message (STATUS "Building Polymer as a standalone project")

  # Policies requried by LLVM.
  if(POLICY CMP0068)
    cmake_policy(SET CMP0068 NEW)
    set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
  endif()

  if(POLICY CMP0075)
    cmake_policy(SET CMP0075 NEW)
  endif()

  if(POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW)
  endif()

  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_STANDARD_REQUIRED YES)

  # ----------------------------- LLVM/MLIR references
  find_package(LLVM REQUIRED CONFIG)
  find_package(MLIR REQUIRED CONFIG)

  include_directories("${LLVM_INCLUDE_DIRS}")
  include_directories("${MLIR_INCLUDE_DIRS}")

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)

  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  set(LLVM_RUNTIME_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/bin")
  set(LLVM_LIBRARY_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/lib")
  set(MLIR_BINARY_DIR "${CMAKE_BINARY_DIR}")

  # ----------------------------- Testing with LIT
  # Define the default arguments to use with 'lit', and an option for the user to
  # override.
  set(LIT_ARGS_DEFAULT "-sv")
  if (MSVC_IDE OR XCODE)
    set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
  endif()
  set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}" CACHE STRING "Default options for lit")

  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
    "${CMAKE_CURRENT_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}")
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")
endif()

set(POLYMER_MAIN_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}" )
set(POLYMER_MAIN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include" )

set(POLYMER_SOURCE_DIR  "${CMAKE_CURRENT_SOURCE_DIR}")
set(POLYMER_BINARY_DIR  "${CMAKE_CURRENT_BINARY_DIR}/bin")
set(POLYMER_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/include" ) 
set(POLYMER_TOOLS_DIR   "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/bin")

# ----------------------------- Customize CMake

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ----------------------------- Customize flags

# From MLIR/CMakeLists.txt
# Forbid implicit function declaration: this may lead to subtle bugs and we
# don't have a reason to support this.
check_c_compiler_flag("-Werror=implicit-function-declaration" C_SUPPORTS_WERROR_IMPLICIT_FUNCTION_DECLARATION)
append_if(C_SUPPORTS_WERROR_IMPLICIT_FUNCTION_DECLARATION "-Werror=implicit-function-declaration" CMAKE_C_FLAGS)


# ----------------------------- Dependencies
# Find whether the GMP package exists.
find_package(GMP REQUIRED)
# TODO: there could be a better way to do this?
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lgmp" )

# Configure and build PLUTO to ${CMAKE_BUILD_DIR}/pluto
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/AddPluto.cmake")
include_directories("${PLUTO_INCLUDE_DIR}")

# ------------------------------------------------- This project
include_directories("${POLYMER_MAIN_INCLUDE_DIR}")
include_directories("${POLYMER_INCLUDE_DIR}")

add_subdirectory(include/polymer)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(test)

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  install(DIRECTORY include/polymer 
    DESTINATION include
    FILES_MATCHING
    PATTERN "*.h"
    )
endif()
