cmake_minimum_required(VERSION 3.10)

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

project(polymer LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/bin")
set(LLVM_LIBRARY_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/lib")
set(MLIR_BINARY_DIR "${CMAKE_BINARY_DIR}")

# Define the default arguments to use with 'lit', and an option for the user to
# override.
set(LIT_ARGS_DEFAULT "-sv")
if (MSVC_IDE OR XCODE)
  set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
endif()
set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}" CACHE STRING "Default options for lit")

# ------------------------------------------------- Dependencies
# OpenScop
# We should prevent the main CMakeLists.txt file in openscop to change the 
# BUILD_SHARED_LIBS flag. 
set(BUILD_SHARED_LIBS FALSE)
# This should add a new library osl.
add_subdirectory(openscop)
include_directories("${CMAKE_SOURCE_DIR}/openscop/include")
include_directories("${CMAKE_BINARY_DIR}/openscop/include")

# ------------------------------------------------- This project
set(POLYMER_MAIN_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include" ) # --src-root
set(POLYMER_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/include" ) # --includedir

set(POLYMER_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(POLYMER_BINARY_DIR "${CMAKE_BINARY_DIR}/bin")
set(POLYMER_TOOLS_DIR  "${CMAKE_BINARY_DIR}/bin")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories("${LLVM_INCLUDE_DIRS}")
include_directories("${MLIR_INCLUDE_DIRS}")
include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("${PROJECT_BINARY_DIR}/include")

add_subdirectory(include/polymer)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(test)
