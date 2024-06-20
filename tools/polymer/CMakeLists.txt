set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
  "${CMAKE_CURRENT_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

set(POLYMER_TRANSFORM_DEPENDS
  MLIRAffineOpsIncGen
  MLIRAffinePassIncGen
  MLIRLoopLikeInterfaceIncGen
)

set(POLYMER_MAIN_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}" )
set(POLYMER_MAIN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include" )

set(POLYMER_SOURCE_DIR  "${CMAKE_CURRENT_SOURCE_DIR}")
set(POLYMER_BINARY_DIR  "${CMAKE_CURRENT_BINARY_DIR}/bin")
set(POLYMER_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/include" ) 
set(POLYMER_TOOLS_DIR   ${CMAKE_BINARY_DIR}/bin)

# ----------------------------- Customize CMake

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ----------------------------- Dependencies
# Find whether the GMP package exists.
find_package(GMP REQUIRED)

if(POLYGEIST_POLYMER_ENABLE_PLUTO)
  # build pluto, openscop, cloog
  execute_process(
    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/build_polymer_deps.sh" "${POLYGEIST_PLUTO_DIR}"
    COMMAND_ERROR_IS_FATAL ANY
  )
  # configure the cmake variables for pluto
  include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/AddPluto.cmake")
  include_directories("${PLUTO_INCLUDE_DIR}")
  message(STATUS "Will build and use Polymer-bundled Pluto")

  include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/OpenScop.cmake")
  include_directories("${OSL_INCLUDE_DIR}")

  include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CLooG.cmake")
  include_directories("${CLOOG_INCLUDE_DIR}")

  set(POLYMER_AVAIL_TARGETS
    PolymerTargetOpenScop
  )
elseif(POLYGEIST_POLYMER_ENABLE_ISL)
  include_directories(${MLIR_MAIN_SRC_DIR}/../polly/lib/External/isl/include)
  include_directories(${MLIR_MAIN_SRC_DIR}/../polly/include)
  include_directories(${MLIR_MAIN_SRC_DIR}/../polly/lib/External/isl/include)
  include_directories(${MLIR_MAIN_SRC_DIR}/../polly/lib/External/isl/include)
  include_directories(${LLVM_BINARY_DIR}/tools/polly/lib/External/isl/include)
  set(POLYMER_AVAIL_TARGETS
    PolymerTargetISL
  )
else()
  message(FATAL_ERROR "Need one of POLYGEIST_POLYMER_ENABLE_PLUTO or POLYGEIST_POLYMER_ENABLE_ISL")
endif()

# ------------------------------------------------- This project
include_directories("${POLYMER_MAIN_INCLUDE_DIR}")
include_directories("${POLYMER_INCLUDE_DIR}")

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()

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
