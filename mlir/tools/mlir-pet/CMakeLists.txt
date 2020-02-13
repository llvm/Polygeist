# pet project.

set(PET_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PET_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

find_package(PET REQUIRED)
find_package(ISL REQUIRED)

add_llvm_executable(mlir-pet
  mlir-pet.cc
  scop.cc
  islAst.cc
  mlirCodegen.cc
  islNodeBuilder.cc
)

message(STATUS "PET include and lib path ->")
message(STATUS ${PET_INCLUDE_PATH})
message(STATUS ${PET_LIBRARY})
message(STATUS "ISL include and lib path ->")
message(STATUS ${ISL_INCLUDE_PATH})
message(STATUS ${ISL_LIBRARY})

#debug
option(PET-TO-MLIR-VERBOSE "debug" OFF)
if (PET-TO-MLIR-VERBOSE)
  add_definitions("-DDEBUG")
endif (PET-TO-MLIR-VERBOSE)

target_include_directories(mlir-pet PRIVATE
  ${PET_INCLUDE_PATH}
  ${ISL_INCLUDE_PATH}
)

target_link_libraries(mlir-pet PRIVATE 
  ${PET_LIBRARY}
  ${ISL_LIBRARY}
  LLVMCore
  LLVMSupport
  
  MLIRSupport
  MLIRIR
  MLIRAnalysis

)

add_subdirectory(test)
