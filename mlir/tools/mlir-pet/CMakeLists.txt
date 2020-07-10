# pet project.

add_llvm_tool(mlir-pet
  mlir-pet.cc
  Lib/scop.cc
  Lib/islAst.cc
  Lib/mlirCodegen.cc
  Lib/islNodeBuilder.cc
)

#debug
option(PET-TO-MLIR-VERBOSE "debug" OFF)
if (PET-TO-MLIR-VERBOSE)
  add_definitions("-DDEBUG")
endif (PET-TO-MLIR-VERBOSE)

target_include_directories(mlir-pet PRIVATE
  "${LLVM_SOURCE_DIR}/../isl/isl/include"
  "${LLVM_SOURCE_DIR}/../pet/pet/include"
)

target_link_libraries(mlir-pet PRIVATE
  LLVMCore
  LLVMSupport

  MLIRSupport
  MLIRIR
  MLIRAnalysis

  PET
  ISL
)

add_subdirectory(Test)
