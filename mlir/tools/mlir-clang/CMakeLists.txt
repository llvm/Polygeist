# Manually set directories to include.
set(CLANG_INCLUDE_DIR ${LLVM_MAIN_SRC_DIR}/../clang/include)
set(CLANG_LIB_DIR ${LLVM_MAIN_SRC_DIR}/../clang/lib)
set(CLANG_TABLEGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/tools/clang/include)

include_directories(SYSTEM ${CLANG_INCLUDE_DIR})
include_directories(SYSTEM ${CLANG_TABLEGEN_OUTPUT_DIR})
include_directories(SYSTEM ${CLANG_LIB_DIR})

# MLIRClang library
add_subdirectory(Lib)

add_clang_tool(mlir-clang
  mlir-clang.cc
)

# MLIRTransformsPassIncGen
target_compile_definitions(mlir-clang PUBLIC -DLLVM_OBJ_ROOT="${CMAKE_BINARY_DIR}")
target_link_libraries(mlir-clang PRIVATE
  LLVMCore
  LLVMOption
  LLVMSupport

  MLIRSCFTransforms

  MLIRSupport
  MLIRIR
  MLIRAnalysis
  MLIRLLVMIR
  MLIRNVVMIR
  MLIRGPU
  MLIRTransforms
  MLIRSCFToStandard
  MLIRStandardToLLVM
  MLIRAffineTransforms
  MLIRAffineToStandard
  MLIRTargetLLVMIRImport

  clangAST
  clangBasic
  clangCodeGen
  clangDriver
  clangFrontend
  clangFrontendTool
  clangLex
  clangSerialization

  MLIRClang
)

add_subdirectory(Test)
