add_clang_tool(mlir-clang
  mlir-clang.cc
)

target_include_directories(mlir-clang PRIVATE
  "${LLVM_SOURCE_DIR}/../clang/include"
  "${CMAKE_BINARY_DIR}/tools/clang/include"
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
)

add_subdirectory(Test)
