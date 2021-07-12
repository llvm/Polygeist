add_clang_tool(mlir-clang
  mlir-clang.cc
  Lib/utils.cc
  Lib/pragmaHandler.cc
)

target_include_directories(mlir-clang PRIVATE
  "${LLVM_SOURCE_DIR}/../clang/include"
  "${CMAKE_BINARY_DIR}/tools/clang/include"
)

target_compile_definitions(mlir-clang PUBLIC -DLLVM_OBJ_ROOT="${LLVM_BINARY_DIR}")
target_link_libraries(mlir-clang PRIVATE
  LLVMCore
  LLVMOption
  LLVMSupport

  MLIRSCFTransforms
  MLIRPolygeist

  MLIRSupport
  MLIRIR
  MLIRAnalysis
  MLIRLLVMIR
  MLIRNVVMIR
  MLIROpenMP
  MLIRGPUOps
  MLIRTransforms
  MLIRSCFToStandard
  MLIRStandardToLLVM
  MLIRAffineTransforms
  MLIRAffineToStandard
  MLIRTargetLLVMIRImport
  MLIRPolygeistTransforms
  MLIRLLVMToLLVMIRTranslation
  MLIRSCFToOpenMP
  MLIROpenMPToLLVM
  MLIROpenMPToLLVMIRTranslation

  clangAST
  clangBasic
  clangCodeGen
  clangDriver
  clangFrontend
  clangFrontendTool
  clangLex
  clangSerialization
)
add_dependencies(mlir-clang MLIRPolygeistOpsIncGen MLIRPolygeistPassIncGen)
add_subdirectory(Test)
