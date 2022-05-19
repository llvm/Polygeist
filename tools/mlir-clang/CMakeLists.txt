set( LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Analysis
  CodeGen
  Core
  AggressiveInstCombine
  InstCombine
  Instrumentation
  MC
  IPO
  MCParser
  ObjCARCOpts
  Option
  ScalarOpts
  Support
  TransformUtils
  Vectorize
)

add_clang_executable(mlir-clang
  mlir-clang.cc
  Lib/CGStmt.cc
  "${LLVM_SOURCE_DIR}/../clang/tools/driver/cc1_main.cpp"
  "${LLVM_SOURCE_DIR}/../clang/tools/driver/cc1as_main.cpp"
  "${LLVM_SOURCE_DIR}/../clang/tools/driver/cc1gen_reproducer_main.cpp"
  Lib/pragmaHandler.cc
  Lib/AffineUtils.cc
  Lib/ValueCategory.cc
  Lib/utils.cc
  Lib/IfScope.cc
  Lib/TypeUtils.cc
  Lib/CGCall.cc 
)
install(TARGETS mlir-clang
EXPORT PolygeistTargets
RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
COMPONENT mlir-clang)

target_include_directories(mlir-clang PRIVATE
  "${LLVM_SOURCE_DIR}/../clang/include"
  "${CMAKE_BINARY_DIR}/tools/clang/include"
)

target_compile_definitions(mlir-clang PUBLIC -DLLVM_OBJ_ROOT="${LLVM_BINARY_DIR}")
target_link_libraries(mlir-clang PRIVATE
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
  MLIRSCFToControlFlow
  MLIRFuncToLLVM
  MLIRAffineTransforms
  MLIRAffineToStandard
  MLIRMathToLLVM
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
