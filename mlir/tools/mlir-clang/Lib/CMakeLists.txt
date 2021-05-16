add_mlir_library(MLIRClang
  MLIRClang.cc
  Utils.cc
  PragmaLowerToHandler.cc

  EXCLUDE_FROM_LIBMLIR

  DEPENDS
  mlir-headers
  ClangDriverOptions

  LINK_COMPONENTS
  Core
  Support

  LINK_LIBS PUBLIC

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
  clangSema
)
