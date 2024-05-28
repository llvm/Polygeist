add_mlir_dialect_library(MLIRSQLTransforms
  SQLLower.cpp
  SQLRaising.cpp

  DEPENDS
  MLIRPolygeistOpsIncGen
  MLIRPolygeistPassIncGen
  MLIRSQLPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRIR
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRPass
  )