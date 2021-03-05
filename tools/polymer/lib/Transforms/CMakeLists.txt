add_mlir_library(PolymerTransforms
  PlutoTransform.cc
  Reg2Mem.cc
  ExtractScopStmt.cc
  ScopStmtOpt.cc

  ADDITIONAL_HEADER_DIRS
  "${POLYMER_MAIN_INCLUDE_DIR}/polymer/Transforms"

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRAnalysis
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRIR
  MLIRStandard
  MLIRSupport
  MLIRAffineToStandard

  PolymerSupport
  PolymerTargetOpenScop
)
