add_mlir_conversion_library(PolymerTransforms
  PlutoTransform.cc
  Reg2Mem.cc
  ExtractScopStmt.cc
  ScopStmtOpt.cc
  LoopAnnotate.cc
  LoopExtract.cc
  FoldSCFIf.cc
  AnnotateScop.cc

  ADDITIONAL_HEADER_DIRS
  "${POLYMER_MAIN_INCLUDE_DIR}/polymer/Transforms"

  DEPENDS
  PolymerTransformsIncGen

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
  MLIRAffineTransforms

  PolymerSupport
  PolymerTargetOpenScop
)
