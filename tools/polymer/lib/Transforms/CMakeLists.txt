add_mlir_library(PolymerTransforms
  InvariantScopTransform.cc
  PlutoTransform.cc

  ADDITIONAL_HEADER_DIRS
  "${POLYMER_MAIN_INCLUDE_DIR}/polymer/Transforms"

  LINK_LIBS PUBLIC
  MLIRAffineOps
  MLIRAnalysis
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRIR
  MLIRStandardOps
  MLIRSupport
  MLIRAffineToStandard

  PolymerSupport
  PolymerTargetOpenScop
)
