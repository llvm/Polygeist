
if(POLYGEIST_POLYMER_ENABLE_PLUTO)
  set(POLYMER_TRANSFORM_SRC
    PlutoTransform.cc
  )
elseif(POLYGEIST_POLYMER_ENABLE_ISL)
  set(POLYMER_TRANSFORM_SRC
    IslExternalTransform.cc
  )
endif()

add_mlir_conversion_library(PolymerTransforms
  Reg2Mem.cc
  ExtractScopStmt.cc
  ScopStmtOpt.cc
  LoopAnnotate.cc
  LoopExtract.cc
  FoldSCFIf.cc
  AnnotateScop.cc
  "${POLYMER_TRANSFORM_SRC}"

  ADDITIONAL_HEADER_DIRS
  "${POLYMER_MAIN_INCLUDE_DIR}/polymer/Transforms"

  DEPENDS
  PolymerTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAnalysis
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRIR
  MLIRFuncDialect
  MLIRSupport
  MLIRAffineToStandard
  MLIRAffineTransforms

  PolymerSupport
  "${POLYMER_AVAIL_TARGETS}"
)
