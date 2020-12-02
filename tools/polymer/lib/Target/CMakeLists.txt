add_mlir_translation_library(PolymerTargetOpenScop
  OpenScop/ConvertFromOpenScop.cc
  OpenScop/ConvertToOpenScop.cc

  ADDITIONAL_HEADER_DIRS
  ${POLYMER_MAIN_INCLUDE_DIR}/polymer/Target/OpenScop

  LINK_COMPONENTS
  Core
  TransformUtils

  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRAffine
  MLIRAffineUtils
  MLIRTranslation

  PolymerSupport
)
