add_mlir_library(POLYMEREmitOpenScop
  EmitOpenScop.cc
  OslScop.cc
  
  ADDITIONAL_HEADER_DIRS

  LINK_LIBS
  PUBLIC
  MLIRAffineOps
  MLIRAffineUtils
  MLIRTranslation

  osl
)