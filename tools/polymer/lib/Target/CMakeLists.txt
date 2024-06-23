if(POLYGEIST_POLYMER_ENABLE_PLUTO)
  add_mlir_translation_library(PolymerTargetOpenScop
    OpenScop/ConvertFromOpenScop.cc
    OpenScop/ConvertToOpenScop.cc

    ADDITIONAL_HEADER_DIRS
    ${POLYMER_MAIN_INCLUDE_DIR}/polymer/Target/OpenScop

    LINK_COMPONENTS
    Core
    TransformUtils

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRAffineDialect
    MLIRAffineUtils
    MLIRSupport

    PolymerSupport
  )
elseif(POLYGEIST_POLYMER_ENABLE_ISL)
  add_mlir_translation_library(PolymerTargetISL
    ISL/ConvertToISL.cc

    ADDITIONAL_HEADER_DIRS
    ${POLYMER_MAIN_INCLUDE_DIR}/polymer/Target/ISL

    LINK_COMPONENTS
    Core
    TransformUtils

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRAffineDialect
    MLIRAffineUtils
    MLIRSupport

    PolymerSupport
  )
endif()
