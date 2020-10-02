add_mlir_library(PolymerSupport
  OslScop.cc

  ADDITIONAL_HEADER_DIRS
  ${POLYMER_MAIN_INCLUDE_DIR}/polymer/Support

  LINK_LIBS PUBLIC
  libosl
  libcloog
  libisl
)
