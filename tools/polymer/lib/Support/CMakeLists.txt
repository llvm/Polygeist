add_mlir_library(PolymerSupport
  OslScop.cc
  OslScopStmtOpSet.cc
  OslSymbolTable.cc
  ScopStmt.cc
  ScatteringUtils.cc

  DEPENDS
  intrinsics_gen

  ADDITIONAL_HEADER_DIRS
  ${POLYMER_MAIN_INCLUDE_DIR}/polymer/Support

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRLoopAnalysis

  # libosl
  # libcloog
  # libisl
  libplutoisl
  libplutoosl
  libplutopip
  libplutopolylib
  libplutocloog
  libplutocandl
  libpluto
)
