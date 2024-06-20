if(POLYGEIST_POLYMER_ENABLE_PLUTO)
  add_mlir_library(PolymerSupport
    OslScop.cc
    OslScopStmtOpSet.cc
    OslSymbolTable.cc
    ScopStmt.cc
    ScatteringUtils.cc
    Utils.cc

    DEPENDS
    mlir-headers

    ADDITIONAL_HEADERS
    ${POLYMER_MAIN_INCLUDE_DIR}/polymer/Support

    LINK_LIBS PUBLIC
    MLIRAnalysis
    MLIRAffineAnalysis

    # libosl
    libcloog
    # libisl
    libplutoisl
    libplutoosl
    libplutopip
    libplutopolylib
    libplutocloog
    libplutocandl
    libpluto
  )
elseif(POLYGEIST_POLYMER_ENABLE_ISL)
  add_mlir_library(PolymerSupport
    IslScop.cc
    ScopStmt.cc
    Utils.cc

    DEPENDS
    mlir-headers

    ADDITIONAL_HEADERS
    ${POLYMER_MAIN_INCLUDE_DIR}/polymer/Support

    LINK_LIBS PUBLIC
    MLIRAnalysis
    MLIRAffineAnalysis

    PollyISL
    Polly
  )
endif()

target_link_libraries(PolymerSupport PUBLIC ${GMP_LIBRARY})
