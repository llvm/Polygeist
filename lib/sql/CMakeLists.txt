add_mlir_dialect_library(MLIRSQL
Dialect.cpp
Ops.cpp

ADDITIONAL_HEADER_DIRS
${PROJECT_SOURCE_DIR}/include/sql

DEPENDS
MLIRSQLOpsIncGen

LINK_LIBS PUBLIC
MLIRIR
MLIRMemRefDialect
MLIRLLVMDialect
MLIROpenMPDialect
MLIRAffineDialect
MLIRSupport
MLIRSCFTransforms
)
