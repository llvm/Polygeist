add_mlir_dialect_library(MLIRSQL
Types.cpp
Dialect.cpp
Ops.cpp
Parser.cpp


ADDITIONAL_HEADER_DIRS
${PROJECT_SOURCE_DIR}/include/sql

DEPENDS
MLIRSQLOpsIncGen
# MLIRSQLTypesIncGen

LINK_LIBS PUBLIC
MLIRIR
)

add_subdirectory(Passes)