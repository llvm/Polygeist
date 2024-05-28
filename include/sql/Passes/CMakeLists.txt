set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name sql)
add_public_tablegen_target(MLIRSQLPassIncGen)

add_mlir_doc(Passes SQLPasses ./ -gen-pass-doc)
