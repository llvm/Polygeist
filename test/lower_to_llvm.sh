polygeist-opt test.mlir --memory-access-generation --mlir-disable-threading -o test_memacc.mlir
polygeist-opt test_memacc.mlir --lower-affine -o test_memacc.mlir
polygeist-opt test_memacc.mlir --convert-polygeist-to-llvm -o test_memacc.mlir