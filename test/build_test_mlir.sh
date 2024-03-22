cgeist test.cpp -raise-scf-to-affine -O2 -S -o test.mlir

polygeist-opt test.mlir --affine-cfg --affine-simplify-structures --affine-parallelize --canonicalize -o test_parallel.mlir
