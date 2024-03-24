echo "[LOG]: converting test.cpp to test.mlir"
cgeist test.cpp -raise-scf-to-affine -O2 -S -o test.mlir

echo "[LOG]: generating memory access information"
bash test_memacc.sh

echo "[LOG]: lowering to LLVM"
bash lower_to_llvm.sh