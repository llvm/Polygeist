#!/usr/bin/bash -x

NAME="$1"
OPT_NAME="${NAME}.pluto-par"
DATASET_SIZE="$2"

ORIG_MLIR_FILE="${NAME}.mlir"
MAIN_FILE="${OPT_NAME}.c"
MAIN_LLVMIR_FILE="${OPT_NAME}.ll"
MLIR_FILE="${OPT_NAME}.mlir"
MLIR_LLVM_FILE="${OPT_NAME}_llvm.mlir"
MLIR_LLVMIR_FILE="${OPT_NAME}_mlir.ll"
MLIR_BC_FILE="${OPT_NAME}_mlir.bc"

RESULT_BC_FILE="result.bc"
RESULT_OBJ_FILE="result.o"

EXE="${NAME}.bin"

LLVM_BINDIR="${PWD}/../../llvm/build/bin"
POLYMER_BINDIR="${PWD}/../../build/bin"

${POLYMER_BINDIR}/polymer-opt \
  -reg2mem \
  -extract-scop-stmt \
  -pluto-par \
  -canonicalize \
  "${ORIG_MLIR_FILE}"  2>/dev/null | tee "${MLIR_FILE}" > /dev/null

${LLVM_BINDIR}/mlir-opt \
  --affine-parallelize \
  --lower-affine \
  --convert-scf-to-std \
  --canonicalize \
  --convert-std-to-llvm='emit-c-wrappers=1' \
  "${MLIR_FILE}" \
  -o "${MLIR_LLVM_FILE}" 

${LLVM_BINDIR}/mlir-translate "${MLIR_LLVM_FILE}" --mlir-to-llvmir -o "${MLIR_LLVMIR_FILE}"

${LLVM_BINDIR}/opt -O3 -march=native "${MLIR_LLVMIR_FILE}" -o "${MLIR_BC_FILE}"

# ${LLVM_BINDIR}/llvm-as "${MLIR_LLVMIR_FILE}" -o "${MLIR_BC_FILE}"

${LLVM_BINDIR}/clang -O3 -march=native -D"${DATASET_SIZE}_DATASET" -emit-llvm "${MAIN_FILE}" -S -o "${MAIN_LLVMIR_FILE}"

${LLVM_BINDIR}/llvm-link "${MAIN_LLVMIR_FILE}" "${MLIR_LLVMIR_FILE}" -o "${RESULT_BC_FILE}"

${LLVM_BINDIR}/llc -filetype=obj "${RESULT_BC_FILE}"

${LLVM_BINDIR}/clang -O3 -march=native "${RESULT_OBJ_FILE}" -o "${EXE}"

"./${EXE}"

rm -f ${RESULT_BC_FILE}
rm -f ${RESULT_OBJ_FILE}
