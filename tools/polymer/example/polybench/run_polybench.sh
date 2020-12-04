#!/bin/bash

SOURCE_DIR="$1"
OUTPUT_DIR="$2"

BINDIR="${PWD}/../../build/bin"

TOTAL_CASES=0
SUCCESSFUL_CASES=0

printf "%50s %5s\n" "Benchmark" "Exit code"

for f in $(find "${SOURCE_DIR}" -name "*.mlir"); do
  DIRNAME=$(dirname "${f}")
  BASENAME=$(basename "${f}")
  NAME="${BASENAME%.*}"

  # Where the output result will be generated to.
  mkdir -p "${OUTPUT_DIR}/${DIRNAME}/${NAME}"
  "${BINDIR}/polymer-opt" "$f" 2>/dev/null | tee "${OUTPUT_DIR}/${DIRNAME}/${NAME}/${NAME}.mlir" >/dev/null
  "${BINDIR}/polymer-opt" -reg2mem -extract-scop-stmt "$f" 2>/dev/null | tee "${OUTPUT_DIR}/${DIRNAME}/${NAME}/${NAME}.scop.mlir" >/dev/null
  # if [ -s "${OUTPUT_DIR}/${DIRNAME}/${NAME}/${NAME}.scop.mlir" ]; then
  #   "${BINDIR}/polymer-translate" -export-scop "${OUTPUT_DIR}/${DIRNAME}/${NAME}/${NAME}.scop.mlir" | tee "${OUTPUT_DIR}/${DIRNAME}/${NAME}/${NAME}.scop" >/dev/null
    # ../pluto/tool/pluto --readscop "${OUTPUT_DIR}/${DIRNAME}/${NAME}.scop" 2>&1 >/dev/null 
    # mv "${NAME}.scop.pluto.cloog" "${OUTPUT_DIR}/${DIRNAME}" 
    # mv "${NAME}.scop.pluto.c" "${OUTPUT_DIR}/${DIRNAME}" 
  # fi

  "${BINDIR}/polymer-opt" -reg2mem -extract-scop-stmt "$f" 2>/dev/null | tee "${OUTPUT_DIR}/${DIRNAME}/${NAME}/${NAME}.scop.mlir" >/dev/null
  # The optimization command
  "${BINDIR}/polymer-opt" -reg2mem -extract-scop-stmt -pluto-opt -canonicalize "$f" 2>/dev/null | "${BINDIR}/polymer-opt" | tee "${OUTPUT_DIR}/${DIRNAME}/${NAME}/${NAME}.pluto.mlir" >/dev/null
  "${BINDIR}/polymer-opt" -reg2mem -extract-scop-stmt -pluto-par -canonicalize "$f" 2>/dev/null | "${BINDIR}/polymer-opt" | tee "${OUTPUT_DIR}/${DIRNAME}/${NAME}/${NAME}.pluto-par.mlir" >/dev/null
  "${BINDIR}/polymer-opt" -reg2mem -extract-scop-stmt -pluto-opt -inline -canonicalize  "$f" 2>/dev/null | "${BINDIR}/polymer-opt" | tee "${OUTPUT_DIR}/${DIRNAME}/${NAME}/${NAME}.pluto-inline.mlir" >/dev/null

  # Report
  EXIT_STATUS="${PIPESTATUS[0]}"
  printf "%50s %5d\n" "${f}" "${EXIT_STATUS}"

  ((TOTAL_CASES=TOTAL_CASES+1))
  if [ ${EXIT_STATUS} -eq 0 ]; then
    ((SUCCESSFUL_CASES=SUCCESSFUL_CASES+1))
  fi

done

echo ""
printf "%50s\n" "Report:"
printf "%50s %5d\n" "Total cases:" "${TOTAL_CASES}"
printf "%50s %5d\n" "Successful cases:" "${SUCCESSFUL_CASES}"
