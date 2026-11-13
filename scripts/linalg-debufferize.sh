#!/bin/bash

# linalg-debufferize.sh
# Script to run the full debufferization pipeline:
# 1. cgeist: C -> MLIR
# 2. polygeist-opt: Raise to linalg (with optional function selection)
# 3. polygeist-opt: Debufferize

set -e  # Exit on error

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <input.c> <function-name>"
    echo ""
    echo "Arguments:"
    echo "  input.c       - Input C file to process"
    echo "  function-name - Function name to select for debufferization (used in --select-func)"
    echo ""
    echo "Example:"
    echo "  $0 kernel_gemm.c gemm"
    echo "  $0 conv.c conv_2d"
    exit 1
fi

INPUT_FILE="$1"
FUNC_NAME="$2"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found"
    exit 1
fi

# Get the base name without extension
BASENAME=$(basename "$INPUT_FILE" .c)
DIRNAME=$(dirname "$INPUT_FILE")

# Create temp directory based on filename
TEMP_DIR="${DIRNAME}/${BASENAME}_debufferize_temp"
mkdir -p "$TEMP_DIR"

echo "=== Linalg Debufferization Pipeline ==="
echo "Input file: $INPUT_FILE"
echo "Function:   $FUNC_NAME"
echo "Temp dir:   $TEMP_DIR"
echo ""

# Step 1: C to MLIR with cgeist
echo "[1/3] Running cgeist: C -> MLIR..."
MLIR_OUTPUT="${TEMP_DIR}/${BASENAME}.mlir"
CMD="cgeist $INPUT_FILE --function=$FUNC_NAME --resource-dir=/usr/lib/clang/14 --raise-scf-to-affine -fPIC -S -g -c -o $MLIR_OUTPUT"
echo "      Command: $CMD"
cgeist "$INPUT_FILE" \
    --function="$FUNC_NAME" \
    --resource-dir=/usr/lib/clang/14 \
    --raise-scf-to-affine \
    -fPIC -S -g -c \
    -o "$MLIR_OUTPUT"
echo "      Output: $MLIR_OUTPUT"

# Step 2: Raise to linalg
echo "[2/3] Running polygeist-opt: Raise to linalg..."
LINALG_OUTPUT="${TEMP_DIR}/${BASENAME}_linalg.mlir"
CMD="polygeist-opt --select-func=func-name=$FUNC_NAME --remove-iter-args --affine-parallelize --raise-affine-to-linalg-pipeline $MLIR_OUTPUT -o $LINALG_OUTPUT"
echo "      Command: $CMD"
polygeist-opt \
    --select-func="func-name=$FUNC_NAME" \
    --remove-iter-args \
    --affine-parallelize \
    --raise-affine-to-linalg-pipeline \
    "$MLIR_OUTPUT" \
    -o "$LINALG_OUTPUT"
echo "      Output: $LINALG_OUTPUT"

# Step 3: Debufferize
echo "[3/3] Running polygeist-opt: Debufferize..."
DEBUF_OUTPUT="${TEMP_DIR}/${BASENAME}_debufferized.mlir"
CMD="polygeist-opt --linalg-debufferize $LINALG_OUTPUT -o $DEBUF_OUTPUT"
echo "      Command: $CMD"
polygeist-opt \
    --linalg-debufferize \
    "$LINALG_OUTPUT" \
    -o "$DEBUF_OUTPUT"
echo "      Output: $DEBUF_OUTPUT"

echo ""
echo "=== Pipeline Complete ==="
echo "Final output: $DEBUF_OUTPUT"
echo ""
echo "Intermediate files in: $TEMP_DIR"
ls -la "$TEMP_DIR"

