#!/usr/bin/env bash
# Shared path setup for correctness and Jetson pipeline scripts.

_POLYGEIST_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${POLYGEIST_ROOT:-$(cd "$_POLYGEIST_SCRIPT_DIR/../.." && pwd)}"
POLYGEIST_ROOT="$REPO_ROOT"
SCRIPT_DIR="${SCRIPT_DIR:-$_POLYGEIST_SCRIPT_DIR}"

if [[ -f "$REPO_ROOT/envsetup.sh" ]]; then
  source "$REPO_ROOT/envsetup.sh"
else
  export PATH="$REPO_ROOT/build/bin:$PATH"
fi

PYTHON="${PYTHON:-python3}"
PY="${PY:-$PYTHON}"
SCRIPTS="${SCRIPTS:-$SCRIPT_DIR}"
RT="${RT:-$REPO_ROOT/runtime}"
MLIR_OPT="${MLIR_OPT:-$REPO_ROOT/llvm-project/build/bin/mlir-opt}"
MLIR_TRANSLATE="${MLIR_TRANSLATE:-$REPO_ROOT/llvm-project/build/bin/mlir-translate}"
CLANG="${CLANG:-$REPO_ROOT/llvm-project/build/bin/clang}"
KERNEL_LIB="${KERNEL_LIB:-$REPO_ROOT/generic_solver/kernel_library_phase2.mlir}"
POLYBENCH_DIR="${POLYBENCH_DIR:-$REPO_ROOT/tools/cgeist/Test/polybench}"

PVASOL_ROOT="${PVASOL_ROOT:-$HOME/pva-solutions}"
CV_CUDA_ROOT="${CV_CUDA_ROOT:-$HOME/cv-cuda}"
CUPVA_SDK_ROOT="${CUPVA_SDK_ROOT:-$HOME/cupva_sdk_include}"
PVA_LIB_STAGE="${PVA_LIB_STAGE:-$HOME/pva_libs}"
JETSON_NVIDIA_LIBS="${JETSON_NVIDIA_LIBS:-$HOME/jetson_nvidia_libs}"
