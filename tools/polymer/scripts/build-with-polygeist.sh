#!/usr/bin/env bash
# This script build and install the Polygeist version that Polymer can work together with.

set -o errexit
set -o pipefail
set -o nounset

# Directory of this script
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd)"
POLYGEIST_DIR="${DIR}/../../Polygeist-polymer"
POLYMER_DIR="${DIR}/../"

# Read the file that records the Polygeist git hash.
POLYGEIST_VERSION="$(cat "${DIR}/../polygeist-version.txt")"

echo ">>> Update and build Polygeist"
echo ""
echo "   The Polygeist version: ${POLYGEIST_VERSION}"
echo ""

echo ">>> Cloning and checkout Polygeist ..."

if [ ! -d "${POLYGEIST_DIR}" ]; then
  git clone https://github.com/wsmoses/Polygeist "${POLYGEIST_DIR}"
fi

cd "${POLYGEIST_DIR}"
git fetch origin "${POLYGEIST_VERSION}"
git checkout "${POLYGEIST_VERSION}"
cd - &>/dev/null

cd "${POLYGEIST_DIR}"


echo ">>> Building LLVM ..."
if [ ! -d "${POLYGEIST_DIR}/llvm-project/build" ]; then
  git submodule init
  git submodule update

  mkdir -p llvm-project/build
  cd llvm-project/build

  # Comment out -G Ninja if you don't want to use that.
  cmake ../llvm \
    -G Ninja \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DLLVM_ENABLE_PROJECTS="llvm;mlir;clang" \
    -DLLVM_OPTIMIZED_TABLEGEN=ON \
    -DLLVM_ENABLE_OCAMLDOC=OFF \
    -DLLVM_ENABLE_BINDINGS=OFF \
    -DLLVM_INSTALL_UTILS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON 
  cmake --build . 
  cmake --build . --target check-mlir
else
  echo "-- LLVM build is skipped."
fi



# Build Polygeist
mkdir -p "${POLYGEIST_DIR}/build"
cd "${POLYGEIST_DIR}/build"

# Comment out -G Ninja if you don't want to use that.
cmake .. \
  -G Ninja \
  -DMLIR_DIR="${POLYGEIST_DIR}/llvm-project/build/lib/cmake/mlir" \
  -DCLANG_DIR="${POLYGEIST_DIR}/llvm-project/build/lib/cmake/clang" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG
cmake --build . --target check-mlir-clang

# Build polymer
cd "${POLYMER_DIR}"
mkdir -p "${POLYMER_DIR}/build"
cd "${POLYMER_DIR}/build"
cmake -G Ninja \
  .. \
  -DMLIR_DIR="${POLYGEIST_DIR}/llvm-project/build/lib/cmake/mlir" \
  -DLLVM_DIR="${POLYGEIST_DIR}/llvm-project/build/lib/cmake/llvm" \
  -DLLVM_EXTERNAL_LIT="${POLYGEIST_DIR}/llvm-project/build/bin/llvm-lit" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG
LD_LIBRARY_PATH="${POLYMER_DIR}/build/pluto/lib:${LD_LIBRARY_PATH}" cmake --build . --target check-polymer

echo ">>> Done!"

echo ""
echo "    Polymer utilities are built under ${POLYGEIST_DIR}/build,"
echo "    and they are linked back to ${POLYMER_DIR}/build."
echo ""
