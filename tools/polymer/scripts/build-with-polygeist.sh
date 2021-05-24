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

echo ">>> Linking Polymer to Polygeist ..."
rm -r "${POLYGEIST_DIR}/mlir/tools/polymer"
ln -s "${POLYMER_DIR}" "${POLYGEIST_DIR}/mlir/tools/polymer"

echo ">>> Building Polygeist ..."
cd "${POLYGEIST_DIR}"
mkdir -p build
cd build

if [ ! -f CMakeCache.txt ]; then
  # Comment out -G Ninja if you don't want to use that.
  cmake ../llvm \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DLLVM_ENABLE_PROJECTS="llvm;mlir;clang" \
    -DLLVM_OPTIMIZED_TABLEGEN=ON \
    -DLLVM_ENABLE_OCAMLDOC=OFF \
    -DLLVM_ENABLE_BINDINGS=OFF \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DBUILD_POLYMER=ON \
    -DPLUTO_LIBCLANG_PREFIX="$(llvm-config --prefix)"
else
  echo "-- CMakeCache.txt file has been found. CMake generation is therefore skipped."
fi

# Build
LD_LIBRARY_PATH="${POLYGEIST_DIR}/build/tools/mlir/tools/polymer/pluto/lib:${LD_LIBRARY_PATH}" cmake --build . --target check-polymer

echo ">>> Done!"

rm "${POLYMER_DIR}/build"
ln -s "${POLYGEIST_DIR}/build" "${POLYMER_DIR}/build"

echo ""
echo "    Polymer utilities are built under ${POLYGEIST_DIR}/build,"
echo "    and they are linked back to ${POLYMER_DIR}/build."
echo ""
