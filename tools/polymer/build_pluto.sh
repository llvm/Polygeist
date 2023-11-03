#!/usr/bin/env sh

set -e
set -x

echo Starting pluto build script with args "$@"

# We assume build dir is absolute
ROOT_DIR="$1"

echo BUILDING PLUTO IN DIR "$ROOT_DIR"

if test -d "$ROOT_DIR"; then
    exit
fi

mkdir -p "$ROOT_DIR"


### LLVM STUFF ###

PLUTO_LLVM_PREFIX="$ROOT_DIR/llvm"

mkdir -p "$PLUTO_LLVM_PREFIX"
cd "$PLUTO_LLVM_PREFIX"

PLUTO_LLVM_SRC_DIR="$PLUTO_LLVM_PREFIX/llvm-project"
PLUTO_LLVM_BUILD_DIR="$PLUTO_LLVM_PREFIX/build"
PLUTO_LLVM_INSTALL_DIR="$PLUTO_LLVM_PREFIX/install"

git clone https://github.com/llvm/llvm-project.git "$PLUTO_LLVM_SRC_DIR" || true

cd "$PLUTO_LLVM_SRC_DIR"

git checkout release/10.x

sed -i.bak -e "/\#include \"llvm\\/Support\\/Signals.h\"/i \#include <stdint.h>" llvm/lib/Support/Signals.cpp
sed -i.bak -e "/\#include <vector>/i \#include <limits>" llvm/utils/benchmark/src/benchmark_register.h

mkdir -p "$PLUTO_LLVM_BUILD_DIR"
cd "$PLUTO_LLVM_BUILD_DIR"
cmake -G Ninja -DCMAKE_INSTALL_PREFIX="$PLUTO_LLVM_INSTALL_DIR" -DLLVM_ENABLE_PROJECTS=clang -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=X86 "$PLUTO_LLVM_SRC_DIR/llvm"
cmake --build . -j --target install


### OPENSCOP ###

PLUTO_PREFIX="$ROOT_DIR/pluto"
PLUTO_SRC_DIR="$PLUTO_PREFIX/pluto"
PLUTO_INSTALL_DIR="$PLUTO_PREFIX/install"
#
#
# OPENSCOP_PREFIX="$PLUTO_SRC_DIR"
# mkdir -p "$OPENSCOP_PREFIX"
# cd "$OPENSCOP_PREFIX"
#
# OPENSCOP_SRC_DIR="$OPENSCOP_PREFIX/openscop"
# git clone https://github.com/periscop/openscop.git "$OPENSCOP_SRC_DIR" || true
# cd "$OPENSCOP_SRC_DIR"
# git checkout 37805d8fef38c2d1b8aa8f5c26b40f79100322e7
#
# OPENSCOP_INSTALL_DIR="$OPENSCOP_PREFIX/install"
#
# "./autogen.sh" && "./configure" --prefix="$OPENSCOP_INSTALL_DIR"
# make install -j


### PLUTO STUFF ###

# include_directories("${llvm10_SOURCE_DIR}/clang/include")

mkdir -p "$PLUTO_PREFIX"
cd "$PLUTO_PREFIX"

#git clone --recurse-submodules https://github.com/kumasento/pluto "$PLUTO_SRC_DIR" || true
git clone https://github.com/kumasento/pluto "$PLUTO_SRC_DIR" || true
cd "$PLUTO_SRC_DIR"
git checkout 5603283fb3e74fb33c380bb52874972b440d51a2
git submodule update --init --recursive

./autogen.sh
./configure --prefix="$PLUTO_INSTALL_DIR" --with-clang-prefix="$PLUTO_LLVM_INSTALL_DIR"
make -j install
