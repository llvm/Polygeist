#!/usr/bin/env sh


set -e
set -x

echo Starting pluto build script with args "$@"

# We assume build dir is absolute
ROOT_DIR="$1"

echo BUILDING CLOOG IN DIR "$ROOT_DIR"


CLOOG_PREFIX="$ROOT_DIR/cloog"
if test -d "$CLOOG_PREFIX"; then
    exit
fi
CLOOG_SRC_DIR="$CLOOG_PREFIX/cloog"
CLOOG_INSTALL_DIR="$CLOOG_PREFIX/install"

mkdir -p "$CLOOG_PREFIX"
cd "$CLOOG_PREFIX"


git clone https://github.com/kumasento/cloog.git "$CLOOG_SRC_DIR" || true
cd "$CLOOG_SRC_DIR"
git checkout 43CFB85ED1E1BA1C2F27B450498522B35467ACE7
git submodule update --init --recursive


"./autogen.sh"
"./configure" --prefix="$CLOOG_INSTALL_DIR" --with-osl-prefix="$ROOT_DIR/openscop/install"
make install -j
