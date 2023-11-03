#!/usr/bin/env sh


set -e
set -x

echo Starting pluto build script with args "$@"

# We assume build dir is absolute
ROOT_DIR="$1"

echo BUILDING OPENSCOP IN DIR "$ROOT_DIR"


OPENSCOP_PREFIX="$ROOT_DIR/openscop"
if test -d "$OPENSCOP_PREFIX"; then
    exit
fi
OPENSCOP_SRC_DIR="$OPENSCOP_PREFIX/openscop"
OPENSCOP_INSTALL_DIR="$OPENSCOP_PREFIX/install"

mkdir -p "$OPENSCOP_PREFIX"
cd "$OPENSCOP_PREFIX"


git clone https://github.com/periscop/openscop.git "$OPENSCOP_SRC_DIR" || true
cd "$OPENSCOP_SRC_DIR"
git checkout 37805d8fef38c2d1b8aa8f5c26b40f79100322e7


"./autogen.sh" && "./configure" --prefix="$OPENSCOP_INSTALL_DIR"
make install -j
