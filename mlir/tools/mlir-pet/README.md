## Build instructions

### Requirements
- working C and C++ toolchains (compiler, linker)
- cmake
- make or ninja
- libgmp
- libz

### Process

1. Configure environment and fetch source.

```
# Before executing, set up the directory paths.
# We assume LLVM git checkout lives in $LLVM_DIR and everything will be
# installed into a single location $INSTALL_DIR.
export LLVM_DIR=...
export ISL_DIR=...
export PET_DIR=...
export INSTALL_DIR=...

git clone https://github.com/LoopTactics/mlir $LLVM_DIR
git clone https://github.com/chelini/isl $ISL_DIR
git clone https://repo.or.cz/pet.git $PET_DIR

# We want the environment to contain the binaries, libraries and headers
# that we are about to install.
export PATH=$INSTALL_DIR/bin:$PATH
export CPATH=$INSTALL_DIR/include:$CPATH
export LIBRARY_PATH=$INSTALL_DIR/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$INSTALL_DIR/lib:$LD_LIBRARY_PATH
```

2. Configure and build LLVM, Clang and MLIR **without** mlir-pet.

```
cd $LLVM_DIR
mkdir -p build
cd build
cmake ../llvm \
  -DCMAKE_PREFIX_PATH=$INSTALL_DIR \
  -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -G Ninja  # optional, to use ninja instead of make
make install  # or ninja; preferably with -j <num-cores>
```

3. Configure and build isl.

```
cd $ISL_DIR
./autogen.sh
./configure --prefix=$INSTALL_DIR
make install  # preferably with -j <num-cores>
```

Note do _NOT_ configure `--with-clang` as it would overwrite the bindings
header.

4. Configure and build pet.

```
cd $PET_DIR
./autogen.sh
./configure --prefix=$INSTALL_DIR --with-clang-prefix=$INSTALL_DIR --with-isl=system \
  --with-isl-prefix=$INSTALL_DIR
make install  # preferablt with -j <num-cores>
```

This will make sure `pet` is linked against the same vesion of LLVM as MLIR.

4. Re-configure MLIR to build mlir-pet.

```
cd $LLVM_DIR/build
cmake ../llvm \
  -DCMAKE_PREFIX_PATH=$INSTALL_DIR \
  -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_PET_BUILD=On \
  -G Ninja  # optional, to use ninja instead of make
make install-mlir-pet  # or ninja install-mlir-pet; preferably with -j <num-cores>
```

### How to run

```
git clone https://github.com/Meinersbur/polybench
cd polybench/polybench-code
for i in `cat utilities/benchmark_list`; do FNAME=`basename $i`; echo $FNAME; clang -E -Iutilities -DPOLYBENCH_USE_SCALAR_LB $i > /tmp/$FNAME; mlir-pet  /tmp/$FNAME; done
```

