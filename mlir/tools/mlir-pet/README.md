## Build instructions

### Requirements
- working C and C++ toolchains (compiler, linker)
- cmake
- make or ninja
- libgmp
- libz

### Process

```
# checkout the repository, 'cd' to it, then
mkdir -p build
cd build
cmake ../llvm \
  -DLLVM_TARGETS_TO_BUILD="host;NVPTX" \
  -DLLVM_ENABLE_PROJECTS="clang;isl;pet;mlir" \
  -DLLVM_ENABLE_ASSERTIONS=On \
  -DCMAKE_BUILD_TYPE=Release \
  -G Ninja  # optional, to use ninja instead of make
make  # or ninja; preferably with -j <num-cores>
```

Note that everything except libgmp is statically linked, the produced binary
should be freely movable around the file system.


### How to run

```
git clone https://github.com/Meinersbur/polybench
cd polybench/polybench-code
for i in `cat utilities/benchmark_list`; do FNAME=`basename $i`; echo $FNAME; clang -E -Iutilities -DPOLYBENCH_USE_SCALAR_LB $i > /tmp/$FNAME; bin/mlir-pet  /tmp/$FNAME; done
```

