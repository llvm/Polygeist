# Ginsbach/LiLAC B300 runbook

All paths below are under `/mnt/forge1/arpitj`. The author repositories are
pinned to LLVM `113d41f6e3695045fcc6e93166b3d1f5df1a8074` and Clang
`5e3d3c0055dc731abd506e18c320f227168f728b`.

## Compiler configuration

```text
cmake -G Ninja repos/ginsbach-llvm-linearalgebra \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DLLVM_TARGETS_TO_BUILD=X86 \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DLLVM_INCLUDE_BENCHMARKS=OFF
PATH=bin:$PATH cmake --build build/ginsbach-linearalgebra \
  --target clang opt lilac_target -- -j32
```

The unmodified build expects `pypy`; a local `pypy -> /usr/bin/python3` shim
was used after confirming that `lib/IDLParser/stage3.py` runs under Python 3.

## Generated CUDA harnesses

```text
git -C repos/ginsbach-llvm-linearalgebra apply \
  patches/ginsbach-linearalgebra-cuda13.patch
cmake --build build/ginsbach-linearalgebra --target lilac_target -- -j8
nvcc -std=c++14 -Xcompiler=-fPIC -shared \
  -Irepos/ginsbach-llvm-linearalgebra/include \
  build/ginsbach-linearalgebra/lilac/gemm_cuda.cc -lcublas \
  -o build/ginsbach-linearalgebra/cuda-harness/libgemm_cuda.so
nvcc -std=c++14 -Xcompiler=-fPIC -shared \
  -Irepos/ginsbach-llvm-linearalgebra/include \
  build/ginsbach-linearalgebra/lilac/spmv_csr_cuda.cc -lcusparse \
  -o build/ginsbach-linearalgebra/cuda-harness/libspmv_csr_cuda.so
```

## Recognition compilation

Every raised source was compiled with the author compiler using:

```text
build/ginsbach-linearalgebra/bin/clang -O2 \
  -fno-vectorize -fno-slp-vectorize -fno-unroll-loops -c INPUT.c -o OUTPUT.o
```

CUDA-timed objects additionally used `-DUSE_CUDA_SYNC`. The retained
`replace-report-*.json`, `replace-source-*.ll`, and `nm` evidence demonstrate
that the ordinary C loop was replaced by the generated harness symbol.

## Measurement commands

The CPU core available and selected for the campaign was logical CPU 10.
Commands were executed sequentially after an idle-GPU preflight:

```text
taskset -c 10 ./gemm_native_cpu 512 512 512
taskset -c 10 ./gemm_raised_naive 512 512 512
CUDA_VISIBLE_DEVICES=0 ./gemm_native_cublas 512 512 512
CUDA_VISIBLE_DEVICES=0 ./gemm_raised_cublas 512 512 512
taskset -c 10 ./spmv_csr_native_cpu 1000000 1000000 27
taskset -c 10 ./spmv_csr_raised_naive 1000000 1000000 27
CUDA_VISIBLE_DEVICES=0 ./spmv_csr_native_cusparse 1000000 1000000 27
CUDA_VISIBLE_DEVICES=0 ./spmv_csr_raised_cusparse 1000000 1000000 27
```

Each executable performs five untimed warmups followed by five synchronized
samples. GPU executables report both synchronized host wall time and CUDA-event
device time. Correctness checks cover every output element and occur after the
timed region.
