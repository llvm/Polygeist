CPU BLAS Runtime Backend
========================

Polygeist's `--lower-kernel-launch-to-cublas` ABI can be linked against a CPU
runtime for host-side correctness and performance experiments. By default,
`runtime/polygeist_cublas_rt_cpu.c` uses simple reference loops. For BLAS-like
symbols, it can instead call an optimized CBLAS implementation such as
OpenBLAS, BLIS, MKL, ArmPL, or NVPL.

Supported CBLAS-routed symbols
------------------------------

When `POLYGEIST_CPU_BLAS=1` is set for a host build, the CPU runtime compiles
with `POLYGEIST_CPU_USE_CBLAS` and routes these symbols through CBLAS:

* `polygeist_cublas_dgemm`
* `polygeist_cublas_sgemm`
* `polygeist_cublas_dgemv`
* `polygeist_cublas_sgemv`
* `polygeist_cublas_dgemv_T`
* `polygeist_cublas_sgemv_T`
* `polygeist_cublas_daxpby`
* `polygeist_cublas_daxpy_unit`
* `polygeist_cublas_dger_rank2`
* `polygeist_cublas_dscal_2d`
* `polygeist_cublas_sgemm_1x1conv`
* `polygeist_cublas_dsyrk`
* `polygeist_cublaslt_matmul_bias_relu`

Stencil, convolution, limiter, pack/unpack, and other non-BLAS symbols still
use the existing CPU reference implementations unless a separate optimized CPU
backend is added for them.

Install OpenBLAS on Ubuntu
--------------------------

For the default OpenBLAS configuration:

```
sudo apt-get update
sudo apt-get install -y libopenblas-dev
```

On this VM, unrelated third-party apt repositories may report warnings during
`apt-get update`. The relevant check is that `cblas.h` and `libopenblas.so`
exist:

```
find /usr/include /usr/local/include -name cblas.h
ldconfig -p | grep openblas
```

Build With OpenBLAS
-------------------

Set `POLYGEIST_CPU_BLAS=1` for host builds:

```
OPENBLAS_NUM_THREADS=1 \
POLYGEIST_CPU_BLAS=1 \
PYTHON=/usr/bin/python3 \
scripts/correctness/polygeist_build.sh \
  issues/proxy_kernel_pipelines/exasp2_pipeline_easy.c \
  --function=exasp2_pipeline_easy \
  --harness=issues/proxy_kernel_pipelines/proxy_pipeline_silicon_large.c \
  --target=host \
  -o /tmp/exasp2_large_raised_host_blas \
  -DEN=512
```

Run:

```
OPENBLAS_NUM_THREADS=1 PROXY_PIPELINE_ITERS=3 \
  /tmp/exasp2_large_raised_host_blas
```

The selected raised function is the only meaningful timing in a raised harness
run. The surrounding harness code is compiled conservatively so it can call the
generated wrapper without inlining the original C function.

Using Another CBLAS Provider
----------------------------

Override compile and link flags with:

```
POLYGEIST_CPU_BLAS=1
POLYGEIST_CPU_BLAS_CFLAGS="<include flags>"
POLYGEIST_CPU_BLAS_LIBS="<library/linker flags>"
```

Examples:

```
POLYGEIST_CPU_BLAS_LIBS="-lblis"
POLYGEIST_CPU_BLAS_LIBS="-L/path/to/mkl/lib -lmkl_rt"
POLYGEIST_CPU_BLAS_CFLAGS="-I/path/to/armpl/include"
POLYGEIST_CPU_BLAS_LIBS="-L/path/to/armpl/lib -larmpl_lp64_mp"
```

Reference Result
----------------

On the current x86_64 Cascade Lake VM, using the ExaSP2 512 proxy pipeline:

```
native C gcc -O3                 225.223896 ms
raised reference CPU shim        467.187299 ms
raised OpenBLAS, 1 thread         19.976757 ms
raised OpenBLAS, 24 threads       52.420795 ms
```

The checksums matched. For this 512-size case, single-threaded OpenBLAS was
faster than 24 OpenBLAS threads because thread startup and synchronization
overhead dominates the relatively small GEMM/GEMV workload.
