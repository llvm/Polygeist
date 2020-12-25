# Polybench Result

## Target

Two configurations of Polybench C code are properly converted to MLIR code: `SMALL` for sanity check, and `EXTRALARGE` for performance evaluation.

The conversion is done by the front-end implemented in wsmoses/MLIR-GPU.

## Sanity check

```shell
./sanity-check SMALL tmp
```

This will check whether the output are the same among the original MLIR code, SCoP statements extracted MLIR code (after `-reg2mem -extract-scop-stmt` passes in Polymer), and the Pluto optimized MLIR code (after `-reg2mem -extract-scop-stmt -pluto-opt -canonicalize`).

```
                               Benchmark       Exit code       Scop Diff      Pluto Diff
        --------------------------------   -------------   -------------   -------------
                      SMALL/deriche.mlir               0               0               0
                   SMALL/covariance.mlir               0               0               0
               SMALL/floyd-warshall.mlir               0               0               0
                  SMALL/gramschmidt.mlir               0               0               0
                     SMALL/cholesky.mlir               0               0               0
                     SMALL/nussinov.mlir               0               0               0
                         SMALL/atax.mlir               0               0               0
                      SMALL/fdtd-2d.mlir               0               0               0
                      SMALL/gesummv.mlir               0               0               0
                      SMALL/doitgen.mlir               0               0               0
                  SMALL/correlation.mlir               0               0               0
                         SMALL/trmm.mlir               0               0               0
                          SMALL/2mm.mlir               0               0               0
                         SMALL/syrk.mlir               0               0               0
                           SMALL/lu.mlir               0               0               0
                          SMALL/adi.mlir               0               0               0
                          SMALL/mvt.mlir               0               0               0
                    SMALL/jacobi-2d.mlir               0               0               0
                         SMALL/symm.mlir               0               0               0
                    SMALL/seidel-2d.mlir               0               0               0
                      SMALL/trisolv.mlir               0               0               0
                       SMALL/ludcmp.mlir               0               0               0
                          SMALL/3mm.mlir               0               0               0
                       SMALL/gemver.mlir               0               0               0
                         SMALL/bicg.mlir               0               0               0
                    SMALL/jacobi-1d.mlir               0               0               0
                        SMALL/syr2k.mlir               0               0               0
                         SMALL/gemm.mlir               0               0               0
                      SMALL/heat-3d.mlir               0               0               0
                       SMALL/durbin.mlir               0               0               0

                                           Report:
                                      Total cases:    30
                                 Successful cases:    30
```

## Performance evaluation

`eval-perf` compares the performance between the C code optimized by Pluto and the MLIR code optimized by Polymer. The optimization options is controlled by command line arguments.

```
$ ./eval-perf -h

    Performance evaluator for Pluto and Polymer.

Usage: 
    -h                  Display this help message
    -d                  Delete the working directory when finished
    -s                  Skip the actual run
    -g                  Skip the Pluto and Polymer codegen
    -v                  Enable vectorization by Clang
    -u                  Enable loop unroll by Clang
    -p                  Enable parallelization
    -f <target>         Evaluate the <target> file
    -t <pluto var>      Specific Pluto variant (i64, 1d, etc.)
```

You can find the run results under the `tmp/` directory.

We normally run against the EXTRALARGE directory.

### Examples

Run the default setting.

```sh
./eval-perf -f EXTRALARGE 
```

Use i64 induction variables for Pluto

```sh
./eval-perf -f EXTRALARGE -t i64
```

Use the double -O3 setting (use -O3 for LLVM IR emission and executable codegen)

```sh
./eval-perf -f EXTRALARGE -c 2
```

Allow vectorization and unrolling (normally come together with double -O3).

```sh
./eval-perf -f EXTRALARGE -c 2 -v -u
```
