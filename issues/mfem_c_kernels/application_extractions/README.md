# Larger MFEM application C hot paths

These files preserve one concrete numerical operator application from each
larger MFEM example or miniapp. They omit mesh I/O, MPI communication, command
line handling, global restriction/prolongation, and iterative solver control.
Those operations do not contain the tensor-product loops targeted by the
raising pipeline.

The extractions use FP64 with `D1D=4` and `Q1D=5`. The element count defaults
to `NE=2` for fast correctness tests and can be changed at compile time with
`-DMFEM_BENCH_NE=<count>`. Pointer extents, scratch tensors, output checks, and
the ABI wrapper scale with this value. `manifest.csv` records the upstream
call site, coverage, and every
represented operator family. Solver convergence, MPI communication, mesh
handling, and global restriction/prolongation remain application orchestration
rather than element tensor kernels. The ex9p entry includes one complete PCG
algebra iteration; its data-dependent convergence loop remains outside the
extraction.

Run:

```sh
python3 scripts/correctness/mfem_application_raise_sweep.py
```

Outputs are written to `results/`, including frontend, raised, debufferized,
matched MLIR, per-entry logs, and `summary.csv`.
