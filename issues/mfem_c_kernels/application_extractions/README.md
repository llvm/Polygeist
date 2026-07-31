# Larger MFEM application C hot paths

These files preserve one concrete numerical operator application from each
larger MFEM example or miniapp. They omit mesh I/O, MPI communication, command
line handling, global restriction/prolongation, and iterative solver control.
Those operations do not contain the tensor-product loops targeted by the
raising pipeline.

The extractions use the corpus specialization `D1D=4`, `Q1D=5`, `NE=2`, and
FP64. `manifest.csv` records the upstream call site and whether the extracted
path is complete. A `partial` entry means the current kernel corpus lacks an
operator used by that application; it does not mean that the extracted C
silently approximates that operator.

Run:

```sh
python3 scripts/correctness/mfem_application_raise_sweep.py
```

Outputs are written to `results/`, including frontend, raised, debufferized,
matched MLIR, per-entry logs, and `summary.csv`.
