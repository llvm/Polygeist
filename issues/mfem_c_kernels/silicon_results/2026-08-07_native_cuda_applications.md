# Native MFEM CUDA application validation (Jetson Orin)

Date: 2026-08-07

## Scope

This validates that the upstream MFEM applications containing the operator
families covered by our extracted/raised application kernels can execute their
native GPU paths.  It is an application-level CUDA execution check, not yet a
timed raised-vs-native comparison.

MFEM revision: `951cf8886b9c0c33fb36a2f0ede268c8d6a0d8b5`

Hardware/configuration:

- NVIDIA Jetson Orin, MAXN mode
- CUDA 12.6, target `sm_87`
- MFEM `Release`, double precision, `MFEM_USE_CUDA=YES`, `MFEM_USE_MPI=YES`
- One MPI rank; GPU-aware MPI disabled

## Results

All genuine CUDA-capable covered applications executed successfully.

1. `mtop_test_iso_elasticity` -- PASS
   - Covered path: DFEM interpolation -> elasticity quadrature function -> integration
   - Parameters: `-d cuda -quad -dfem -prl 0 -no-vis -no-pv`
   - Size: 576 elements, 5,040 unknowns
   - Linear solve converged in 56 iterations.

2. `dfem-minimal-surface` -- PASS
   - Covered path: DFEM interpolation -> nonlinear quadrature function -> integration
   - Parameters: `-d cuda -o 3 -r 1 -der 1 -no-vis`
   - Newton solve reached relative residual `8.18e-11`.

3. `ex35p`, H1 -- PASS after using a non-degenerate mesh size
   - Covered path: PA diffusion and mass
   - Parameters: `-d cuda -pa -hex -p 0 -o 2 -rs 1 -rp 1 -no-vis`
   - Size: linear system 8,802; LOBPCG converged in 18 iterations.
   - The earlier `-rs 0 -rp 0` smoke case was too small for the requested
     five-mode eigensolve and reported MFEM's `GEVP solver failure`.

4. `ex35p`, H(curl) -- PASS
   - Covered path: PA curl-curl and vector mass
   - Parameters: `-d cuda -pa -hex -p 1 -o 2 -rs 0 -rp 0 -no-vis`
   - LOBPCG converged in 2 iterations.

5. `ex35p`, H(div) -- PASS
   - Covered path: PA div-div and vector mass
   - Parameters: `-d cuda -pa -hex -p 2 -o 2 -rs 0 -rp 0 -no-vis`
   - LOBPCG completed in 26 iterations.

6. `ex9p` -- PASS
   - Covered path: PA mass and DG convection during time stepping
   - Parameters: `-d cuda -pa -rs 0 -rp 0 -o 3 -tf 0.02 -dt 0.01 -no-vis -no-visit -no-paraview`
   - Completed two time steps.

7. `grad_div` -- PASS
   - Covered path: PA div-div
   - Parameters: `-d cuda -lor -rs 0 -rp 0 -o 2`
   - LOR-AMS converged in 20 iterations; reported L2 error `1.1688e-01`.

8. `abs-l1-jacobi`, mass -- PASS
   - Covered path: PA mass
   - Parameters: `-d cuda -a 3 -i 0 -o 3 -rs 1 -rp 0 -ni 30 -no-vis`
   - Converged in 10 iterations; L2 error `8.73866e-04`.

9. `abs-l1-jacobi`, diffusion -- PASS
   - Covered path: PA diffusion
   - Parameters: `-d cuda -a 3 -i 1 -o 3 -rs 1 -rp 0 -ni 30 -no-vis`
   - Converged in 10 iterations; L2 error `1.15796e-03`.

10. `abs-l1-jacobi`, curl-curl -- PASS
    - Covered path: PA curl-curl and vector mass
    - Parameters: `-d cuda -a 3 -i 2 -o 3 -rs 1 -rp 0 -ni 300 -no-vis`
    - Converged in 95 iterations; L2 error `2.40579e-03`.

## Candidate that is not a native CUDA application

`navier_tgv` was built but excluded from the CUDA run set.  At this MFEM
revision it uses PA operators but does not construct an MFEM `Device` or expose
a `-d/--device` option, so the application executes on CPU as written.  Adding
a device option would be an upstream-source modification, not validation of an
existing CUDA application variant.

## Important interpretation

`Device configuration: cuda,cpu` was printed by every passing CUDA run.  These
results prove that the native MFEM application and its covered operator path can
execute with MFEM's CUDA backend.  They do not prove numerical equivalence to
our extracted raised program; that requires a paired harness with identical
inputs, outputs, problem size, and an in-process warm timing loop.
