# MFEM application hot-operator ports

These harnesses connect the extracted MFEM partial-assembly operators to the
serial examples that exercise the same algorithmic families:

- `ex1_diffusion_3d.c`: Example 1, H1 diffusion.
- `ex3_curlcurl_3d.c`: Example 3, H(curl) curl-curl.
- `ex4_divdiv_3d.c`: Example 4, H(div) div-div.
- `ex9_mass_2d.c` and `ex9_convection_2d.c`: Example 9, DG mass and convection.

Each harness checks one compiler-raised operator application against the
faithful extracted C implementation, then reports warmed per-application time.
The current extracted functions are specialized to `D1D=4`, `Q1D=5`, and
`NE=2`. Consequently these are hot-operator ports, not yet drop-in replacements
for MFEM's arbitrary element batches. Repeating a two-element function call is
also intentionally not presented as an optimized GPU integration: it measures
the launch/transfer cost of the current ABI as well as the library operation.

Run `./run_native.sh` for the faithful-versus-stage-sliced CPU check. Current
measurements and the raised-pipeline blockers are recorded in
`RESULTS_2026_07_31.md`.
