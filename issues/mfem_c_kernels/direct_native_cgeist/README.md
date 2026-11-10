# Direct upstream MFEM CUDA to cgeist audit

Date: 2026-09-09

## Scope

This is a focused test of one unchanged upstream MFEM CUDA operator, not a
normalized C extraction:

- operator: 2D partial-assembly mass apply
- upstream entry point: `mfem::internal::SmemPAMassApply2D<4,5>`
- element implementation: `SmemPAMassApply2D_Element<4,5,...>`
- source: `third_party/mfem/fem/integ/bilininteg_mass_kernels.hpp`
- MFEM revision: `951cf8886b9c0c33fb36a2f0ede268c8d6a0d8b5`
- benchmark translation unit:
  `issues/mfem_c_kernels/benchmarks/mfem_native_cuda_publication_bench.cpp`
- workload: FP64, `NE=1024`, `D1D=4`, `Q1D=5`, CUDA `sm_87`

No source-name-based recognition, replacement adapter, or rewritten
computational kernel is used here.

## Isolated native CUDA extraction

As a follow-up, `isolated_mass2_native.cu` extracts the same
`SmemPAMassApply2D_Element<4,5,8,true>` computation from the MFEM containers
and dispatch framework. It retains:

- MFEM's `5 x 5 x 8` CUDA block and element mapping through `threadIdx.z`;
- the `BBt`, `sm0`, and `sm1` shared-memory arrays;
- all five block barriers;
- the four forward/diagonal/backward contraction stages;
- MFEM's column-major `DeviceTensor` indexing; and
- accumulating `Y += ...` output semantics.

Only the `Array`, `Vector`, `DeviceTensor`, and `forall_2D_batch` scaffolding is
replaced with flat allocations, equivalent explicit indexing, and a standalone
harness. This is therefore an **extracted upstream native CUDA fixture**, not
an untouched MFEM application and not a new Polygeist library match.

The fixture was cross-compiled on the x86 host with nvcc 12.6.85 for AArch64
and `sm_87`, then the completed executable was transferred to Orin. No source
or CUDA compilation occurred on the board. At `NE=1024`, an independent CPU
reference compared all 16,384 output elements:

- correctness: PASS
- mismatches: 0
- non-finite values: 0
- maximum absolute error: 0
- maximum relative error: 0
- reference and GPU checksum: `1.3971338260835462`

The five synchronized resident spot-check samples were 112.736, 114.464,
114.720, 110.816, and 112.128 us. These are retained only to prove executable
operation. They are not a paper-facing performance result because a full
fixed-clock and idle-board preflight was not performed. The sanitized record
is in `isolated_mass2_native_result.csv`.

## Result

The native half succeeds, but direct cgeist CUDA compilation is not yet an
end-to-end success.

| Stage | Status | Evidence |
|---|---|---|
| Cross-compile unchanged MFEM operator with nvcc 12.6.85 | PASS | AArch64 `sm_87` executable, SHA-256 `c92d965b8f425239bfe934f4bf6c8bf7abfa5ce7669a5e64f1d19cafa380bd78` |
| Execute native CUDA binary on Orin | PASS | exit 0; 20 timed samples; final checksum `-0.10241386693633237` |
| Independent checksum comparison | PASS | expected `-0.10241386693633237`; exact difference 0 |
| Parse the same CUDA translation unit with cgeist | PASS | cgeist exits 0 and emits 2,326 lines of MLIR |
| Recover the operator arithmetic | PARTIAL PASS | emitted lambda body contains the four nested multiply/add contraction stages (`scf.for`, `arith.mulf`, `arith.addf`) |
| Preserve executable CUDA device structure | FAIL | the `gpu.launch` calls a device stub whose final CUDA kernel remains a declaration; the recovered arithmetic is in a separate host/device lambda body and does not retain the original thread/shared-memory synchronization semantics |
| Verify emitted MLIR with `mlir-opt` | FAIL | first error: `invalid memref element type` for a `memref` whose element is an LLVM struct |
| Integrated cgeist CUDA lowering/object generation | FAIL | `error: no CUDA support, aborting`; this cgeist build was not configured with the CUDA emission backend |
| Run a Polygeist-generated version on silicon | BLOCKED | no valid lowered object was produced |

The fresh native run produced 20 samples with minimum 59.744 us, median
61.376 us, maximum 76.128 us, and mean 63.480 us. These samples are a
functional spot check, not a new paper-facing performance row: only one
process was run and fixed-clock/idle-board publication preflight was not
repeated.

## Interpretation

This experiment validates the original qualification precisely:

- cgeist can parse this complex MFEM CUDA instantiation and recover its scalar
  mathematical loop body;
- nvcc can cross-compile the same source and the resulting native executable
  is correct on Orin;
- cgeist cannot yet turn this exact upstream CUDA path into valid, executable
  GPU MLIR in the current build.

Therefore this case must be recorded as **frontend emission/lowering
blocked**, not as a direct upstream MFEM CUDA to Polygeist success. The next
compiler work is structural: connect the translated device computation to the
GPU launch while preserving CUDA thread IDs, shared address-space allocations,
and barriers, then eliminate the illegal LLVM-struct-in-memref representation.

## Toolchain

- Polygeist repository: `f37f06fbb8040ac158449c0e1521142d32fc8726`
- LLVM submodule: `26eb4285b56edd8c897642078d91f16ff0fd3472`
- cgeist: LLVM 18.0.0git, optimized assertions build
- nvcc: CUDA 12.6, V12.6.85
- AArch64 host compiler: GCC 11.4.0

## Reproduction

Run `run.sh` from the repository root. It performs only host-side translation,
verification, and cross-compilation. It does not compile on the Jetson. To run
the completed AArch64 binary through a locally configured silicon profile:

```sh
POLYGEIST_SILICON_PROFILE=<configured SSH hop> \
  scripts/correctness/run_jetson.sh --exe \
  /tmp/mfem-direct-native-cgeist/native-build/bin/mass2_mfem_native_cuda \
  direct_native_mfem_mass2
```

The local silicon profile supplies access information; no credentials or
resolved board endpoints belong in this repository.

Build the isolated native fixture with:

```sh
issues/mfem_c_kernels/direct_native_cgeist/build_isolated_native.sh
```

Its default executable is `/tmp/mfem-isolated-mass2/isolated_mass2_native` and
can be deployed with the same `run_jetson.sh --exe` mechanism.

The follow-up raising audit is documented in `RAISING_RESULT.md`. In short,
the pipeline raises only the final length-five dot reduction and discovers no
complete external-library contraction (`0/1` Linalg bodies matched). The
remaining contraction stages stay behind CUDA thread-stride loops, barriers,
and reused shared-memory buffers.
