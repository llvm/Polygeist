# Mixed generated-kernel and CUDA-library graph validation

Date: 2026-08-19

## Result

Polygeist can now execute a compiler-generated GPU kernel and CUDA library
operations in one ordered pipeline and one reusable CUDA Graph. The generated
kernel uses the standard `mgpu*` ABI emitted by MLIR's GPU-to-LLVM conversion;
it does not need a handwritten per-kernel runtime wrapper.

The validated sequence is:

```text
CUDA memset
  -> cuBLAS DAXPY
  -> generated scale kernel (loaded from cubin through the MLIR mgpu ABI)
  -> cuBLAS DAXPY
```

All operations use the same runtime-owned nonblocking CUDA stream. Module and
function handles are populated during warmup, capture records only device
work, and subsequent calls replay the graph with stable device pointers.

## Silicon

- Device: Jetson AGX Orin (`nvidia@192.168.58.1`, via `pva-general`)
- CUDA: 12.6
- Problem: `N=4096`, 5,000 iterations per process
- Repetitions: 3 independent processes
- Correctness: PASS in every run, maximum absolute error `1.11e-16`

Warm per-iteration times (microseconds):

```text
normal dispatch: 37.554643, 38.456909, 38.288979
CUDA Graph:      24.334182, 25.572288, 25.783437
```

Using medians, graph replay is `38.288979 / 25.572288 = 1.497x` faster and
removes 33.2% of the launch-bound sequence time.

Large-data correctness was also checked with `N=1,048,576`, 20 iterations,
three processes; every run passed with zero measured error in the direct CUDA
wrapper fixture.

Logs:

- `scripts/correctness/logs/mixed_mgpu_no_graph_exact_abi_20260819_100526.silicon.log`
- `scripts/correctness/logs/mixed_mgpu_graph_exact_abi_20260819_100517.silicon.log`
- `scripts/correctness/logs/mixed_cuda_graph_generated_library_20260819_20260819_095343.silicon.log`

## Compiler and runtime changes

1. `WrapKernelLaunchPipeline` recognizes graph-safe `gpu.launch_func`
   operations and arbitrary declarations/calls carrying
   `polygeist.cuda_graph_safe`. It forms one maximal scope containing generated
   launches and library calls.
2. `polygeist_cuda_graph_stream()` exposes the runtime-owned stream through a
   CUDA-independent header.
3. The CUDA runtime implements MLIR's `mgpuModuleLoad`,
   `mgpuModuleGetFunction`, `mgpuLaunchKernel`, and stream ABI. Cubin/function
   handles are cached and unload is deferred to runtime teardown.
4. CUDA Driver API symbols are resolved from `libcuda.so.1` lazily, preserving
   the existing `cudart`-based link contract of current executables.

## Validation coverage and remaining work

The compiler regression checks both an arbitrary generated wrapper call and a
real `gpu.launch_func` between CUDA-library calls. The standalone MLIR fixture
also lowers its `gpu.func` body through GPU-to-NVVM successfully.

The current local LLVM build does not include the NVPTX target, so it cannot
serialize that NVVM module to cubin locally. Silicon therefore validates the
exact resulting `mgpu*` runtime ABI with an equivalent cubin produced by nvcc.
Enabling LLVM's NVPTX target will make the final serialization step fully
MLIR-native without changing the runtime ABI.

This change supplies the mixed graph execution layer. It does not yet discover
and partition every complete MFEM application graph automatically. The next
compiler layer must keep the raised semantic graph connected, choose library
subgraphs versus generated residual partitions, and annotate the selected
generated launches before this lowering runs.

## Larger MFEM application extraction

The graph path was subsequently tested on the extracted complete Mass3D
operator branch from MFEM's `abs-l1-jacobi` miniapp. This is the existing
`mfem_app_abs_l1_mass_3d` larger-application benchmark, not the small isolated
kernel fixture.

- Problem: f64, `NE=1024`, `D1D=4`, `Q1D=5`
- Raised operation: one composed eight-input cuTensorNet network
- Timing: one untimed warmup and 20 warm iterations per process
- Repetitions: five independent processes
- Correctness: PASS in all ten comparison processes, maximum absolute and
  relative error `6.9388939039072284e-18`

Using the median of all five process results:

```text
same raised executable, graph disabled: 655.124802 us
same raised executable, graph enabled:  549.356802 us
graph speedup:                            1.1925x
graph time reduction:                    16.14%
native MFEM resident CUDA baseline:      137.964800 us
raised graph / native:                     3.982x slower
same-run scalar CPU reference:           770.934392 us
raised graph / scalar CPU:                 1.403x faster
```

This confirms that the graph infrastructure works on a larger MFEM-derived
operator and removes a measurable part of the launch/dispatch cost. It does
not close the remaining native gap: one general cuTensorNet network still
moves intermediates through global memory and carries more general planning
and execution overhead than MFEM's fused element-local CUDA kernel.

Application logs:

- `scripts/correctness/logs/mfem_abs_mass_complete_graph_disabled_20260819_101915.silicon.log`
- `scripts/correctness/logs/mfem_abs_mass_complete_graph_wrapped_20260819_101903.silicon.log`

## 100x-slow MFEM kernel test

The exact extracted `abs_l1_diffusion_3d` operator was tested at f64,
`NE=1024`, `D1D=4`, `Q1D=5`. Its raised path contains 14 pairwise
cuTensorNet contractions and is representative of the severe multi-launch
cases.

With graph replay disabled, all five processes passed with maximum
absolute/relative error `6.9388939039072284e-18`. The median raised time was
`44802.444801 us`; the resident native MFEM CUDA baseline is `304.099200 us`,
so this current raised executable is `147.33x` slower.

Graph capture is not yet legal for this operator. Capture aborts with:

```text
cudaHostRegister(..., 655360) failed:
operation not permitted when stream is capturing
```

The reason is structural rather than a cuTensorNet incompatibility. This
raised function creates large scratch tensors on every invocation. Their host
addresses differ from the warmup invocation, so the runtime encounters an
unregistered scratch pointer while recording the next invocation. CUDA does
not permit host registration during stream capture. In addition, host-side
metadata construction between contractions currently causes the compiler to
form 14 separate one-call graph scopes instead of one 14-call graph.

This test identifies the requirements for applying graph replay to the 100x
bucket:

1. Hoist scratch allocation to persistent caller-owned or device-resident
   storage so every replay sees identical addresses.
2. Precompute/register buffer mappings and tensor metadata before capture.
3. Form one graph scope around the complete 14-call device sequence after ABI
   preparation, rather than one graph per library call.
4. Keep residual generated kernels and library calls on the same runtime
   stream inside that scope.

No graph performance number is reported because capture did not complete.

Logs:

- `scripts/correctness/logs/mfem_abs_diff_170x_no_graph_20260819_102705.silicon.log`
- `scripts/correctness/logs/mfem_abs_diff_170x_graph_20260819_102651.silicon.log`

## Persistent-workspace follow-up for Diffusion3D

The scratch-address blocker was subsequently fixed with the opt-in
`--plan-persistent-gpu-workspace=function=...` pass. It replaces eligible
static tensor and memref scratch roots with separate 4096-byte-aligned private
`memref.global` workspaces. The Jetson build pipeline runs standard shaped-dim
resolution first, which makes ABI compatibility snapshots with constant
runtime dimensions statically recognizable, and applies the planner again
after submap lowering. The policy is deliberately single-instance and
non-reentrant; caller-owned workspace remains the long-term ABI.

Two additional correctness fixes were required:

1. cuTensorNet ABI lowering now forwards a direct launch-result consumer to
   the output view that the library updated in place, rather than supporting
   only a terminal `tensor.insert_slice` user.
2. Host mappings are cached at page granularity, and compiler-created global
   workspaces are page-aligned so unrelated scratch objects cannot share a
   CUDA registration page.

At f64, `NE=1024`, `D1D=4`, `Q1D=5`, five independent processes passed in
both configurations with max absolute/relative error
`6.9388939039072284e-18`:

```text
original per-invocation scratch, graph disabled: 44802.444801 us
persistent scratch, graph disabled:              12249.702401 us
persistent scratch, graph enabled:               11705.311947 us
speedup versus original raised path:                  3.8275x
graph-only speedup on persistent path:                1.0465x
native resident MFEM CUDA baseline:                 304.099200 us
persistent graph / native:                           38.4918x slower
```

Thus the address and capture failures are fixed, but launch replay is not the
dominant remaining optimization. The residual three quadrature
`linalg.generic` computations still execute as CPU loops, compatibility
snapshots/copies remain, and the 14 contractions are captured as separate
one-call graphs. An experimental distribution of the residual sum-product
body into nine cuTensorNet networks was rejected after silicon correctness
failed (`max_abs=4.1607504755165935e-4`); it is not part of the production
rewrite.

Passing logs:

- `scripts/correctness/logs/mfem_abs_diff_persistent_graph_static_snapshots_20260819_20260819_104957.silicon.log`
- `scripts/correctness/logs/mfem_abs_diff_persistent_nograph_20260819_20260819_105011.silicon.log`

## Complete residual-GPU graph follow-up

The remaining mixed pipeline has now been lowered end to end. A new
`--prepare-gpu-residual-pipeline` bridge converts the residual bufferized
Linalg stages and compatibility copies to generated GPU kernels, registers
the application and persistent-workspace buffers, and marks those launches as
graph-safe. `WrapKernelLaunchPipeline` can then form a maximal device sequence
across generated launches, tensor-library calls, and capture-invariant
metadata.

The Jetson ABI required one additional general fix. MLIR reports a logical
operand count to `mgpuLaunchKernel`, but a ranked memref expands to many PTX
parameters. The runtime now enumerates the actual CUDA parameter layout with
`cuFuncGetParamInfo` and translates every registered host pointer to its
mapped device address. This matters on the alternate Orin, where host and GPU
virtual addresses are not identical.

For `abs_l1_diffusion_3d`, the final device sequence is:

```text
one CUDA Graph scope
  34 compiler-generated GPU launches
  14 cuTensorNet contraction calls
  0 residual host computations
  0 host-side memref.copy operations
```

Static scratch hoisting is essential: without it, 48 separate graph scopes
are required because allocations occur between device operations. With the
persistent-workspace pass there is exactly one begin/end graph scope in the
compiler IR. Warmup executes once, the second call captures, and all measured
calls replay the resulting graph.

Five independent processes were run on Jetson AGX Orin at f64, `NE=1024`,
`D1D=4`, `Q1D=5`, with 20 warm replays per process. All passed with maximum
absolute/relative error `6.9388939039072284e-18`.

```text
raised warm times (us): 14628.297603, 14631.734393, 14628.470386,
                        14597.697603, 14593.115193
raised median:          14628.297603 us
same-run CPU median:     3309.452813 us
native resident CUDA:     304.099200 us
raised / native:            48.104x slower
```

This completes the execution/capture work, but does not make CUDA Graphs fuse
kernel bodies. The 23 semantic compatibility copies are now device kernels
rather than host loops, and the residual pointwise/reduction stages are also
device kernels; they still read and write global memory. The native MFEM
kernel keeps the element-local pipeline inside one hand-fused CUDA kernel.
Closing the remaining gap therefore requires eliminating compatibility
materializations and fusing generated residual stages with adjacent tensor
operations, not more graph wrapping.

Passing log:

- `scripts/correctness/logs/mfem_abs_diff_complete_residual_gpu_graph_20260819.silicon.log`

## Retained semantic destination experiment

The contraction ABI can now consume the result-free memref form produced by
One-Shot Bufferize. `kernel.launch` already implements
`BufferizableOpInterface`; the missing piece was that
`lowerCutensornetContraction2F64` only accepted one tensor result and therefore
forced Diffusion3D through the legacy opaque-call snapshot path. The lowering
now accepts either one tensor result or a bufferized destination launch and,
for the latter, emits the runtime call directly against the proven destination
buffer without recreating a tensor snapshot.

Additional general changes support the complete path:

1. `polygeist-opt` registers all standard bufferization passes, including
   `--empty-tensor-to-alloc-tensor`.
2. cuTensorNet view metadata peels a bufferization
   `to_memref(polygeist.submap(...))` boundary and recovers the original
   tensor-view provenance. This points the library descriptor at the flat ABI
   buffer and lets dead view materializations disappear.
3. Legal non-broadcast flat tensor submaps can become zero-copy strided
   `memref.reinterpret_cast` views. Zero-stride broadcasts remain semantic
   metadata because MLIR memref layouts prohibit zero strides.
4. The GPU residual preparation pass removes terminal
   `dst -> temporary -> dst` bufferization round trips, including values
   forwarded through `scf.for` iter_args.
5. CUDA Graph maximal-sequence analysis recognizes
   `memref.extract_strided_metadata` as capture-invariant descriptor work.

On the Diffusion3D IR this changes the top-level device structure from:

```text
legacy complete graph:   34 generated launches + 14 cuTensorNet calls
retained destination:    23 generated launches + 14 cuTensorNet calls
compatibility copies:    23 -> 1
CUDA Graph scopes:       1 -> 1
```

Five independent Jetson processes passed with maximum error
`6.9388939039072284e-18`. Warm raised times were:

```text
15020.823991, 15017.368016, 15020.931186, 15049.446397, 15022.779186 us
median: 15020.931186 us
```

This is 2.68% slower than the prior `14628.297603 us` complete-graph median,
despite eliminating 22 copies. The result is therefore retained as compiler
infrastructure but is not enabled automatically for this application. On the
unified-memory Orin, those copies were cheaper than the remaining general
view/residual kernels and cuTensorNet execution; copy count was not the
dominant cost.

The semantic reason complete cuTensorNet composition stops is also now
explicit. Each of the three quadrature generics computes:

```text
out += coefficient * (a0*b0 + a1*b1 + a2*b2)
```

with a reduction over the finite-element component mode. A cuTensorNet
network represents a product of tensor factors followed by contraction; it
does not represent an arbitrary sum of three independent product networks.
These stages can be implemented as generated fused kernels, or decomposed
into multiple accumulating library contractions, but cannot legally become
one cuTensorNet product network. Likewise, a CUDA Graph schedules the 37
top-level operations but cannot inline their kernel bodies.

Passing log:

- `scripts/correctness/logs/mfem_abs_diff_retained_destination_graph_20260819.silicon.log`
