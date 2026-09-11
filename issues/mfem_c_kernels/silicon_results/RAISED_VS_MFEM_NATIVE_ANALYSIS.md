# Raised Polygeist vs. native MFEM CUDA: runtime analysis

## Benchmark configuration

- Hardware: NVIDIA Jetson Orin (`sm_87`)
- Power mode: MAXN
- CUDA: 12.6
- Data type: `f64`
- Elements: `NE=1024`
- Basis dimensions: `D1D=4`, `Q1D=5`
- Timing: 20 warm iterations per process; reported raised value is the median
  of process runs 2--4, excluding the first cold CUDA process
- Correctness: all 18 matcher-covered normalized kernels passed checksum and
  maximum-output comparison within floating-point roundoff: ten PA operators
  and eight DFEM interpolation/integration maps

The full measurements are stored in
[`native_vs_raised_large_ne.csv`](native_vs_raised_large_ne.csv).

## Meaning of the measurements

`raised_runtime_us` measures the extracted C kernel after the Polygeist
raising, matching, ABI generation, and CUDA-library lowering pipeline. The
runtime now caches prepared cuTensorNet networks, optimizer state, workspace
descriptors, and scratch allocations by contraction signature. The value still
includes the current compatibility host-pointer ABI, synchronization at host
boundaries, correctness-first tensor snapshots, and unmatched operations that
lower to host loops.

`raised_runtime_us_before_plan_cache` preserves the earlier measurement, and
`plan_cache_speedup` reports the improvement from the new runtime/compiler
path. The gain is between `1.30x` and `3.23x` across the ten operators.

`mfem_native_runtime_us` measures MFEM's existing specialized CUDA
implementation with its input and output tensors resident on the GPU. For the
eight DFEM rows, the benchmark wrapper dispatches MFEM's own inline
tensor-product device routine with one CUDA thread block per element. Each
iteration is synchronized before timing is recorded.

The eight newly covered DFEM maps range from `21.341x` to `200.504x` slower
than native MFEM CUDA. Their individual numbers and the original ten PA
comparisons are stored together in the CSV above.

`raised_over_native` is:

```text
raised_runtime_us / mfem_native_runtime_us
```

Therefore, a value of `58.3` means that the current raised execution is 58.3
times slower than the native MFEM CUDA implementation. A value below one
would mean that the raised implementation is faster.

## Representative result: PA Mass 2D

For `mfem_pa_mass_apply_2d_stage_sliced`:

```text
Raised before:      7.67 ms
Raised with cache:  2.38 ms
MFEM native CUDA: 40.8 us
Cache improvement:  3.23x
Raised/native:      58.3x
```

Both implementations produced the same checksum and maximum absolute output
value within floating-point roundoff.

## What the raised implementation deploys

The original Mass 2D computation has five stages:

1. Interpolate in the x direction.
2. Interpolate in the y direction.
3. Multiply by the quadrature coefficient `D`.
4. Integrate in the x direction.
5. Integrate in the y direction and accumulate into `Y`.

The current matcher converts three of those stages into independent
`@cutensornetContraction2_f64` launches. The pointwise multiplication and
final reduction remain `linalg.generic` operations and are ultimately
compiled as AArch64 host loops.

A simplified view of the deployed execution remains:

```text
cuTensorNet contraction 1: interpolate X in x
synchronize

cuTensorNet contraction 2: interpolate in y
synchronize

host loop: multiply the quadrature tensor by D

cuTensorNet contraction 3: integrate in x
synchronize

host loop: final integration and accumulation into Y
```

The stored matcher result is
[`match_results/mass_apply_2d_stage_sliced/matched.mlir`](../match_results/mass_apply_2d_stage_sliced/matched.mlir).

### Work performed by cuTensorNet contractions

The first call for a new contraction signature performs the expensive setup:

```c
reuse_or_create_global_cutensornet_handle();
cutensornetCreateNetwork(handle, &network);
cutensornetNetworkAppendTensor(...);
cutensornetNetworkAppendTensor(...);
cutensornetNetworkSetOutputTensor(...);

cutensornetCreateContractionOptimizerConfig(...);
cutensornetCreateContractionOptimizerInfo(...);
cutensornetContractionOptimize(...);

cutensornetCreateWorkspaceDescriptor(...);
cutensornetWorkspaceComputeContractionSizes(...);
cudaMalloc(scratch);
cutensornetNetworkPrepareContraction(...);

cache[signature] = {network, optimizer, workspace, scratch};
```

Every later call with the same device, ranks, extents, strides, and modes does
only:

```c
entry = cache_lookup(signature);
register_or_find_mapped_host_buffers(A, B, C);
cutensornetNetworkSetInputTensorMemory(...);  // rebind A and B
cutensornetNetworkSetOutputTensorMemory(...); // rebind C
cutensornetNetworkContract(...);
sync_only_at_the_end_of_the_safe_GPU_region();
```

The cache has 64 LRU entries and is enabled by default. It can be disabled
with `POLYGEIST_CUTENSORNET_PLAN_CACHE=0`; cache statistics are printed when
`POLYGEIST_RT_CACHE_STATS=1`.

The implementation is in
[`runtime/polygeist_cublas_rt_cuda.c`](../../../runtime/polygeist_cublas_rt_cuda.c),
in `polygeist_cutensornet_contraction2_f64`.

### Historical setup time versus GPU time

Runtime instrumentation for a warmed Mass 2D iteration reported representative
values of:

```text
Contraction 1: host 2.07 ms, GPU 0.030 ms
Contraction 2: host 2.13 ms, GPU 0.032 ms
Contraction 3: host 2.16 ms, GPU 0.036 ms
```

The three contractions perform approximately `0.10 ms` of actual GPU work,
but spend approximately `6.3 ms` in their host-side call paths. The rest of
the approximately `7.5-7.7 ms` execution time comes from synchronization,
intermediate tensor handling, and the two residual host stages.

Thus, cuTensorNet's mathematical contraction was not intrinsically taking
multiple milliseconds. The new cache removes repeated reconstruction,
optimization, preparation, and destruction. For example, a Mass 2D process
reports `3` misses followed by `63` hits. The remaining `2.38 ms` is dominated
by host/device boundaries, output snapshots, synchronization, separate
launches, and residual host stages.

### Host-memory behavior

The current Jetson runtime normally uses `cudaHostRegister` and
`cudaHostGetDevicePointer` to expose host buffers as mapped memory. It caches
registrations for reuse. Consequently, repeated bulk `cudaMemcpy` transfers
are not the main cause of this result.

The remaining memory-related costs are mapped-host access, registration on
first use or cache replacement, temporary allocation, and the CPU/GPU
boundaries caused by residual host stages.

## What native MFEM deploys

For Mass 2D, MFEM invokes one specialized launcher resembling:

```c++
mfem::forall_2D_batch(NE, Q1D, Q1D, NBZ,
  [=] MFEM_HOST_DEVICE (int e) {
    internal::SmemPAMassApply2D_Element<4, 5, NBZ>(
        e, NE, B, D, X, Y);
  });
```

Inside that one device kernel, MFEM performs all five stages:

```c++
MFEM_SHARED real_t B_and_Bt[...];
MFEM_SHARED real_t scratch0[...];
MFEM_SHARED real_t scratch1[...];

load B and X into shared memory;

DQ = B * X;                 // interpolation in x
QQ = B * DQ;                // interpolation in y
QQ *= coefficient_D;        // quadrature pointwise stage
QD = transpose(B) * QQ;     // integration in x
Y += transpose(B) * QD;     // integration in y
```

The relevant MFEM implementation is in:

- [`SmemPAMassApply2D`](../../../third_party/mfem/fem/integ/bilininteg_mass_kernels.hpp)
- `SmemPAMassApply2D_Element` in the same file

MFEM therefore benefits from:

- One CUDA kernel launch instead of three library calls and two host stages.
- No runtime contraction-network optimizer.
- No per-stage plan creation or destruction.
- GPU-resident input and output tensors.
- Intermediate values retained in registers and shared memory.
- Compile-time specialization for `D1D=4` and `Q1D=5`.
- Loop unrolling and a launch configuration tailored to the finite-element
  operator.
- No global-memory materialization between the five mathematical stages.

## Execution structure comparison

```text
Raised Polygeist (current cached compatibility ABI)
---------------------------------------------------
cached plan lookup + pointer rebind
  -> GPU contraction
  -> synchronize
  -> host-visible output snapshot
cached plan lookup + pointer rebind
  -> GPU contraction
  -> synchronize
  -> host-visible output snapshot
host pointwise loop
cached plan lookup + pointer rebind
  -> GPU contraction
  -> synchronize
  -> host-visible output snapshot
host reduction/accumulation loop

MFEM native CUDA
----------------
one specialized CUDA launch
  -> interpolation
  -> coefficient application
  -> integration
  -> output accumulation
```

## Main causes of the slowdown

In descending order of importance for the current measurements:

1. **Loss of fusion.** One finite-element operator becomes multiple library
   launches plus residual host loops.
2. **CPU/GPU boundaries.** Unmatched pointwise and reduction stages execute on
   the host between GPU contractions.
3. **Correctness-first host snapshots and synchronization.** The current
   opaque pointer call does not model its tensor write for bufferization, so
   each output is copied to a fresh host-visible tensor. These host operations
   correctly terminate an asynchronous GPU region.
4. **Materialized intermediate tensors.** Values that MFEM keeps in shared
   memory become standalone tensors visible across calls.
5. **A general-purpose library is being used for very small fixed
   contractions.** MFEM's specialized `4x5` tensor-product kernel has much
   less machinery and exposes more compile-time optimization opportunities.

Repeated per-signature cuTensorNet planning was previously the largest fixed
overhead. It is now removed on cache hits, but the structural costs above
remain.

Both implementations use `f64`, so the Jetson's relatively limited FP64
throughput is not the primary explanation for the ratio. The large difference
comes from execution structure and runtime overhead.

## Improvements needed

### Implemented: cache and reuse library state

- One cuTensorNet handle is reused.
- Network descriptors and optimized plans are cached by contraction signature.
- Prepared workspaces and scratch allocations remain live in the cache.
- Input/output pointers are rebound on each cache hit.
- Cache entries are destroyed at explicit runtime teardown or process exit.

This produced `1.30x`--`3.23x` end-to-end speedups, but does not eliminate
multiple launches or global intermediates.

### Implemented foundation: safe pipeline scopes and device-pointer ABI

- `WrapKernelLaunchPipeline` now forms maximal block-local GPU-only regions.
  It defers synchronization across library calls and metadata/view operations,
  but ends the region before any host tensor computation.
- `polygeist_cutensornet_contraction2_f64_device` accepts CUDA device pointers
  directly and skips host registration/mapping.
- `--lower-kernel-launch-to-cublas=device-resident-cutensornet=true`, exposed
  by `POLYGEIST_DEVICE_RESIDENT_ABI=1` in the build script, selects this ABI.
- The compiler rejects this mode if residual `linalg`, loops, host memref
  accesses/copies, tensor element accesses, or non-device-produced operands
  remain. It therefore fails closed instead of treating a host pointer as a
  device pointer.

The device runtime ABI passed a Jetson smoke test: a device-resident `2x3` by
`3x2` contraction produced `[58, 64, 139, 154]` exactly, with one cache miss
and one hit. Current MFEM stage pipelines intentionally fail the legality gate
because their residual pointwise/reduction stages and snapshot copies still
execute on the host. Making the whole raised graph device-resident requires
lowering those residual stages to GPU library operations and replacing opaque
call snapshots with a bufferizable library-call representation.

### Performance target: fuse the recognized stage graph

The long-term transformation should be:

```text
recognized interpolation
  -> recognized quadrature operation
  -> recognized integration
  -> one fused GPU implementation
```

The raising and matching work already recovers the stage semantics. The
remaining compiler work is to preserve the complete dependency graph, select
fusion boundaries, and lower the graph to one optimized implementation rather
than independently lowering each `linalg.generic` operation.

Plan caching has made the present approach substantially less slow. Matching
native MFEM performance, however, still requires fusion and GPU-resident
intermediates.
