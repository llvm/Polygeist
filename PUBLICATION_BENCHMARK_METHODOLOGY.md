# Publication Benchmark Methodology

This document is the canonical measurement protocol for paper-facing
PolyBench, ATen, MFEM, Llama, PVA, and related benchmark results. Historical
measurements that do not satisfy this protocol may be retained, but they must
be labelled as historical or superseded rather than mixed into the publication
set.

## Paper methodology paragraph

For each workload and implementation, we compile 1 executable from pinned
source, compiler, and library revisions, since our compilation pipeline is
deterministic, and evaluate compilation variability separately. We run each
executable in 1 process, with 5 untimed warm-up iterations followed by 5
synchronized timed iterations. We report the median of the 5 measurements,
together with their interquartile range and minimum and maximum values; results
are accepted only when the complete output passes the declared correctness
criterion. Our primary evaluation runs native CPU, raised
CPU-library, native accelerator, and raised accelerator implementations on the
same Jetson AGX Orin, using identical inputs, shapes, datatypes, and output
semantics. All AArch64 executables are cross-compiled on a 24-core Intel
Cascade Lake x86-64 host with approximately 64 GB of RAM and deployed to the
Orin, while existing x86 measurements are retained as a separately labelled
portability comparison. CPU execution is pinned to 1 declared core with
library threading fixed to 1, and GPU and PVA measurements use fixed power,
clock, and fan settings. We report synchronized steady-state resident time as
the common comparison metric, excluding allocation, library initialization,
plan construction, and initial and final data transfers; CUDA-event device
time and PVA submit-to-completion time are reported separately, while a
secondary end-to-end measurement includes mandatory transfers, dispatch, and
synchronization. For the equality-saturation ablation and compilation-cost
study, we compile each input 5 times with Egglog enabled and 5 times with it
disabled, use a 10-second saturation limit per candidate, and record
compilation time, saturation time, peak memory consumption, e-graph size,
iteration count, recovered external-library calls, and timeout outcomes.

## Required execution protocol

- Primary hardware: use the single selected Jetson AGX Orin configured through
  the local silicon-runner profile. Record a non-secret board identifier in
  retained campaign metadata, but keep access details out of the repository.
- Build placement: cross-compile all AArch64 CPU, CUDA, and PVA artifacts on
  the x86 development host. Do not compile project code on the Jetson.
- Runtime repetitions: use 1 process with 5 warm-ups and 5 synchronized timed
  iterations for every workload/implementation configuration.
- Aggregation: take the median of the 5 timed iterations and retain their
  minimum, maximum, and interquartile range. Do not substitute a best-of-N
  timing or describe these
  within-process samples as cross-process variability.
- CPU configuration: pin the primary result to 1 declared Orin CPU core and
  force BLAS/OpenMP libraries to 1 thread. Any all-core result is secondary and
  must be labelled separately.
- Accelerator configuration: fix and record the Orin power mode, clocks, fan,
  temperature, software versions, and throttling state. Run CPU, GPU, and PVA
  configurations sequentially and alternate their order across workloads to
  reduce systematic thermal-order bias.
- Common headline scope: synchronized host wall time around the steady-state
  resident logical operation or application graph. Buffers, handles,
  descriptors, and plans are created and warmed before timing.
- Backend breakdowns: report CUDA-event device time and PVA
  submit-to-completion wall time separately. Never label synchronized PVA wall
  time as device-only time.
- Secondary end-to-end scope: include mandatory transfers, dispatch, and
  synchronization, but exclude compilation and process startup. Never combine
  resident and end-to-end values in one ratio.
- Correctness: compare complete outputs when practical, use exact equality for
  integer results and declared `atol`/`rtol` for floating point, and explicitly
  reject mismatched NaNs and infinities. Correctness work stays outside the
  timed region.
- Provenance: retain exact commands plus source, compiler, library, harness,
  executable, and input hashes for every publication row.
- External implementation requirement: count a match only when it lowers to a
  pre-existing external library or platform API. Analysis-only matches and
  project-authored computational kernels do not count.

## Secondary x86 results

Keep the existing Intel Cascade Lake results as separately labelled
portability and related-system evidence. Polly and KernelFaRer comparisons
remain x86-only. Do not combine x86 CPU and Orin accelerator measurements into
a same-hardware speedup.

## Equality-saturation and compilation-cost experiment

For each input, run 10 fresh compiler invocations: 5 with Egglog enabled and 5
with it disabled. Use identical compiler revisions, inputs, machines, and
resource limits. Record total compilation time, raising time, matching time,
saturation time, peak RSS, e-graph nodes and equivalence classes, iteration
count, executable external-library matches, and timeout outcome. Unless the
paper explicitly states otherwise, apply a 10-second saturation limit to each
candidate. Report timeouts and unmatched inputs directly; do not recover
ablation matches through duplicated hardcoded semantic matchers.
