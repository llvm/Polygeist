// RUN: rm -rf %t && mkdir -p %t
// RUN: env POLYGEIST_IR_VIEWER_OUT=%t POLYGEIST_SECTION42_RESULTS_DIR=%S/../../issues/polybench_section42 %python %S/../../scripts/correctness/build_ce_viewer.py --polybench-results-only
// RUN: FileCheck %s --input-file=%t/polybench.html
// RUN: test ! -e %t/polybenchgpu.html
// RUN: test ! -e %t/polybench-section42.html

// CHECK: PolyBench four-runtime correctness-gated results.
// CHECK: Clang -O3
// CHECK: <th>datatype</th><th>native CPU runtime</th><th>raised CPU runtime</th><th>native GPU runtime</th><th>raised GPU runtime</th>
// CHECK: external CPU library
// CHECK: PolyBenchGPU CUDA
// CHECK: external CUDA library
// CHECK-DAG: data-filter="passed modified"
// CHECK-DAG: data-filter="failed unavailable"
// CHECK: GEMM datatype comparison
// CHECK: <td><b>FP64</b></td><td>74.829 ms<br><span class="scope">E2E 81.750 ms</span></td><td>40.242 ms
// CHECK: <td><b>FP32</b></td><td>33.011 ms<br><span class="scope">E2E 36.785 ms</span></td><td>1.370 ms
// CHECK: r.dataset.filter.split(" ").includes(v)

// The public PolyBench results view has one row per manifest kernel, an
// explicit datatype, and four primary runtime columns. A separate GEMM table
// preserves the additional FP32 experiment without replacing the FP64 row.
// Obsolete split result pages are removed by the
// results-only build mode so stale hardcoded GPU measurements cannot survive.
// Filter tags can overlap: a normalized external source can independently be
// marked passed or failed as well as modified-source.
