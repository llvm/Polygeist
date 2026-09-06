// RUN: rm -rf %t && mkdir -p %t
// RUN: env POLYGEIST_IR_VIEWER_OUT=%t POLYGEIST_SECTION42_RESULTS_DIR=%S/../../issues/polybench_section42 %python %S/../../scripts/correctness/build_ce_viewer.py --polybench-results-only
// RUN: FileCheck %s --input-file=%t/polybench.html
// RUN: test ! -e %t/polybenchgpu.html
// RUN: test ! -e %t/polybench-section42.html

// CHECK: PolyBench four-runtime correctness-gated results.
// CHECK: <th>native CPU runtime</th><th>raised CPU runtime</th><th>native GPU runtime</th><th>raised GPU runtime</th>
// CHECK: Clang -O3
// CHECK: external CPU library
// CHECK: PolyBenchGPU CUDA
// CHECK: external CUDA library
// CHECK: data-filter="passed modified"
// CHECK: data-filter="failed unavailable modified"
// CHECK: r.dataset.filter.split(" ").includes(v)

// The public PolyBench results view has one row per manifest kernel and four
// primary runtime columns. Obsolete split result pages are removed by the
// results-only build mode so stale hardcoded GPU measurements cannot survive.
// Filter tags can overlap: a normalized external source can independently be
// marked passed or failed as well as modified-source.
