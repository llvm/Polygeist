// RUN: /usr/bin/python3 %S/../../scripts/correctness/egglog_library_variants.py \
// RUN:   --source mlir --symbol cublasDgemm --output %t.csv --timeout 5 --iterations 8 --jobs 4 \
// RUN:   | FileCheck %s
// RUN: FileCheck %s --check-prefix=CSV < %t.csv

// CHECK: definitions=1 formulas=2 cases=10 match=10 no_match=0 timeout=0 error=0
// CSV: source,canonical_origin,symbol,body_index,variant,canonical_nodes,variant_nodes,iterations,timeout_s,status,proof_elapsed_ms,wall_elapsed_ms,rule_matches,detail
// CSV-DAG: mlir,mlir,cublasDgemm,0:0,commute_all,{{[0-9]+}},{{[0-9]+}},8,5.0,match,
// CSV-DAG: mlir,mlir,cublasDgemm,0:0,associate_left,{{[0-9]+}},{{[0-9]+}},8,5.0,match,
// CSV-DAG: mlir,mlir,cublasDgemm,0:0,associate_right,{{[0-9]+}},{{[0-9]+}},8,5.0,match,
// CSV-DAG: mlir,mlir,cublasDgemm,0:0,add_zero,{{[0-9]+}},{{[0-9]+}},8,5.0,match,
// CSV-DAG: mlir,mlir,cublasDgemm,0:0,mul_one,{{[0-9]+}},{{[0-9]+}},8,5.0,match,

// This file intentionally contains no IR. The audit reads the production
// kernel library and proves five generated variants of both GEMM bodies.
