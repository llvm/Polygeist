// RUN: rm -rf %t && mkdir -p %t
// RUN: printf 'suite,input_id,path\npolybench,fixture,%s\n' %s > %t/source.csv
// RUN: %python %S/../../scripts/correctness/generate_eqsat_ir_variants.py --manifest %t/source.csv --output %t/out --mlir-opt polygeist-opt --variants swap_add add_zero_rhs reassociate_add | FileCheck %s --check-prefix=SUMMARY
// RUN: FileCheck %s --check-prefix=ZERO < %t/out/inputs/fixture__add_zero_rhs.mlir
// RUN: FileCheck %s --check-prefix=ASSOC < %t/out/inputs/fixture__reassociate_add.mlir
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %t/out/inputs/fixture__add_zero_rhs.mlir --dry-run --matcher-mode egglog --disable-semantic-fallback 2>&1 | FileCheck %s --check-prefix=EGG
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %t/out/inputs/fixture__add_zero_rhs.mlir --dry-run --matcher-mode syntactic --disable-semantic-fallback 2>&1 | FileCheck %s --check-prefix=SYN

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @fixture(%alpha: f64, %a: tensor<4x4xf64>, %b: tensor<4x4xf64>, %c: tensor<4x4xf64>) -> tensor<4x4xf64> {
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%a, %b : tensor<4x4xf64>, tensor<4x4xf64>) outs(%c : tensor<4x4xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %1 = arith.mulf %alpha, %in : f64
      %2 = arith.mulf %1, %in_0 : f64
      %3 = arith.addf %out, %2 : f64
      %4 = arith.addf %3, %out : f64
      linalg.yield %3 : f64
    } -> tensor<4x4xf64>
    return %0 : tensor<4x4xf64>
  }
}

// SUMMARY: "campaign_inputs": 4
// SUMMARY: "generated_variants": 3
// SUMMARY: "verifier_failures": 0
// ZERO: arith.constant 0.000000e+00 : f64
// ZERO: arith.addf %3, %{{.*}} : f64
// ZERO: linalg.yield %{{.*}} : f64
// ASSOC: arith.addf %2, %out : f64
// ASSOC: arith.addf %out, %{{.*}} : f64
// EGG: match
// EGG: cublasDgemm_alpha_only
// SYN: no_match
