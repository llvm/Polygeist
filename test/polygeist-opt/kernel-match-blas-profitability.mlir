// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s --dry-run 2>&1 | FileCheck %s

#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d1)>
#map4 = affine_map<(d0, d1) -> (d0)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map6 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map7 = affine_map<(d0, d1, d2) -> (d0, d1)>

module {
  // This is representative of the length-3/length-5 reductions raised from
  // NPB UA.  Its dynamic memref spelling hides the size, but the submap
  // operands retain it.  Keep the loop instead of paying for one GPU call.
  func.func @tiny_dot(%storage: memref<f64>) {
    %c5 = arith.constant 5 : index
    %x_storage = memref.alloca() : memref<5xf64>
    %y_storage = memref.alloca() : memref<5xf64>
    %x = polygeist.submap(%x_storage, %c5) {map = #map0}
        : (memref<5xf64>, index) -> memref<?xf64>
    %y = polygeist.submap(%y_storage, %c5) {map = #map0}
        : (memref<5xf64>, index) -> memref<?xf64>
    %out = polygeist.submap(%storage, %c5) {map = #map1}
        : (memref<f64>, index) -> memref<?xf64>
    linalg.generic {indexing_maps = [#map0, #map0, #map0],
                    iterator_types = ["reduction"]}
        ins(%x, %y : memref<?xf64>, memref<?xf64>)
        outs(%out : memref<?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out_value: f64):
      %product = arith.mulf %in, %in_0 : f64
      %sum = arith.addf %out_value, %product : f64
      linalg.yield %sum : f64
    }
    return
  }

  // Unknown sizes are deliberately not rejected: absence of proof that a
  // call is profitable is not proof that it is unprofitable.
  func.func @dynamic_dot(%x: memref<?xf64>, %y: memref<?xf64>,
                         %storage: memref<f64>, %n: index) {
    %out_dynamic = polygeist.submap(%storage, %n) {map = #map1}
        : (memref<f64>, index) -> memref<?xf64>
    linalg.generic {indexing_maps = [#map0, #map0, #map0],
                    iterator_types = ["reduction"]}
        ins(%x, %y : memref<?xf64>, memref<?xf64>)
        outs(%out_dynamic : memref<?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out_value: f64):
      %product = arith.mulf %in, %in_0 : f64
      %sum = arith.addf %out_value, %product : f64
      linalg.yield %sum : f64
    }
    return
  }

  func.func @tiny_gemv_subtract(
      %a: tensor<5x5xf64>, %x: tensor<5xf64>, %y: tensor<5xf64>)
      -> tensor<5xf64> {
    %result = linalg.generic {
        indexing_maps = [#map2, #map3, #map4],
        iterator_types = ["parallel", "reduction"]}
        ins(%a, %x : tensor<5x5xf64>, tensor<5xf64>)
        outs(%y : tensor<5xf64>) {
    ^bb0(%av: f64, %xv: f64, %out: f64):
      %product = arith.mulf %av, %xv : f64
      %updated = arith.subf %out, %product : f64
      linalg.yield %updated : f64
    } -> tensor<5xf64>
    return %result : tensor<5xf64>
  }

  func.func @tiny_gemm_subtract(
      %a: tensor<5x5xf64>, %b: tensor<5x5xf64>, %c: tensor<5x5xf64>)
      -> tensor<5x5xf64> {
    %result = linalg.generic {
        indexing_maps = [#map5, #map6, #map7],
        iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%a, %b : tensor<5x5xf64>, tensor<5x5xf64>)
        outs(%c : tensor<5x5xf64>) {
    ^bb0(%av: f64, %bv: f64, %out: f64):
      %product = arith.mulf %av, %bv : f64
      %updated = arith.subf %out, %product : f64
      linalg.yield %updated : f64
    } -> tensor<5x5xf64>
    return %result : tensor<5x5xf64>
  }
}

// CHECK: profitability_reject body#[0]  cublasDdot_memref[iterations=5,minimum=256]
// CHECK: match          body#[1]  cublasDdot_memref
// CHECK: profitability_reject body#[2]  cublasDgemv_subtract[iterations=25,minimum=256]
// CHECK: profitability_reject body#[3]  cublasDgemm_subtract[iterations=125,minimum=256]
