// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s --dry-run --show-structured-regions 2>&1 | FileCheck %s --check-prefix=ANALYZE
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s --enable-structured-rewrite | sed '/^\/\/ \(ANALYZE\|REWRITE\)/d' | FileCheck %s --check-prefix=REWRITE

#map0 = affine_map<(d0, d1)[s0] -> (s0, d0, d1)>
#map1 = affine_map<(d0)[s0] -> (s0, d0)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map5 = affine_map<(d0, d1) -> (d0, d1)>
#map6 = affine_map<(d0, d1) -> (d1)>
#map7 = affine_map<(d0, d1) -> (d0)>

module {
  kernel.defn @cublasDgemm_strided_batched_subtract(
      %a: tensor<?x?x?xf64>, %b: tensor<?x?x?xf64>,
      %c: tensor<?x?x?xf64>) -> tensor<?x?x?xf64> {
    kernel.yield %c : tensor<?x?x?xf64>
  }
  kernel.defn @cublasDgemv_strided_batched_subtract(
      %a: tensor<?x?x?xf64>, %x: tensor<?x?xf64>,
      %y: tensor<?x?xf64>) -> tensor<?x?xf64> {
    kernel.yield %y : tensor<?x?xf64>
  }

  // This is the shape obtained after BT's per-line lhs workspace has been
  // privatized: each outer iteration touches a distinct leading batch slice.
  func.func @outer_line_gemm(
      %a: memref<?x?x?xf64> {llvm.noalias},
      %b: memref<?x?x?xf64> {llvm.noalias},
      %c: memref<?x?x?xf64> {llvm.noalias}, %lines: index) {
    affine.for %line = 0 to %lines {
      %as = polygeist.submap(%a, %line) {map = #map0}
          : (memref<?x?x?xf64>, index) -> memref<?x?xf64>
      %bs = polygeist.submap(%b, %line) {map = #map0}
          : (memref<?x?x?xf64>, index) -> memref<?x?xf64>
      %cs = polygeist.submap(%c, %line) {map = #map0}
          : (memref<?x?x?xf64>, index) -> memref<?x?xf64>
      linalg.generic {indexing_maps = [#map2, #map3, #map4],
                      iterator_types = ["parallel", "parallel", "reduction"]}
          ins(%as, %bs : memref<?x?xf64>, memref<?x?xf64>)
          outs(%cs : memref<?x?xf64>) {
      ^bb0(%av: f64, %bv: f64, %out: f64):
        %product = arith.mulf %av, %bv : f64
        %updated = arith.subf %out, %product : f64
        linalg.yield %updated : f64
      }
    }
    return
  }

  func.func @outer_line_gemv(
      %a: memref<?x?x?xf64> {llvm.noalias},
      %x: memref<?x?xf64> {llvm.noalias},
      %y: memref<?x?xf64> {llvm.noalias}, %lines: index) {
    affine.for %line = 0 to %lines {
      %as = polygeist.submap(%a, %line) {map = #map0}
          : (memref<?x?x?xf64>, index) -> memref<?x?xf64>
      %xs = polygeist.submap(%x, %line) {map = #map1}
          : (memref<?x?xf64>, index) -> memref<?xf64>
      %ys = polygeist.submap(%y, %line) {map = #map1}
          : (memref<?x?xf64>, index) -> memref<?xf64>
      linalg.generic {indexing_maps = [#map5, #map6, #map7],
                      iterator_types = ["parallel", "reduction"]}
          ins(%as, %xs : memref<?x?xf64>, memref<?xf64>)
          outs(%ys : memref<?xf64>) {
      ^bb0(%av: f64, %xv: f64, %out: f64):
        %product = arith.mulf %av, %xv : f64
        %updated = arith.subf %out, %product : f64
        linalg.yield %updated : f64
      }
    }
    return
  }
}

// ANALYZE: structured_fusion body#[0]
// ANALYZE-SAME: extracted=looped_gemm_as_batched_gemm
// ANALYZE: structured_fusion body#[1]
// ANALYZE-SAME: extracted=looped_gemv_as_batched_gemv

// REWRITE-LABEL: func.func @outer_line_gemm
// REWRITE: kernel.launch @cublasDgemm_strided_batched_subtract
// REWRITE-NOT: affine.for
// REWRITE-LABEL: func.func @outer_line_gemv
// REWRITE: kernel.launch @cublasDgemv_strided_batched_subtract
// REWRITE-NOT: affine.for
