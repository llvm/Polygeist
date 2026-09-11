// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s --dry-run --show-structured-regions 2>&1 | FileCheck %s
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s --enable-structured-rewrite | FileCheck %s --check-prefix=REWRITE
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s --enable-structured-rewrite | polygeist-opt --lower-kernel-launch-to-cublas | FileCheck %s --check-prefix=LOWER

#id = affine_map<(d0) -> (d0)>
#col = affine_map<(d0)[s0] -> (d0, s0)>

module {
  kernel.defn @cublasDgemm_simple(
      %a: tensor<?x?xf64>, %b: tensor<?x?xf64>, %c: tensor<?x?xf64>)
      -> tensor<?x?xf64> {
    kernel.yield %c : tensor<?x?xf64>
  }

  // Two local temporary stages feed a third stage.  The outer affine loops
  // and inner linalg loop together form a three-dimensional schedule.
  func.func @safe_staged_stencil(%a: memref<?x?x?xf64>,
                                 %b: memref<?x?x?xf64>,
                                 %out: memref<?x?x?xf64>,
                                 %n: index) {
    %t0 = memref.alloca() : memref<35xf64>
    %t1 = memref.alloca() : memref<35xf64>
    affine.for %z = 1 to %n {
      affine.for %y = 1 to %n {
        %a0 = polygeist.submap(%a, %z, %y) {map = #id} : (memref<?x?x?xf64>, index, index) -> memref<?xf64>
        %a1 = polygeist.submap(%a, %z, %y) {map = #id} : (memref<?x?x?xf64>, index, index) -> memref<?xf64>
        %v0 = polygeist.submap(%t0) {map = #id} : (memref<35xf64>) -> memref<?xf64>
        linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel"]}
            ins(%a0, %a1 : memref<?xf64>, memref<?xf64>)
            outs(%v0 : memref<?xf64>) {
        ^bb0(%x: f64, %y0: f64, %old: f64):
          %sum = arith.addf %x, %y0 : f64
          linalg.yield %sum : f64
        }
        %b0 = polygeist.submap(%b, %z, %y) {map = #id} : (memref<?x?x?xf64>, index, index) -> memref<?xf64>
        %b1 = polygeist.submap(%b, %z, %y) {map = #id} : (memref<?x?x?xf64>, index, index) -> memref<?xf64>
        %v1 = polygeist.submap(%t1) {map = #id} : (memref<35xf64>) -> memref<?xf64>
        linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel"]}
            ins(%b0, %b1 : memref<?xf64>, memref<?xf64>)
            outs(%v1 : memref<?xf64>) {
        ^bb0(%x: f64, %y0: f64, %old: f64):
          %sum = arith.addf %x, %y0 : f64
          linalg.yield %sum : f64
        }
        %r0 = polygeist.submap(%t0) {map = #id} : (memref<35xf64>) -> memref<?xf64>
        %r1 = polygeist.submap(%t1) {map = #id} : (memref<35xf64>) -> memref<?xf64>
        %dst = polygeist.submap(%out, %z, %y) {map = #id} : (memref<?x?x?xf64>, index, index) -> memref<?xf64>
        linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel"]}
            ins(%r0, %r1 : memref<?xf64>, memref<?xf64>)
            outs(%dst : memref<?xf64>) {
        ^bb0(%x: f64, %y0: f64, %old: f64):
          %sum = arith.addf %x, %y0 : f64
          linalg.yield %sum : f64
        }
      }
    }
    return
  }

  // A function argument may alias another argument, so it is deliberately
  // not accepted as a removable producer temporary.
  func.func @unknown_alias(%a: memref<?xf64>, %tmp: memref<?xf64>,
                           %out: memref<?xf64>) {
    linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
        ins(%a : memref<?xf64>) outs(%tmp : memref<?xf64>) {
    ^bb0(%x: f64, %old: f64):
      linalg.yield %x : f64
    }
    linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
        ins(%tmp : memref<?xf64>) outs(%out : memref<?xf64>) {
    ^bb0(%x: f64, %old: f64):
      linalg.yield %x : f64
    }
    return
  }

  // One loop around a GEMV-shaped generic contributes a second parallel
  // dimension.  Egglog therefore proves the combined 2-parallel/1-reduction
  // schedule has GEMM iteration structure.
  func.func @looped_gemv(%a: memref<?x?xf64> {llvm.noalias},
                         %b: memref<?x?xf64> {llvm.noalias},
                         %c: memref<?x?xf64> {llvm.noalias}, %batches: index) {
    affine.for %batch = 0 to %batches {
      %bs = polygeist.submap(%b, %batch) {map = #col} : (memref<?x?xf64>, index) -> memref<?xf64>
      %cs = polygeist.submap(%c, %batch) {map = #col} : (memref<?x?xf64>, index) -> memref<?xf64>
      linalg.generic {
          indexing_maps = [affine_map<(i, k) -> (i, k)>,
                           affine_map<(i, k) -> (k)>,
                           affine_map<(i, k) -> (i)>],
          iterator_types = ["parallel", "reduction"]}
          ins(%a, %bs : memref<?x?xf64>, memref<?xf64>)
          outs(%cs : memref<?xf64>) {
      ^bb0(%x: f64, %y0: f64, %old: f64):
        %product = arith.mulf %x, %y0 : f64
        %sum = arith.addf %old, %product : f64
        linalg.yield %sum : f64
      }
    }
    return
  }

  func.func @looped_reduction_with_side_effect(
      %a: memref<?x?xf64>, %b: memref<?x?xf64>, %c: memref<?x?xf64>,
      %flag: memref<?xi32>, %batches: index, %zero: i32) {
    affine.for %batch = 0 to %batches {
      affine.store %zero, %flag[%batch] : memref<?xi32>
      %bs = polygeist.submap(%b, %batch) {map = #col} : (memref<?x?xf64>, index) -> memref<?xf64>
      %cs = polygeist.submap(%c, %batch) {map = #col} : (memref<?x?xf64>, index) -> memref<?xf64>
      linalg.generic {
          indexing_maps = [affine_map<(i, k) -> (i, k)>,
                           affine_map<(i, k) -> (k)>,
                           affine_map<(i, k) -> (i)>],
          iterator_types = ["parallel", "reduction"]}
          ins(%a, %bs : memref<?x?xf64>, memref<?xf64>)
          outs(%cs : memref<?xf64>) {
      ^bb0(%x: f64, %y0: f64, %old: f64):
        %product = arith.mulf %x, %y0 : f64
        %sum = arith.addf %old, %product : f64
        linalg.yield %sum : f64
      }
    }
    return
  }
}

// CHECK: structured_fusion body#[0, 1, 2]
// CHECK-SAME: parent_loops=2 lifted_parallel_dims=3
// CHECK-SAME: temporary_roots=['%t0', '%t1']
// CHECK: structured_reject body#[3, 4]
// CHECK-SAME: temporary_roots=['%tmp']
// CHECK: structured_fusion body#[5]
// CHECK-SAME: lifted_parallel_dims=2
// CHECK-SAME: extracted=looped_gemv_as_gemm
// CHECK: structured_reject body#[6]

// REWRITE-LABEL: func.func @looped_gemv
// REWRITE: bufferization.to_tensor %{{.*}} restrict
// REWRITE: kernel.launch @cublasDgemm_simple
// REWRITE-NOT: affine.for
// REWRITE-LABEL: func.func @looped_reduction_with_side_effect
// REWRITE: affine.for

// LOWER-LABEL: func.func @looped_gemv
// LOWER: call @polygeist_cublas_dgemm
// LOWER-NOT: affine.for
// LOWER-LABEL: func.func @looped_reduction_with_side_effect
// LOWER: affine.for
