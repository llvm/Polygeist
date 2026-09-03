// RUN: polygeist-opt --empty-tensor-to-alloc-tensor \
// RUN:   '--one-shot-bufferize=allow-unknown-ops bufferize-function-boundaries' \
// RUN:   --lower-kernel-launch-to-cublas %s | FileCheck %s

#a = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#b = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#c = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#flat = affine_map<(d0, d1, d2, d3, d4) ->
                    (d4 + d3 * 5 + d2 * 20 + d1 * 80 + d0 * 320)>

module {
  kernel.defn @cutensornetContraction2_f64_r5r5r4(
      %a: tensor<?x?x?x?x?xf64>, %b: tensor<?x?x?x?x?xf64>,
      %c: tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64> {
    kernel.yield %c : tensor<?x?x?x?xf64>
  }

  func.func @bufferized_contraction(
      %a: memref<?x?x?x?x?xf64>, %b: memref<?x?x?x?x?xf64>,
      %c: memref<?x?x?x?xf64>) {
    %at = bufferization.to_tensor %a restrict writable
        : memref<?x?x?x?x?xf64>
    %bt = bufferization.to_tensor %b restrict writable
        : memref<?x?x?x?x?xf64>
    %ct = bufferization.to_tensor %c restrict writable
        : memref<?x?x?x?xf64>
    %result = kernel.launch @cutensornetContraction2_f64_r5r5r4(
        %at, %bt, %ct) {
          contraction_maps = [#a, #b, #c]
        } : (tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>,
             tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>
    %result_memref = bufferization.to_memref %result
        : memref<?x?x?x?xf64>
    memref.copy %result_memref, %c
        : memref<?x?x?x?xf64> to memref<?x?x?x?xf64>
    return
  }

  func.func @bufferized_flat_submap_contraction(
      %a: memref<?xf64>, %b: memref<?xf64>,
      %c: memref<?x?x?x?xf64>) {
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %at = bufferization.to_tensor %a restrict writable : memref<?xf64>
    %bt = bufferization.to_tensor %b restrict writable : memref<?xf64>
    %ct = bufferization.to_tensor %c restrict writable
        : memref<?x?x?x?xf64>
    %av = polygeist.submap(%at, %c2, %c4, %c4, %c4, %c5) {map = #flat}
        : (tensor<?xf64>, index, index, index, index, index) ->
          tensor<?x?x?x?x?xf64>
    %bv = polygeist.submap(%bt, %c2, %c4, %c4, %c4, %c5) {map = #flat}
        : (tensor<?xf64>, index, index, index, index, index) ->
          tensor<?x?x?x?x?xf64>
    %result = kernel.launch @cutensornetContraction2_f64_r5r5r4(
        %av, %bv, %ct) {
          contraction_maps = [#a, #b, #c]
        } : (tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>,
             tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>
    %result_memref = bufferization.to_memref %result
        : memref<?x?x?x?xf64>
    memref.copy %result_memref, %c
        : memref<?x?x?x?xf64> to memref<?x?x?x?xf64>
    return
  }
}

// CHECK-LABEL: func.func @bufferized_contraction
// CHECK: call @polygeist_cutensornet_contraction2_f64
// CHECK-NOT: kernel.launch
// CHECK-NOT: bufferization.to_tensor
// CHECK-NOT: bufferization.to_memref
// CHECK-NOT: memref.copy

// CHECK-LABEL: func.func @bufferized_flat_submap_contraction
// CHECK: call @polygeist_cutensornet_contraction2_f64
// CHECK-NOT: polygeist.submap
// CHECK-NOT: bufferization.to_memref
// CHECK-NOT: memref.copy
