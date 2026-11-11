// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s --enable-structured-rewrite > %t.matched.mlir
// RUN: sed '/^\/\/ CHECK/d' %t.matched.mlir | FileCheck %s
// RUN: /usr/bin/python3 %S/../../scripts/correctness/inject_kernel_library.py %t.matched.mlir %S/../../generic_solver/kernel_library_phase2.mlir -o %t.abi.mlir
// RUN: polygeist-opt %t.abi.mlir --lower-kernel-launch-to-cublas --canonicalize -o /dev/null
// RUN: sed 's/cmpi sge/cmpi sgt/g' %s > %t.predicate.mlir
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %t.predicate.mlir --enable-structured-rewrite | sed '/^\/\/ CHECK/d' | FileCheck %s --check-prefix=CHECK-NEGATIVE
// RUN: sed 's/0.000000e+00/1.000000e+00/g' %s > %t.seed.mlir
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %t.seed.mlir --enable-structured-rewrite | sed '/^\/\/ CHECK/d' | FileCheck %s --check-prefix=CHECK-NEGATIVE

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d2 + d0 * 25 + d1 * 5)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d3 * 11 + d0 * 110)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map4 = affine_map<(d0) -> (d0 * 2)>
#map5 = affine_map<(d0) -> (d0 * 2 + 2)>
module {
  func.func @window_gradient_2d(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 4.000000e+00 : f32
    %c10 = arith.constant 10 : index
    %c5 = arith.constant 5 : index
    %c2 = arith.constant 2 : index
    %c220 = arith.constant 220 : index
    %0 = bufferization.to_tensor %arg0 : memref<?xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?xf32>
    %2 = polygeist.submap(%1, %c220) {map = #map} : (tensor<?xf32>, index) -> tensor<?xf32>
    %3 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%2 : tensor<?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    } -> tensor<?xf32>
    %4 = polygeist.submapInverse(%1, %3, %c220) {map = #map} : (tensor<?xf32>, tensor<?xf32>, index) -> tensor<?xf32>
    %5 = polygeist.submap(%0, %c2, %c5, %c5, %c10, %c10) {map = #map1} : (tensor<?xf32>, index, index, index, index, index) -> tensor<?x?x?x?x?xf32>
    %6 = polygeist.submap(%4, %c2, %c5, %c5, %c10, %c10) {map = #map2} : (tensor<?xf32>, index, index, index, index, index) -> tensor<?x?x?x?x?xf32>
    %7 = linalg.generic {doc = "", indexing_maps = [#map3, #map3], iterator_types = ["parallel", "reduction", "reduction", "parallel", "parallel"], library_call = ""} ins(%5 : tensor<?x?x?x?x?xf32>) outs(%6 : tensor<?x?x?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %10 = linalg.index 1 : index
      %11 = linalg.index 2 : index
      %12 = arith.divf %in, %cst_0 : f32
      %13 = arith.addf %out, %12 : f32
      %14 = linalg.index 4 : index
      %15 = affine.apply #map4(%11)
      %16 = arith.cmpi sge, %14, %15 : index
      %17 = affine.apply #map5(%11)
      %18 = arith.cmpi slt, %14, %17 : index
      %19 = arith.andi %16, %18 : i1
      %20 = arith.select %19, %13, %out : f32
      %21 = linalg.index 3 : index
      %22 = affine.apply #map4(%10)
      %23 = arith.cmpi sge, %21, %22 : index
      %24 = affine.apply #map5(%10)
      %25 = arith.cmpi slt, %21, %24 : index
      %26 = arith.andi %23, %25 : i1
      %27 = arith.select %26, %20, %out : f32
      linalg.yield %27 : f32
    } -> tensor<?x?x?x?x?xf32>
    %8 = polygeist.submapInverse(%4, %7, %c2, %c5, %c5, %c10, %c10) {map = #map2} : (tensor<?xf32>, tensor<?x?x?x?x?xf32>, index, index, index, index, index) -> tensor<?xf32>
    %9 = bufferization.to_memref %8 : memref<?xf32>
    memref.copy %9, %arg1 : memref<?xf32> to memref<?xf32>
    return
  }
}

// CHECK: kernel.launch @cudnnAveragePool_f32_flat2
// CHECK-NOT: linalg.generic
// CHECK-NEGATIVE-NOT: kernel.launch @cudnnAveragePool_f32_flat2
// CHECK-NEGATIVE: linalg.generic
// CHECK-NEGATIVE-NOT: kernel.launch @cudnnAveragePool_f32_flat2
