// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s --enable-structured-rewrite > %t.matched.mlir
// RUN: sed '/^\/\/ CHECK/d' %t.matched.mlir | FileCheck %s
// RUN: /usr/bin/python3 %S/../../scripts/correctness/inject_kernel_library.py %t.matched.mlir %S/../../generic_solver/kernel_library_phase2.mlir -o %t.abi.mlir
// RUN: polygeist-opt %t.abi.mlir --lower-kernel-launch-to-cublas --canonicalize -o /dev/null
// RUN: sed 's/cmpi sge/cmpi sgt/g' %s > %t.predicate.mlir
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %t.predicate.mlir --enable-structured-rewrite | sed '/^\/\/ CHECK/d' | FileCheck %s --check-prefix=CHECK-NEGATIVE
// RUN: sed 's/0.000000e+00/1.000000e+00/g' %s > %t.seed.mlir
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %t.seed.mlir --enable-structured-rewrite | sed '/^\/\/ CHECK/d' | FileCheck %s --check-prefix=CHECK-NEGATIVE

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3 + d0 * 36 + d1 * 12 + d2 * 4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d6 + d4 * 56 + d0 * 336 + d5 * 8)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>
#map4 = affine_map<(d0) -> (d0 * 2)>
#map5 = affine_map<(d0) -> (d0 * 2 + 2)>
module {
  func.func @window_gradient_3d(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 8.000000e+00 : f32
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c6 = arith.constant 6 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c672 = arith.constant 672 : index
    %0 = bufferization.to_tensor %arg0 : memref<?xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?xf32>
    %2 = polygeist.submap(%1, %c672) {map = #map} : (tensor<?xf32>, index) -> tensor<?xf32>
    %3 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%2 : tensor<?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    } -> tensor<?xf32>
    %4 = polygeist.submapInverse(%1, %3, %c672) {map = #map} : (tensor<?xf32>, tensor<?xf32>, index) -> tensor<?xf32>
    %5 = polygeist.submap(%0, %c2, %c3, %c3, %c4, %c6, %c6, %c8) {map = #map1} : (tensor<?xf32>, index, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?x?xf32>
    %6 = polygeist.submap(%4, %c2, %c3, %c3, %c4, %c6, %c6, %c8) {map = #map2} : (tensor<?xf32>, index, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?x?xf32>
    %7 = linalg.generic {doc = "", indexing_maps = [#map3, #map3], iterator_types = ["parallel", "reduction", "reduction", "reduction", "parallel", "parallel", "parallel"], library_call = ""} ins(%5 : tensor<?x?x?x?x?x?x?xf32>) outs(%6 : tensor<?x?x?x?x?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %10 = linalg.index 1 : index
      %11 = linalg.index 2 : index
      %12 = linalg.index 3 : index
      %13 = arith.divf %in, %cst_0 : f32
      %14 = arith.addf %out, %13 : f32
      %15 = linalg.index 6 : index
      %16 = affine.apply #map4(%12)
      %17 = arith.cmpi sge, %15, %16 : index
      %18 = affine.apply #map5(%12)
      %19 = arith.cmpi slt, %15, %18 : index
      %20 = arith.andi %17, %19 : i1
      %21 = arith.select %20, %14, %out : f32
      %22 = linalg.index 5 : index
      %23 = affine.apply #map4(%11)
      %24 = arith.cmpi sge, %22, %23 : index
      %25 = affine.apply #map5(%11)
      %26 = arith.cmpi slt, %22, %25 : index
      %27 = arith.andi %24, %26 : i1
      %28 = arith.select %27, %21, %out : f32
      %29 = linalg.index 4 : index
      %30 = affine.apply #map4(%10)
      %31 = arith.cmpi sge, %29, %30 : index
      %32 = affine.apply #map5(%10)
      %33 = arith.cmpi slt, %29, %32 : index
      %34 = arith.andi %31, %33 : i1
      %35 = arith.select %34, %28, %out : f32
      linalg.yield %35 : f32
    } -> tensor<?x?x?x?x?x?x?xf32>
    %8 = polygeist.submapInverse(%4, %7, %c2, %c3, %c3, %c4, %c6, %c6, %c8) {map = #map2} : (tensor<?xf32>, tensor<?x?x?x?x?x?x?xf32>, index, index, index, index, index, index, index) -> tensor<?xf32>
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
