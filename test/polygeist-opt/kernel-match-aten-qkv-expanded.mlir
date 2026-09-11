// RUN: sed '/^\/\/ \(CHECK\|BAD\)/d' %s | /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py /dev/stdin | FileCheck %s --check-prefix=CHECK
// RUN: sed '/^\/\/ \(CHECK\|BAD\)/d' %s | sed 's/(d0, d1, 2, d2, d3)/(d0, d1, 1, d2, d3)/' | /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py /dev/stdin | FileCheck %s --check-prefix=BAD

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, 1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, 2, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (0, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (1, d2, d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (2, d2, d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module {
  func.func @qkv(%arg0: memref<?x16x3x4x8xf32>,
                 %arg1: memref<?x4x8xf32>, %scale: f32,
                 %arg2: memref<?x4x16x8xf32>,
                 %arg3: memref<?x4x16x8xf32>,
                 %arg4: memref<?x4x16x8xf32>) {
    %c2 = arith.constant 2 : index
    %c16 = arith.constant 16 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %src = bufferization.to_tensor %arg0 : memref<?x16x3x4x8xf32>
    %bias = bufferization.to_tensor %arg1 : memref<?x4x8xf32>
    %q = bufferization.to_tensor %arg2 : memref<?x4x16x8xf32>
    %k = bufferization.to_tensor %arg3 : memref<?x4x16x8xf32>
    %v = bufferization.to_tensor %arg4 : memref<?x4x16x8xf32>

    %qsrc = polygeist.submap(%src, %c2, %c16, %c4, %c8) {map = #map0} : (tensor<?x16x3x4x8xf32>, index, index, index, index) -> tensor<?x?x?x?xf32>
    %qbias = polygeist.submap(%bias, %c2, %c16, %c4, %c8) {map = #map3} : (tensor<?x4x8xf32>, index, index, index, index) -> tensor<?x?x?x?xf32>
    %qout = polygeist.submap(%q, %c2, %c16, %c4, %c8) {map = #map6} : (tensor<?x4x16x8xf32>, index, index, index, index) -> tensor<?x?x?x?xf32>
    %qresult = linalg.generic {doc = "", indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} ins(%qsrc, %qbias : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) outs(%qout : tensor<?x?x?x?xf32>) {
    ^bb0(%x: f32, %b: f32, %out: f32):
      %sum = arith.addf %x, %b : f32
      %scaled = arith.mulf %sum, %scale : f32
      linalg.yield %scaled : f32
    } -> tensor<?x?x?x?xf32>
    %qinv = polygeist.submapInverse(%q, %qresult, %c2, %c16, %c4, %c8) {map = #map6} : (tensor<?x4x16x8xf32>, tensor<?x?x?x?xf32>, index, index, index, index) -> tensor<?x4x16x8xf32>
    %qm = bufferization.to_memref %qinv : memref<?x4x16x8xf32>
    memref.copy %qm, %arg2 : memref<?x4x16x8xf32> to memref<?x4x16x8xf32>

    %ksrc = polygeist.submap(%src, %c2, %c16, %c4, %c8) {map = #map1} : (tensor<?x16x3x4x8xf32>, index, index, index, index) -> tensor<?x?x?x?xf32>
    %kbias = polygeist.submap(%bias, %c2, %c16, %c4, %c8) {map = #map4} : (tensor<?x4x8xf32>, index, index, index, index) -> tensor<?x?x?x?xf32>
    %kout = polygeist.submap(%k, %c2, %c16, %c4, %c8) {map = #map6} : (tensor<?x4x16x8xf32>, index, index, index, index) -> tensor<?x?x?x?xf32>
    %kresult = linalg.generic {doc = "", indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} ins(%ksrc, %kbias : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) outs(%kout : tensor<?x?x?x?xf32>) {
    ^bb0(%x: f32, %b: f32, %out: f32):
      %sum = arith.addf %x, %b : f32
      linalg.yield %sum : f32
    } -> tensor<?x?x?x?xf32>
    %kinv = polygeist.submapInverse(%k, %kresult, %c2, %c16, %c4, %c8) {map = #map6} : (tensor<?x4x16x8xf32>, tensor<?x?x?x?xf32>, index, index, index, index) -> tensor<?x4x16x8xf32>
    %km = bufferization.to_memref %kinv : memref<?x4x16x8xf32>
    memref.copy %km, %arg3 : memref<?x4x16x8xf32> to memref<?x4x16x8xf32>

    %vsrc = polygeist.submap(%src, %c2, %c16, %c4, %c8) {map = #map2} : (tensor<?x16x3x4x8xf32>, index, index, index, index) -> tensor<?x?x?x?xf32>
    %vbias = polygeist.submap(%bias, %c2, %c16, %c4, %c8) {map = #map5} : (tensor<?x4x8xf32>, index, index, index, index) -> tensor<?x?x?x?xf32>
    %vout = polygeist.submap(%v, %c2, %c16, %c4, %c8) {map = #map6} : (tensor<?x4x16x8xf32>, index, index, index, index) -> tensor<?x?x?x?xf32>
    %vresult = linalg.generic {doc = "", indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} ins(%vsrc, %vbias : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) outs(%vout : tensor<?x?x?x?xf32>) {
    ^bb0(%x: f32, %b: f32, %out: f32):
      %sum = arith.addf %x, %b : f32
      linalg.yield %sum : f32
    } -> tensor<?x?x?x?xf32>
    %vinv = polygeist.submapInverse(%v, %vresult, %c2, %c16, %c4, %c8) {map = #map6} : (tensor<?x4x16x8xf32>, tensor<?x?x?x?xf32>, index, index, index, index) -> tensor<?x4x16x8xf32>
    %vm = bufferization.to_memref %vinv : memref<?x4x16x8xf32>
    memref.copy %vm, %arg4 : memref<?x4x16x8xf32> to memref<?x4x16x8xf32>
    return
  }
}

// CHECK: kernel.launch @cudnnTransformBiasRescaleQKV_f32_memref
// CHECK-NOT: memref.copy
// CHECK-NOT: linalg.generic
// BAD-NOT: kernel.launch @cudnnTransformBiasRescaleQKV_f32_memref
// BAD: linalg.generic
