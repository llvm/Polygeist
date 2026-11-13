// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s --enable-structured-rewrite > %t.matched.mlir
// RUN: sed '/^\/\/ CHECK/d' %t.matched.mlir | FileCheck %s
// RUN: /usr/bin/python3 %S/../../scripts/correctness/inject_kernel_library.py %t.matched.mlir %S/../../generic_solver/kernel_library_phase2.mlir -o %t.abi.mlir
// RUN: polygeist-opt %t.abi.mlir --lower-kernel-launch-to-cublas --canonicalize -o /dev/null
// RUN: sed 's/0.000000e+00/1.000000e+00/g' %s > %t.seed.mlir
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %t.seed.mlir --enable-structured-rewrite | sed '/^\/\/ CHECK/d' | FileCheck %s --check-prefix=CHECK-NEGATIVE
// RUN: sed 's/arith.addf/arith.subf/g' %s > %t.operation.mlir
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %t.operation.mlir --enable-structured-rewrite | sed '/^\/\/ CHECK/d' | FileCheck %s --check-prefix=CHECK-NEGATIVE

#map = affine_map<(d0, d1, d2)[s0, s1, s2, s3] -> (s0, s1, s2, s3)>
#map1 = affine_map<(d0, d1, d2)[s0, s1] -> (s0, s1, d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2)[s0, s1, s2, s3] -> (s0, d0 + s1, d1 + s2, d2 + s3)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @scatter_volume(%arg0: memref<?x6x7x8xf32>, %arg1: memref<?x3x3x3x3xf32>, %arg2: memref<?x8x9x10xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c3 = arith.constant 3 : index
    %cst = arith.constant 0.000000e+00 : f32
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [2160], strides: [1] : memref<?x8x9x10xf32> to memref<2160xf32>
    linalg.fill ins(%cst : f32) outs(%reinterpret_cast : memref<2160xf32>)
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 3 {
        affine.for %arg5 = 0 to 6 {
          affine.for %arg6 = 0 to 7 {
            affine.for %arg7 = 0 to 8 {
              %0 = polygeist.submap(%arg0, %arg3, %arg5, %arg6, %arg7, %c3, %c3, %c3) {map = #map} : (memref<?x6x7x8xf32>, index, index, index, index, index, index, index) -> memref<?x?x?xf32>
              %1 = polygeist.submap(%arg1, %arg3, %arg4, %c3, %c3, %c3) {map = #map1} : (memref<?x3x3x3x3xf32>, index, index, index, index, index) -> memref<?x?x?xf32>
              %2 = polygeist.submap(%arg2, %arg4, %arg5, %arg6, %arg7, %c3, %c3, %c3) {map = #map2} : (memref<?x8x9x10xf32>, index, index, index, index, index, index, index) -> memref<?x?x?xf32>
              linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0, %1 : memref<?x?x?xf32>, memref<?x?x?xf32>) outs(%2 : memref<?x?x?xf32>) {
              ^bb0(%in: f32, %in_0: f32, %out: f32):
                %3 = arith.mulf %in, %in_0 : f32
                %4 = arith.addf %out, %3 : f32
                linalg.yield %4 : f32
              }
            }
          }
        }
      } {polygeist.was_parallel}
    }
    return
  }
}

// CHECK: kernel.launch @cudnnConvolutionTranspose3D_f32_memref
// CHECK-NOT: affine.for
// CHECK-NEGATIVE-NOT: kernel.launch @cudnnConvolutionTranspose3D_f32_memref
// CHECK-NEGATIVE: affine.for
// CHECK-NEGATIVE-NOT: kernel.launch @cudnnConvolutionTranspose3D_f32_memref
