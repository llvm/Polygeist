// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s --enable-structured-rewrite > %t.matched.mlir
// RUN: sed '/^\/\/ CHECK/d' %t.matched.mlir | FileCheck %s
// RUN: /usr/bin/python3 %S/../../scripts/correctness/inject_kernel_library.py %t.matched.mlir %S/../../generic_solver/kernel_library_phase2.mlir -o %t.abi.mlir
// RUN: polygeist-opt %t.abi.mlir --lower-kernel-launch-to-cublas --canonicalize -o /dev/null
// RUN: sed -e 's/ogt/olt/g' -e 's/64/7/g' -e 's/63/6/g' %s > %t.min.mlir
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %t.min.mlir --enable-structured-rewrite | sed '/^\/\/ CHECK/d' | FileCheck %s --check-prefix=CHECK-MIN
// RUN: sed 's/cmpf ogt/cmpf oge/g' %s > %t.last-tie.mlir
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %t.last-tie.mlir --enable-structured-rewrite | sed '/^\/\/ CHECK/d' | FileCheck %s --check-prefix=CHECK-NEGATIVE

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @first_winner(%answer: memref<?xi32>, %data: memref<?x64xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4 = arith.constant 4 : index
    %c63 = arith.constant 63 : index
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca(%c4) : memref<?xi32>
    %alloca_0 = memref.alloca(%c4) : memref<?xf32>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%alloca : memref<?xi32>) {
    ^bb0(%out: i32):
      linalg.yield %c0_i32 : i32
    }
    %subview = memref.subview %data[0, 0] [%c4, 1] [1, 1] : memref<?x64xf32> to memref<?xf32, strided<[64]>>
    %subview_1 = memref.subview %alloca_0[0] [%c4] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%subview : memref<?xf32, strided<[64]>>) outs(%subview_1 : memref<?xf32, strided<[1]>>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    %subview_2 = memref.subview %data[0, 1] [%c4, %c63] [1, 1] : memref<?x64xf32> to memref<?x?xf32, strided<[64, 1], offset: 1>>
    %subview_3 = memref.subview %alloca[0] [%c4] [1] : memref<?xi32> to memref<?xi32, strided<[1]>>
    %reinterpret_cast = memref.reinterpret_cast %alloca_0 to offset: [0], sizes: [%c4], strides: [1] : memref<?xf32> to memref<4xf32>
    linalg.generic {indexing_maps = [#map1, #map2, #map2], iterator_types = ["parallel", "reduction"]} ins(%subview_2 : memref<?x?xf32, strided<[64, 1], offset: 1>>) outs(%subview_3, %reinterpret_cast : memref<?xi32, strided<[1]>>, memref<4xf32>) {
    ^bb0(%in: f32, %out: i32, %out_4: f32):
      %0 = linalg.index 1 : index
      %1 = arith.addi %0, %c1 : index
      %2 = arith.index_cast %1 : index to i32
      %3 = arith.cmpf ogt, %in, %out_4 : f32
      %4 = arith.select %3, %2, %out : i32
      %5 = arith.select %3, %in, %out_4 : f32
      linalg.yield %4, %5 : i32, f32
    }
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%alloca : memref<?xi32>) outs(%answer : memref<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    }
    return
  }
}

// CHECK: kernel.launch @cubSegmentedArgMax_f32_i32_memref
// CHECK-NOT: linalg.generic
// CHECK-MIN: kernel.launch @cubSegmentedArgMin_f32_i32_memref
// CHECK-MIN-NOT: linalg.generic
// CHECK-NEGATIVE-NOT: kernel.launch @cubSegmentedArg
// CHECK-NEGATIVE: linalg.generic
// CHECK-NEGATIVE-NOT: kernel.launch @cubSegmentedArg
