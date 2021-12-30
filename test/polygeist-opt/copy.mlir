// RUN: polygeist-opt --inline --split-input-file %s | FileCheck %s

module {
  func @_Z3fooR5MHalf(%arg0: memref<?x1xi32>) -> f32 {
    %c0 = arith.constant 0 : index
    %0 = memref.alloca() : memref<1x1xi32>
    %1 = memref.alloca() : memref<1x1xi32>
    %2 = memref.cast %1 : memref<1x1xi32> to memref<?x1xi32>
    %3 = "polygeist.subindex"(%arg0, %c0) : (memref<?x1xi32>, index) -> memref<?xi32>
    %4 = "polygeist.memref2pointer"(%3) : (memref<?xi32>) -> !llvm.ptr<i8>
    %5 = "polygeist.pointer2memref"(%4) : (!llvm.ptr<i8>) -> memref<?x1xi32>
    call @_ZN6AMHalfC1ERKS_(%2, %5) : (memref<?x1xi32>, memref<?x1xi32>) -> ()
    %6 = memref.load %1[%c0, %c0] : memref<1x1xi32>
    memref.store %6, %0[%c0, %c0] : memref<1x1xi32>
    %7 = memref.cast %0 : memref<1x1xi32> to memref<?x1xi32>
    %8 = call @_Z4meta6AMHalf(%7) : (memref<?x1xi32>) -> f32
    return %8 : f32
  }
  func private @_Z4meta6AMHalf(memref<?x1xi32>) -> f32 attributes {llvm.linkage = #llvm.linkage<external>}
  func @_ZN6AMHalfC1ERKS_(%arg0: memref<?x1xi32>, %arg1: memref<?x1xi32>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %c0 = arith.constant 0 : index
    %0 = memref.load %arg1[%c0, %c0] : memref<?x1xi32>
    memref.store %0, %arg0[%c0, %c0] : memref<?x1xi32>
    return
  }
}

// CHECK:   func @_Z3fooR5MHalf(%arg0: memref<?x1xi32>) -> f32
// CHECK-NEXT:     %c0 = arith.constant 0 : index
// CHECK-NEXT:     %0 = memref.alloca() : memref<1x1xi32>
// CHECK-NEXT:     %1 = memref.alloca() : memref<1x1xi32>
// CHECK-NEXT:     %2 = memref.load %arg0[%c0, %c0] : memref<?x1xi32>
// CHECK-NEXT:     memref.store %2, %1[%c0, %c0] : memref<1x1xi32>
// CHECK-NEXT:     %3 = memref.load %1[%c0, %c0] : memref<1x1xi32>
// CHECK-NEXT:     memref.store %3, %0[%c0, %c0] : memref<1x1xi32>
// CHECK-NEXT:     %4 = memref.cast %0 : memref<1x1xi32> to memref<?x1xi32>
// CHECK-NEXT:     %5 = call @_Z4meta6AMHalf(%4) : (memref<?x1xi32>) -> f32
// CHECK-NEXT:     return %5 : f32
// CHECK-NEXT:   }
