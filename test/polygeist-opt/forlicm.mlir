// RUN: polygeist-opt --parallel-licm --split-input-file %s | FileCheck %s

module {
  func.func @main(%11 : index) {
    %12 = memref.alloc(%11) : memref<?xf32>
      %28 = memref.alloca() : memref<16x16xf32>
      affine.for %arg4 = 0 to 16 {
        %29 = affine.load %28[0, 0] : memref<16x16xf32>
        %a = affine.load %12[0] : memref<?xf32>
        %z = arith.addf %a, %29 : f32
        affine.store %z, %12[0] : memref<?xf32>
      }
    return
  }
}

// CHECK: #set = affine_set<() : (15 >= 0)>
// CHECK-NEXT: module {
// CHECK-NEXT:   func.func @main(%arg0: index) {
// CHECK-NEXT:     %0 = memref.alloc(%arg0) : memref<?xf32>
// CHECK-NEXT:     %1 = memref.alloca() : memref<16x16xf32>
// CHECK-NEXT:     affine.if #set() {
// CHECK-NEXT:       %2 = affine.load %1[0, 0] : memref<16x16xf32>
// CHECK-NEXT:       affine.for %arg1 = 0 to 16 {
// CHECK-NEXT:         %3 = affine.load %0[0] : memref<?xf32>
// CHECK-NEXT:         %4 = arith.addf %3, %2 : f32
// CHECK-NEXT:         affine.store %4, %0[0] : memref<?xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT: }
