// RUN: polygeist-opt %s --raise-affine-to-linalg | FileCheck %s
module {
  func.func @fill(%out: memref<?x8x16xf32>) {
    %zero = arith.constant 0.0 : f32
    %ptr = "polygeist.memref2pointer"(%out) :
        (memref<?x8x16xf32>) -> !llvm.ptr
    affine.for %i = 0 to 4096 {
      %ii = arith.index_cast %i : index to i32
      %elt = llvm.getelementptr %ptr[%ii] :
          (!llvm.ptr, i32) -> !llvm.ptr, f32
      llvm.store %zero, %elt : f32, !llvm.ptr
    }
    return
  }
}
// CHECK: memref.reinterpret_cast
// CHECK: linalg.fill
// CHECK-NOT: affine.for
// CHECK-NOT: llvm.store
