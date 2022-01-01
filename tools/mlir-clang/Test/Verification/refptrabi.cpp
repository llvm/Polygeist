// RUN: mlir-clang %s --function=ll -S | FileCheck %s

struct alignas(2) Half {
  unsigned short x;

  Half() = default;
};

extern "C" {

float thing(Half);

float ll(void* data) {
    return thing(*(Half*)data);
}

}

// CHECK:   func @ll(%arg0: !llvm.ptr<i8>) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %0 = memref.alloca() : memref<1x1xi16>
// CHECK-NEXT:     %1 = memref.alloca() : memref<1x1xi16>
// CHECK-NEXT:     %2 = llvm.bitcast %arg0 : !llvm.ptr<i8> to !llvm.ptr<i16>
// CHECK-NEXT:     %3 = llvm.getelementptr %2[%c0_i32] : (!llvm.ptr<i16>, i32) -> !llvm.ptr<i16>
// CHECK-NEXT:     %4 = llvm.load %3 : !llvm.ptr<i16>
// CHECK-NEXT:     affine.store %4, %1[0, 0] : memref<1x1xi16>
// CHECK-NEXT:     %5 = affine.load %1[0, 0] : memref<1x1xi16>
// CHECK-NEXT:     affine.store %5, %0[0, 0] : memref<1x1xi16>
// CHECK-NEXT:     %6 = memref.cast %0 : memref<1x1xi16> to memref<?x1xi16>
// CHECK-NEXT:     %7 = call @thing(%6) : (memref<?x1xi16>) -> f32
// CHECK-NEXT:     return %7 : f32
// CHECK-NEXT:   }
