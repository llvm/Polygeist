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
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %0 = memref.alloca() : memref<1x1xi16>
// CHECK-NEXT:     %1 = memref.alloca() : memref<1x1xi16>
// CHECK-NEXT:     %2 = memref.alloca() : memref<1x1xi16>
// CHECK-NEXT:     %3 = llvm.alloca %c1_i64 x !llvm.ptr<i8> : (i64) -> !llvm.ptr<ptr<i8>>
// CHECK-NEXT:     llvm.store %arg0, %3 : !llvm.ptr<ptr<i8>>
// CHECK-NEXT:     %4 = llvm.load %3 : !llvm.ptr<ptr<i8>>
// CHECK-NEXT:     %5 = llvm.bitcast %4 : !llvm.ptr<i8> to !llvm.ptr<struct<(i16)>>
// CHECK-NEXT:     %6 = llvm.getelementptr %5[%c0_i32, %c0_i32] : (!llvm.ptr<struct<(i16)>>, i32, i32) -> !llvm.ptr<i16>
// CHECK-NEXT:     %7 = llvm.load %6 : !llvm.ptr<i16>
// CHECK-NEXT:     affine.store %7, %1[0, 0] : memref<1x1xi16>
// CHECK-NEXT:     %8 = affine.load %1[0, 0] : memref<1x1xi16>
// CHECK-NEXT:     affine.store %8, %2[0, 0] : memref<1x1xi16>
// CHECK-NEXT:     %9 = affine.load %1[0, 0] : memref<1x1xi16>
// CHECK-NEXT:     llvm.store %9, %6 : !llvm.ptr<i16>
// CHECK-NEXT:     %10 = affine.load %2[0, 0] : memref<1x1xi16>
// CHECK-NEXT:     affine.store %10, %0[0, 0] : memref<1x1xi16>
// CHECK-NEXT:     %11 = memref.cast %0 : memref<1x1xi16> to memref<?x1xi16>
// CHECK-NEXT:     %12 = call @thing(%11) : (memref<?x1xi16>) -> f32
// CHECK-NEXT:     return %12 : f32
// CHECK-NEXT:   }
