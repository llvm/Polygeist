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
// CHECK-NEXT:     %0 = memref.alloca() : memref<1x1xi16>
// CHECK-NEXT:     %1 = memref.alloca() : memref<1x1xi16>
// CHECK-NEXT:     %2 = llvm.bitcast %arg0 : !llvm.ptr<i8> to !llvm.ptr<i16>
// CHECK-NEXT:     %3 = llvm.load %2 : !llvm.ptr<i16>
// CHECK-NEXT:     affine.store %3, %1[0, 0] : memref<1x1xi16>
// CHECK-NEXT:     %4 = affine.load %1[0, 0] : memref<1x1xi16>
// CHECK-NEXT:     affine.store %4, %0[0, 0] : memref<1x1xi16>
// CHECK-NEXT:     %5 = memref.cast %0 : memref<1x1xi16> to memref<?x1xi16>
// CHECK-NEXT:     %6 = call @thing(%5) : (memref<?x1xi16>) -> f32
// CHECK-NEXT:     return %6 : f32
// CHECK-NEXT:   }
