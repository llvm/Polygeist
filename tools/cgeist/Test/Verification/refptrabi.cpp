// RUN: cgeist %s --function=ll -S | FileCheck %s

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

// CHECK:   func @ll(%arg0: memref<?xi8>) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %0 = memref.alloca() : memref<1x1xi16>
// CHECK-NEXT:     %1 = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr<i8>
// CHECK-NEXT:     %2 = llvm.bitcast %1 : !llvm.ptr<i8> to !llvm.ptr<i16>
// CHECK-NEXT:     %3 = llvm.load %2 : !llvm.ptr<i16>
// CHECK-NEXT:     affine.store %3, %0[0, 0] : memref<1x1xi16>
// CHECK-NEXT:     %4 = memref.cast %0 : memref<1x1xi16> to memref<?x1xi16>
// CHECK-NEXT:     %5 = call @thing(%4) : (memref<?x1xi16>) -> f32
// CHECK-NEXT:     return %5 : f32
// CHECK-NEXT:   }
