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

// CHECK:   func @ll(%[[arg0:.+]]: memref<?xi8>) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %[[V0:.+]] = memref.alloca() : memref<1x1xi16>
// CHECK-NEXT:     %[[V1:.+]] = "polygeist.memref2pointer"(%[[arg0]]) : (memref<?xi8>) -> !llvm.ptr<i8>
// CHECK-NEXT:     %[[V2:.+]] = llvm.bitcast %[[V1]] : !llvm.ptr<i8> to !llvm.ptr<i16>
// CHECK-NEXT:     %[[V3:.+]] = llvm.load %[[V2]] : !llvm.ptr<i16>
// CHECK-NEXT:     affine.store %[[V3]], %[[V0]][0, 0] : memref<1x1xi16>
// CHECK-NEXT:     %[[V4:.+]] = memref.cast %[[V0]] : memref<1x1xi16> to memref<?x1xi16>
// CHECK-NEXT:     %[[V5:.+]] = call @thing(%[[V4]]) : (memref<?x1xi16>) -> f32
// CHECK-NEXT:     return %[[V5]] : f32
// CHECK-NEXT:   }
