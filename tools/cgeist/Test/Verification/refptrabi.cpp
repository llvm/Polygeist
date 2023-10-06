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

// CHECK-LABEL:   func.func @ll(
// CHECK-SAME:                  %[[VAL_0:[A-Za-z0-9_]*]]: memref<?xi8>) -> f32
// CHECK:           %[[VAL_1:[A-Za-z0-9_]*]] = memref.alloca() : memref<1x1xi16>
// CHECK:           %[[VAL_2:[A-Za-z0-9_]*]] = "polygeist.memref2pointer"(%[[VAL_0]]) : (memref<?xi8>) -> !llvm.ptr
// CHECK:           %[[VAL_3:[A-Za-z0-9_]*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> i16
// CHECK:           affine.store %[[VAL_3]], %[[VAL_1]][0, 0] : memref<1x1xi16>
// CHECK:           %[[VAL_4:[A-Za-z0-9_]*]] = memref.cast %[[VAL_1]] : memref<1x1xi16> to memref<?x1xi16>
// CHECK:           %[[VAL_5:[A-Za-z0-9_]*]] = call @thing(%[[VAL_4]]) : (memref<?x1xi16>) -> f32
// CHECK:           return %[[VAL_5]] : f32
// CHECK:         }
