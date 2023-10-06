// RUN: cgeist %s --function=* -S | FileCheck %s

class D {
  double a;
  double b;
};

class QStream {
  D device_;
  int id;
};

QStream ilaunch_kernel(QStream x) {
  return x;
}

// CHECK-LABEL:   func.func @_Z14ilaunch_kernel7QStream(
// CHECK-SAME:                                          %[[VAL_0:[A-Za-z0-9_]*]]: !llvm.struct<(struct<(f64, f64)>, i32)>) -> !llvm.struct<(struct<(f64, f64)>, i32)>
// CHECK:           %[[VAL_1:[A-Za-z0-9_]*]] = memref.alloca() : memref<1x!llvm.struct<(struct<(f64, f64)>, i32)>>
// CHECK:           %[[VAL_2:[A-Za-z0-9_]*]] = memref.cast %[[VAL_1]] : memref<1x!llvm.struct<(struct<(f64, f64)>, i32)>> to memref<?x!llvm.struct<(struct<(f64, f64)>, i32)>>
// CHECK:           %[[VAL_3:[A-Za-z0-9_]*]] = memref.alloca() : memref<1x!llvm.struct<(struct<(f64, f64)>, i32)>>
// CHECK:           %[[VAL_4:[A-Za-z0-9_]*]] = memref.cast %[[VAL_3]] : memref<1x!llvm.struct<(struct<(f64, f64)>, i32)>> to memref<?x!llvm.struct<(struct<(f64, f64)>, i32)>>
// CHECK:           affine.store %[[VAL_0]], %[[VAL_3]][0] : memref<1x!llvm.struct<(struct<(f64, f64)>, i32)>>
// CHECK:           call @_ZN7QStreamC1EOS_(%[[VAL_2]], %[[VAL_4]]) : (memref<?x!llvm.struct<(struct<(f64, f64)>, i32)>>, memref<?x!llvm.struct<(struct<(f64, f64)>, i32)>>) -> ()
// CHECK:           %[[VAL_5:[A-Za-z0-9_]*]] = affine.load %[[VAL_1]][0] : memref<1x!llvm.struct<(struct<(f64, f64)>, i32)>>
// CHECK:           return %[[VAL_5]] : !llvm.struct<(struct<(f64, f64)>, i32)>
// CHECK:         }

// CHECK-LABEL:   func.func @_ZN7QStreamC1EOS_(
// CHECK-SAME:                                 %[[VAL_0:[A-Za-z0-9_]*]]: memref<?x!llvm.struct<(struct<(f64, f64)>, i32)>>,
// CHECK-SAME:                                 %[[VAL_1:[A-Za-z0-9_]*]]: memref<?x!llvm.struct<(struct<(f64, f64)>, i32)>>)
// CHECK:           %[[VAL_2:[A-Za-z0-9_]*]] = "polygeist.memref2pointer"(%[[VAL_0]]) : (memref<?x!llvm.struct<(struct<(f64, f64)>, i32)>>) -> !llvm.ptr
// CHECK:           %[[VAL_3:[A-Za-z0-9_]*]] = "polygeist.memref2pointer"(%[[VAL_1]]) : (memref<?x!llvm.struct<(struct<(f64, f64)>, i32)>>) -> !llvm.ptr
// CHECK:           %[[VAL_4:[A-Za-z0-9_]*]] = llvm.load %[[VAL_3]] : !llvm.ptr -> f64
// CHECK:           llvm.store %[[VAL_4]], %[[VAL_2]] : f64, !llvm.ptr
// CHECK:           %[[VAL_5:[A-Za-z0-9_]*]] = llvm.getelementptr %[[VAL_3]][1] : (!llvm.ptr) -> !llvm.ptr, f64
// CHECK:           %[[VAL_6:[A-Za-z0-9_]*]] = llvm.load %[[VAL_5]] : !llvm.ptr -> f64
// CHECK:           %[[VAL_7:[A-Za-z0-9_]*]] = llvm.getelementptr %[[VAL_2]][1] : (!llvm.ptr) -> !llvm.ptr, f64
// CHECK:           llvm.store %[[VAL_6]], %[[VAL_7]] : f64, !llvm.ptr
// CHECK:           %[[VAL_8:[A-Za-z0-9_]*]] = llvm.getelementptr %[[VAL_3]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(f64, f64)>, i32)>
// CHECK:           %[[VAL_9:[A-Za-z0-9_]*]] = llvm.load %[[VAL_8]] : !llvm.ptr -> i32
// CHECK:           %[[VAL_10:[A-Za-z0-9_]*]] = llvm.getelementptr %[[VAL_2]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(f64, f64)>, i32)>
// CHECK:           llvm.store %[[VAL_9]], %[[VAL_10]] : i32, !llvm.ptr
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @_ZN1DC1EOS_(
// CHECK-SAME:                           %[[VAL_0:[A-Za-z0-9_]*]]: memref<?x2xf64>,
// CHECK-SAME:                           %[[VAL_1:[A-Za-z0-9_]*]]: memref<?x2xf64>)
// CHECK:           %[[VAL_2:[A-Za-z0-9_]*]] = affine.load %[[VAL_1]][0, 0] : memref<?x2xf64>
// CHECK:           affine.store %[[VAL_2]], %[[VAL_0]][0, 0] : memref<?x2xf64>
// CHECK:           %[[VAL_3:[A-Za-z0-9_]*]] = affine.load %[[VAL_1]][0, 1] : memref<?x2xf64>
// CHECK:           affine.store %[[VAL_3]], %[[VAL_0]][0, 1] : memref<?x2xf64>
// CHECK:           return
// CHECK:         }

