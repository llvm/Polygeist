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

// CHECK:   func.func @_Z14ilaunch_kernel7QStream(%arg0: !llvm.struct<(struct<(f64, f64)>, i32)>) -> !llvm.struct<(struct<(f64, f64)>, i32)> attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     0 = memref.alloca() : memref<1x!llvm.struct<(struct<(f64, f64)>, i32)>>
// CHECK-NEXT:     %1 = memref.cast %0 : memref<1x!llvm.struct<(struct<(f64, f64)>, i32)>> to memref<?x!llvm.struct<(struct<(f64, f64)>, i32)>>
// CHECK-NEXT:     %2 = memref.alloca() : memref<1x!llvm.struct<(struct<(f64, f64)>, i32)>>
// CHECK-NEXT:     %3 = memref.cast %2 : memref<1x!llvm.struct<(struct<(f64, f64)>, i32)>> to memref<?x!llvm.struct<(struct<(f64, f64)>, i32)>>
// CHECK-NEXT:     ffine.store %arg0, %2[0] : memref<1x!llvm.struct<(struct<(f64, f64)>, i32)>>
// CHECK-NEXT:     all @_ZN7QStreamC1EOS_(%1, %3) : (memref<?x!llvm.struct<(struct<(f64, f64)>, i32)>>, memref<?x!llvm.struct<(struct<(f64, f64)>, i32)>>) -> ()
// CHECK-NEXT:     4 = affine.load %0[0] : memref<1x!llvm.struct<(struct<(f64, f64)>, i32)>>
// CHECK-NEXT:     return %4 : !llvm.struct<(struct<(f64, f64)>, i32)>
// CHECK-NEXT:   }
// CHECK:   func.func @_ZN7QStreamC1EOS_(%arg0: memref<?x!llvm.struct<(struct<(f64, f64)>, i32)>>, %arg1: memref<?x!llvm.struct<(struct<(f64, f64)>, i32)>>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:     %0 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(struct<(f64, f64)>, i32)>>) -> !llvm.ptr<struct<(struct<(f64, f64)>, i32)>>
// CHECK-NEXT:     1 = "polygeist.memref2pointer"(%arg1) : (memref<?x!llvm.struct<(struct<(f64, f64)>, i32)>>) -> !llvm.ptr<struct<(struct<(f64, f64)>, i32)>>
// CHECK-NEXT:     2 = llvm.bitcast %1 : !llvm.ptr<struct<(struct<(f64, f64)>, i32)>> to !llvm.ptr<f64>
// CHECK-NEXT:     %3 = llvm.load %2 : !llvm.ptr<f64>
// CHECK-NEXT:     4 = llvm.bitcast %0 : !llvm.ptr<struct<(struct<(f64, f64)>, i32)>> to !llvm.ptr<f64>
// CHECK-NEXT:     llvm.store %3, %4 : !llvm.ptr<f64>
// CHECK-NEXT:     5 = llvm.getelementptr %2[1] : (!llvm.ptr<f64>) -> !llvm.ptr<f64>
// CHECK-NEXT:     %6 = llvm.load %5 : !llvm.ptr<f64>
// CHECK-NEXT:     %7 = llvm.getelementptr %4[1] : (!llvm.ptr<f64>) -> !llvm.ptr<f64>
// CHECK-NEXT:     llvm.store %6, %7 : !llvm.ptr<f64>
// CHECK-NEXT:     8 = llvm.getelementptr %1[0, 1] : (!llvm.ptr<struct<(struct<(f64, f64)>, i32)>>) -> !llvm.ptr<i32>
// CHECK-NEXT:     %9 = llvm.load %8 : !llvm.ptr<i32>
// CHECK-NEXT:     %10 = llvm.getelementptr %0[0, 1] : (!llvm.ptr<struct<(struct<(f64, f64)>, i32)>>) -> !llvm.ptr<i32>
// CHECK-NEXT:     lvm.store %9, %10 : !llvm.ptr<i32>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func.func @_ZN1DC1EOS_(%arg0: memref<?x2xf64>, %arg1: memref<?x2xf64>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:     %0 = affine.load %arg1[0, 0] : memref<?x2xf64>
// CHECK-NEXT:     affine.store %0, %arg0[0, 0] : memref<?x2xf64>
// CHECK-NEXT:     %1 = affine.load %arg1[0, 1] : memref<?x2xf64>
// CHECK-NEXT:     affine.store %1, %arg0[0, 1] : memref<?x2xf64>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
