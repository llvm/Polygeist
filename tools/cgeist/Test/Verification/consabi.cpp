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

// CHECK:   func.func @_Z14ilaunch_kernel7QStream(%[[arg0:.+]]: !llvm.struct<(struct<(f64, f64)>, i32)>) -> !llvm.struct<(struct<(f64, f64)>, i32)> attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %[[V0:.+]] = memref.alloca() : memref<1x!llvm.struct<(struct<(f64, f64)>, i32)>>
// CHECK-NEXT:     %[[V1:.+]] = memref.cast %[[V0]] : memref<1x!llvm.struct<(struct<(f64, f64)>, i32)>> to memref<?x!llvm.struct<(struct<(f64, f64)>, i32)>>
// CHECK-NEXT:     %[[V2:.+]] = memref.alloca() : memref<1x!llvm.struct<(struct<(f64, f64)>, i32)>>
// CHECK-NEXT:     %[[V3:.+]] = memref.cast %[[V2]] : memref<1x!llvm.struct<(struct<(f64, f64)>, i32)>> to memref<?x!llvm.struct<(struct<(f64, f64)>, i32)>>
// CHECK-NEXT:     affine.store %[[arg0]], %[[V2]][0] : memref<1x!llvm.struct<(struct<(f64, f64)>, i32)>>
// CHECK-NEXT:     call @_ZN7QStreamC1EOS_(%[[V1]], %[[V3]]) : (memref<?x!llvm.struct<(struct<(f64, f64)>, i32)>>, memref<?x!llvm.struct<(struct<(f64, f64)>, i32)>>) -> ()
// CHECK-NEXT:     %[[V4:.+]] = affine.load %[[V0]][0] : memref<1x!llvm.struct<(struct<(f64, f64)>, i32)>>
// CHECK-NEXT:     return %[[V4]] : !llvm.struct<(struct<(f64, f64)>, i32)>
// CHECK-NEXT:   }
// CHECK:   func.func @_ZN7QStreamC1EOS_(%[[arg0:.+]]: memref<?x!llvm.struct<(struct<(f64, f64)>, i32)>>, %[[arg1:.+]]: memref<?x!llvm.struct<(struct<(f64, f64)>, i32)>>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:     %[[V0:.+]] = "polygeist.memref2pointer"(%[[arg0]]) : (memref<?x!llvm.struct<(struct<(f64, f64)>, i32)>>) -> !llvm.ptr<struct<(struct<(f64, f64)>, i32)>>
// CHECK-NEXT:     %[[V1:.+]] = "polygeist.memref2pointer"(%[[arg1]]) : (memref<?x!llvm.struct<(struct<(f64, f64)>, i32)>>) -> !llvm.ptr<struct<(struct<(f64, f64)>, i32)>>
// CHECK-NEXT:     %[[V2:.+]] = llvm.bitcast %[[V1]] : !llvm.ptr<struct<(struct<(f64, f64)>, i32)>> to !llvm.ptr<f64>
// CHECK-NEXT:     %[[V3:.+]] = llvm.load %[[V2]] : !llvm.ptr<f64>
// CHECK-NEXT:     %[[V4:.+]] = llvm.bitcast %[[V0]] : !llvm.ptr<struct<(struct<(f64, f64)>, i32)>> to !llvm.ptr<f64>
// CHECK-NEXT:     llvm.store %[[V3]], %[[V4]] : !llvm.ptr<f64>
// CHECK-NEXT:     %[[V5:.+]] = llvm.getelementptr %[[V2]][1] : (!llvm.ptr<f64>) -> !llvm.ptr<f64>
// CHECK-NEXT:     %[[V6:.+]] = llvm.load %[[V5]] : !llvm.ptr<f64>
// CHECK-NEXT:     %[[V7:.+]] = llvm.getelementptr %[[V4]][1] : (!llvm.ptr<f64>) -> !llvm.ptr<f64>
// CHECK-NEXT:     llvm.store %[[V6]], %[[V7]] : !llvm.ptr<f64>
// CHECK-NEXT:     %[[V8:.+]] = llvm.getelementptr %[[V1]][0, 1] : (!llvm.ptr<struct<(struct<(f64, f64)>, i32)>>) -> !llvm.ptr<i32>
// CHECK-NEXT:     %[[V9:.+]] = llvm.load %[[V8]] : !llvm.ptr<i32>
// CHECK-NEXT:     %[[V10:.+]] = llvm.getelementptr %[[V0]][0, 1] : (!llvm.ptr<struct<(struct<(f64, f64)>, i32)>>) -> !llvm.ptr<i32>
// CHECK-NEXT:     llvm.store %[[V9]], %[[V10]] : !llvm.ptr<i32>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func.func @_ZN1DC1EOS_(%[[arg0:.+]]: memref<?x2xf64>, %[[arg1:.+]]: memref<?x2xf64>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:     %[[V0:.+]] = affine.load %[[arg1]][0, 0] : memref<?x2xf64>
// CHECK-NEXT:     affine.store %[[V0]], %[[arg0]][0, 0] : memref<?x2xf64>
// CHECK-NEXT:     %[[V1:.+]] = affine.load %[[arg1]][0, 1] : memref<?x2xf64>
// CHECK-NEXT:     affine.store %[[V1]], %[[arg0]][0, 1] : memref<?x2xf64>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
