// RUN: mlir-clang %s --function=* -S | FileCheck %s

extern "C" {

double kernel_deriche(int x, float y) {
    ([&y,x]() {
        y *= x;
    })();
    return y;
}

}

// CHECK:   func @kernel_deriche(%arg0: i32, %arg1: f32) -> f64 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %0 = llvm.alloca %c1_i64 x !llvm.struct<packed (ptr<f32>, i32, array<4 x i8>)> : (i64) -> !llvm.ptr<struct<packed (ptr<f32>, i32, array<4 x i8>)>>
// CHECK-NEXT:     %1 = llvm.alloca %c1_i64 x !llvm.struct<packed (ptr<f32>, i32, array<4 x i8>)> : (i64) -> !llvm.ptr<struct<packed (ptr<f32>, i32, array<4 x i8>)>>
// CHECK-NEXT:     %2 = llvm.getelementptr %1[%c0_i32, %c1_i32] : (!llvm.ptr<struct<packed (ptr<f32>, i32, array<4 x i8>)>>, i32, i32) -> !llvm.ptr<i32>
// CHECK-NEXT:     llvm.store %arg0, %2 : !llvm.ptr<i32>
// CHECK-NEXT:     %3 = llvm.load %1 : !llvm.ptr<struct<packed (ptr<f32>, i32, array<4 x i8>)>>
// CHECK-NEXT:     llvm.store %3, %0 : !llvm.ptr<struct<packed (ptr<f32>, i32, array<4 x i8>)>>
// CHECK-NEXT:     call @_ZZ14kernel_dericheENK3$_0clEv(%0) : (!llvm.ptr<struct<packed (ptr<f32>, i32, array<4 x i8>)>>) -> ()
// CHECK-NEXT:     %4 = arith.extf %arg1 : f32 to f64
// CHECK-NEXT:     return %4 : f64
// CHECK-NEXT:   }
// CHECK:   func private @_ZZ14kernel_dericheENK3$_0clEv(%arg0: !llvm.ptr<struct<packed (ptr<f32>, i32, array<4 x i8>)>>) attributes {llvm.linkage = #llvm.linkage<internal>} {
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %0 = llvm.getelementptr %arg0[%c0_i32, %c0_i32] : (!llvm.ptr<struct<packed (ptr<f32>, i32, array<4 x i8>)>>, i32, i32) -> !llvm.ptr<ptr<f32>>
// CHECK-NEXT:     %1 = llvm.load %0 : !llvm.ptr<ptr<f32>>
// CHECK-NEXT:     %2 = llvm.getelementptr %arg0[%c0_i32, %c1_i32] : (!llvm.ptr<struct<packed (ptr<f32>, i32, array<4 x i8>)>>, i32, i32) -> !llvm.ptr<i32>
// CHECK-NEXT:     %3 = llvm.load %2 : !llvm.ptr<i32>
// CHECK-NEXT:     %4 = arith.sitofp %3 : i32 to f32
// CHECK-NEXT:     %5 = llvm.load %1 : !llvm.ptr<f32>
// CHECK-NEXT:     %6 = arith.mulf %5, %4 : f32
// CHECK-NEXT:     llvm.store %6, %1 : !llvm.ptr<f32>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
