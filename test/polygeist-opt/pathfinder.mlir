// RUN: polygeist-opt --convert-polygeist-to-llvm %s | FileCheck %s

module {
  func @funcasfda(%arg0: memref<256x3xi32>, %arg1 : index) -> memref<3xi32> {
    %87 = "polygeist.subindex"(%arg0, %arg1) : (memref<256x3xi32>, index) -> memref<3xi32>
    return %87 : memref<3xi32>
  }
}

// CHECK:   llvm.func @funcasfda(%arg0: !llvm.ptr<i32>, %arg1: i64) -> !llvm.ptr<i32>
// CHECK-NEXT:     %0 = llvm.mlir.undef : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:     %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:     %2 = llvm.insertvalue %arg0, %1[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:     %3 = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:     %4 = llvm.insertvalue %3, %2[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:     %5 = llvm.mlir.constant(256 : index) : i64
// CHECK-NEXT:     %6 = llvm.insertvalue %5, %4[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:     %7 = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT:     %8 = llvm.insertvalue %7, %6[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:     %9 = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT:     %10 = llvm.insertvalue %9, %8[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:     %11 = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:     %12 = llvm.insertvalue %11, %10[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:     %13 = llvm.extractvalue %12[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:     %14 = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:     %15 = llvm.mul %arg1, %14  : i64
// CHECK-NEXT:     %16 = llvm.extractvalue %12[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:     %17 = llvm.extractvalue %12[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:     %18 = llvm.getelementptr %13[%15] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
// CHECK-NEXT:     %19 = llvm.extractvalue %12[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:     %20 = llvm.mlir.undef : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:     %21 = llvm.insertvalue %19, %20[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:     %22 = llvm.insertvalue %18, %21[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:     %23 = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:     %24 = llvm.insertvalue %23, %22[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:     %25 = llvm.insertvalue %16, %24[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:     %26 = llvm.insertvalue %17, %25[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:     %27 = llvm.extractvalue %26[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:     llvm.return %27 : !llvm.ptr<i32>
