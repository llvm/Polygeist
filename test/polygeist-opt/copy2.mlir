// RUN: polygeist-opt --canonicalize --split-input-file %s | FileCheck %s

module  {
  func private @_ZN11ACUDAStreamC1EOS_(%arg0: !llvm.ptr<struct<(struct<(i32, i32)>)>>, %arg1: !llvm.ptr<struct<(struct<(i32, i32)>)>>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
	%c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = llvm.getelementptr %arg0[%c0_i32, %c0_i32] : (!llvm.ptr<struct<(struct<(i32, i32)>)>>, i32, i32) -> !llvm.ptr<struct<(i32, i32)>>
    %1 = llvm.getelementptr %arg1[%c0_i32, %c0_i32] : (!llvm.ptr<struct<(struct<(i32, i32)>)>>, i32, i32) -> !llvm.ptr<struct<(i32, i32)>>
    %2 = "polygeist.pointer2memref"(%0) : (!llvm.ptr<struct<(i32, i32)>>) -> memref<?x2xi32>
    %3 = "polygeist.pointer2memref"(%1) : (!llvm.ptr<struct<(i32, i32)>>) -> memref<?x2xi32>
	%a0 = memref.load %3[%c0, %c0] : memref<?x2xi32>
    memref.store %a0, %2[%c0, %c0] : memref<?x2xi32>
    %a1 = memref.load %3[%c0, %c1] : memref<?x2xi32>
    memref.store %a1, %2[%c0, %c1] : memref<?x2xi32>
    return
  }
}

// CHECK:   func private @_ZN11ACUDAStreamC1EOS_(%arg0: !llvm.ptr<struct<(struct<(i32, i32)>)>>, %arg1: !llvm.ptr<struct<(struct<(i32, i32)>)>>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-DAG:     %c0_i32 = arith.constant 0 : i32
// CHECK-DAG:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %0 = llvm.bitcast %arg1 : !llvm.ptr<struct<(struct<(i32, i32)>)>> to !llvm.ptr<i32>
// CHECK-NEXT:     %1 = llvm.getelementptr %0[%c0_i32] : (!llvm.ptr<i32>, i32) -> !llvm.ptr<i32>
// CHECK-NEXT:     %2 = llvm.load %1 : !llvm.ptr<i32>
// CHECK-NEXT:     %3 = llvm.bitcast %arg0 : !llvm.ptr<struct<(struct<(i32, i32)>)>> to !llvm.ptr<i32>
// CHECK-NEXT:     %4 = llvm.getelementptr %3[%c0_i32] : (!llvm.ptr<i32>, i32) -> !llvm.ptr<i32>
// CHECK-NEXT:     llvm.store %2, %4 : !llvm.ptr<i32>
// CHECK-NEXT:     %5 = llvm.bitcast %arg1 : !llvm.ptr<struct<(struct<(i32, i32)>)>> to !llvm.ptr<i32>
// CHECK-NEXT:     %6 = llvm.getelementptr %5[%c1_i32] : (!llvm.ptr<i32>, i32) -> !llvm.ptr<i32>
// CHECK-NEXT:     %7 = llvm.load %6 : !llvm.ptr<i32>
// CHECK-NEXT:     %8 = llvm.bitcast %arg0 : !llvm.ptr<struct<(struct<(i32, i32)>)>> to !llvm.ptr<i32>
// CHECK-NEXT:     %9 = llvm.getelementptr %8[%c1_i32] : (!llvm.ptr<i32>, i32) -> !llvm.ptr<i32>
// CHECK-NEXT:     llvm.store %7, %9 : !llvm.ptr<i32>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
