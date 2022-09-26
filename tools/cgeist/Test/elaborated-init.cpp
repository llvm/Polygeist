// RUN: cgeist %s --function='*' -S | FileCheck %s
struct A {
  using TheType = int[4];
};

void testArrayInitExpr()
{
  A::TheType a{1,2,3,4};
  auto l = [a]{
  };
}

// CHECK:   func.func private @_ZZ17testArrayInitExprvEN3$_0C1EOS_(%arg0: memref<?x!llvm.struct<(array<4 x i32>)>>, %arg1: memref<?x!llvm.struct<(array<4 x i32>)>>) attributes {llvm.linkage = #llvm.linkage<internal>} {
// CHECK-NEXT:     %0 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(array<4 x i32>)>>) -> !llvm.ptr<struct<(array<4 x i32>)>>
// CHECK-NEXT:     %1 = llvm.getelementptr %0[0, 0] : (!llvm.ptr<struct<(array<4 x i32>)>>) -> !llvm.ptr<array<4 x i32>>
// CHECK-NEXT:     %2 = "polygeist.memref2pointer"(%arg1) : (memref<?x!llvm.struct<(array<4 x i32>)>>) -> !llvm.ptr<struct<(array<4 x i32>)>>
// CHECK-NEXT:     %3 = llvm.getelementptr %2[0, 0] : (!llvm.ptr<struct<(array<4 x i32>)>>) -> !llvm.ptr<array<4 x i32>>
// CHECK-NEXT:     %4 = llvm.bitcast %3 : !llvm.ptr<array<4 x i32>> to !llvm.ptr<i32>
// CHECK-NEXT:     %5 = llvm.bitcast %1 : !llvm.ptr<array<4 x i32>> to !llvm.ptr<i32>
// CHECK-NEXT:     affine.for %arg2 = 0 to 4 {
// CHECK-NEXT:       %6 = arith.index_cast %arg2 : index to i32
// CHECK-NEXT:       %7 = llvm.getelementptr %4[%6] : (!llvm.ptr<i32>, i32) -> !llvm.ptr<i32>
// CHECK-NEXT:       %8 = llvm.load %7 : !llvm.ptr<i32>
// CHECK-NEXT:       %9 = llvm.getelementptr %5[%6] : (!llvm.ptr<i32>, i32) -> !llvm.ptr<i32>
// CHECK-NEXT:       llvm.store %8, %9 : !llvm.ptr<i32>
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
