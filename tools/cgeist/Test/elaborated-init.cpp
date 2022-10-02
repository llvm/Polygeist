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

// CHECK:   func.func @_Z17testArrayInitExprv()
// CHECK-NEXT:     %c4_i32 = arith.constant 4 : i32
// CHECK-NEXT:     %c3_i32 = arith.constant 3 : i32
// CHECK-NEXT:     %c2_i32 = arith.constant 2 : i32
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %0 = memref.alloca() : memref<1x!llvm.struct<(array<4 x i32>)>>
// CHECK-NEXT:     %1 = "polygeist.memref2pointer"(%0) : (memref<1x!llvm.struct<(array<4 x i32>)>>) -> !llvm.ptr<struct<(array<4 x i32>)>>
// CHECK-NEXT:     %2 = llvm.getelementptr %1[0, 0] : (!llvm.ptr<struct<(array<4 x i32>)>>) -> !llvm.ptr<array<4 x i32>>
// CHECK-NEXT:     %3 = llvm.bitcast %2 : !llvm.ptr<array<4 x i32>> to !llvm.ptr<i32>
// CHECK-NEXT:     llvm.store %c1_i32, %3 : !llvm.ptr<i32>
// CHECK-NEXT:     %4 = llvm.getelementptr %3[1] : (!llvm.ptr<i32>) -> !llvm.ptr<i32>
// CHECK-NEXT:     llvm.store %c2_i32, %4 : !llvm.ptr<i32>
// CHECK-NEXT:     %5 = llvm.getelementptr %3[2] : (!llvm.ptr<i32>) -> !llvm.ptr<i32>
// CHECK-NEXT:     llvm.store %c3_i32, %5 : !llvm.ptr<i32>
// CHECK-NEXT:     %6 = llvm.getelementptr %3[3] : (!llvm.ptr<i32>) -> !llvm.ptr<i32>
// CHECK-NEXT:     llvm.store %c4_i32, %6 : !llvm.ptr<i32>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
