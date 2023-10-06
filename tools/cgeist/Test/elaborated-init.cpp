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

// CHECK-LABEL:   func.func @_Z17testArrayInitExprv()  
// CHECK:           %[[VAL_0:[A-Za-z0-9_]*]] = arith.constant 4 : i32
// CHECK:           %[[VAL_1:[A-Za-z0-9_]*]] = arith.constant 3 : i32
// CHECK:           %[[VAL_2:[A-Za-z0-9_]*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_3:[A-Za-z0-9_]*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_4:[A-Za-z0-9_]*]] = memref.alloca() : memref<1x!llvm.struct<(array<4 x i32>)>>
// CHECK:           %[[VAL_5:[A-Za-z0-9_]*]] = "polygeist.memref2pointer"(%[[VAL_4]]) : (memref<1x!llvm.struct<(array<4 x i32>)>>) -> !llvm.ptr
// CHECK:           llvm.store %[[VAL_3]], %[[VAL_5]] : i32, !llvm.ptr
// CHECK:           %[[VAL_6:[A-Za-z0-9_]*]] = llvm.getelementptr %[[VAL_5]][1] : (!llvm.ptr) -> !llvm.ptr, i32
// CHECK:           llvm.store %[[VAL_2]], %[[VAL_6]] : i32, !llvm.ptr
// CHECK:           %[[VAL_7:[A-Za-z0-9_]*]] = llvm.getelementptr %[[VAL_5]][2] : (!llvm.ptr) -> !llvm.ptr, i32
// CHECK:           llvm.store %[[VAL_1]], %[[VAL_7]] : i32, !llvm.ptr
// CHECK:           %[[VAL_8:[A-Za-z0-9_]*]] = llvm.getelementptr %[[VAL_5]][3] : (!llvm.ptr) -> !llvm.ptr, i32
// CHECK:           llvm.store %[[VAL_0]], %[[VAL_8]] : i32, !llvm.ptr
// CHECK:           return
// CHECK:         }

