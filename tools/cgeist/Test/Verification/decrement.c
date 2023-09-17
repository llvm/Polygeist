// RUN: cgeist %s --function=* -S | FileCheck %s

int prefix_decrement(int x)
{
    return --x;
}

int postfix_decrement(int x)
{
    return x--;
}

// CHECK: func.func @prefix_decrement(%[[arg0:.+]]: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:   %[[c1_i32:.+]] = arith.constant -1 : i32
// CHECK-NEXT:   %[[V0:.+]] = arith.addi %[[arg0]], %[[c1_i32]] : i32
// CHECK-NEXT:   return %[[V0]] : i32
// CHECK-NEXT: }

// CHECK: func.func @postfix_decrement(%[[arg0:.+]]: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:   return %[[arg0]] : i32
// CHECK-NEXT: }
