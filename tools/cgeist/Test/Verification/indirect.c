// RUN: cgeist %s --function=main -S | FileCheck %s
// RUN: cgeist %s --function=main -S --emit-llvm | FileCheck %s --check-prefix=LLCHECK

int square(int x) {
  return x*x;
}
int meta(int (*f)(int), int x) {
  return f(x);
}

int printf(const char*, ...);
int main() {
  printf("sq(%d)=%d\n", 3, meta(square, 3));
  return 0;
}

// CHECK:   func.func @main() -> i32 attributes
// CHECK-DAG:     %[[c3_i32:.+]] = arith.constant 3 : i32
// CHECK-DAG:     %[[c0_i32:.+]] = arith.constant 0 : i32
// CHECK-NEXT:     %[[V0:.+]] = llvm.mlir.addressof @str0 : !llvm.ptr<array<11 x i8>>
// CHECK-NEXT:     %[[V1:.+]] = llvm.getelementptr %[[V0]][0, 0] : (!llvm.ptr<array<11 x
// i8>>) -> !llvm.ptr<i8> CHECK-NEXT:     %[[V2:.+]] = polygeist.get_func @square :
// !llvm.ptr<func<i32 (i32)>> CHECK-NEXT:     %[[V3:.+]] =
// "polygeist.pointer2memref"(%[[V2]]) : (!llvm.ptr<func<i32 (i32)>>) ->
// memref<?x!llvm.func<i32 (i32)>> CHECK-NEXT:     %[[V4:.+]] = call @meta(%[[V3:.+]], %[[c3_i32:.+]])
// : (memref<?x!llvm.func<i32 (i32)>>, i32) -> i32 CHECK-NEXT:     %[[V5:.+]] =
// llvm.call @printf(%[[V1]], %[[c3_i32]], %[[V4]]) : (!llvm.ptr<i8>, i32, i32) -> i32
// CHECK-NEXT:     return %[[c0_i32]] : i32
// CHECK-NEXT:   }

// LLCHECK: define i32 @main()
// LLCHECK:   %[[V1:.+]] = call i32 @meta(i32 (i32)* @square, i32 3)
// LLCHECK:   %[[V2:.+]] = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @str0, i32 0, i32 0), i32 3, i32 %[[V1]])
// LLCHECK:   ret i32 0
// LLCHECK: }
