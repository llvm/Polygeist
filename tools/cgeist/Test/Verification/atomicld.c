// RUN: cgeist %s %stdinclude --function=ld -S | FileCheck %s
// RUN: cgeist %s %stdinclude --function=ld -S -emit-llvm | FileCheck %s --check-prefix=LLVM

int ld(int* x, int i) {
  int res;
  __atomic_load(&x[i], &res, 0);
  return res;
}

// CHECK:   func.func @ld(%arg0: memref<?xi32>, %arg1: i32) -> i32 
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %0 = arith.index_cast %arg1 : i32 to index
// CHECK-NEXT:     %1 = memref.atomic_rmw addi %c0_i32, %arg0[%0] : (i32, memref<?xi32>) -> i32
// CHECK-NEXT:     return %1 : i32
// CHECK-NEXT:   }

// LLVM: define i32 @ld(ptr %0, i32 %1)
// LLVM:         %[[VAL_0:[A-Za-z0-9_]*]] = sext i32 %1 to i64
// LLVM:         %[[VAL_2:[A-Za-z0-9_]*]] = getelementptr i32, ptr %0, i64 %[[VAL_0]]
// LLVM:         %[[VAL_4:[A-Za-z0-9_]*]] = atomicrmw add ptr %[[VAL_2]], i32 0 acq_rel, align 4
// LLVM:         ret i32 %[[VAL_4]]
