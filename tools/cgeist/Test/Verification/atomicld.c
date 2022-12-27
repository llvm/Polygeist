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

// LLVM: define i32 @ld(i32* %0, i32 %1)
// LLVM-NEXT:   %3 = sext i32 %1 to i64
// LLVM-NEXT:   %4 = getelementptr i32, i32* %0, i64 %3,
// LLVM-NEXT:   %5 = atomicrmw add i32* %4, i32 0 acq_rel, align 4
// LLVM-NEXT:   ret i32 %5, !dbg !10
// LLVM-NEXT: }


