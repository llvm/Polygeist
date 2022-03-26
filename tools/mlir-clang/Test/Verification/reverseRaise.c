// RUN: mlir-clang %s --function=kernel_correlation --raise-scf-to-affine -S | FileCheck %s

#define DATA_TYPE double

#define SCALAR_VAL(x) ((double)x)

void use(int i);

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
void kernel_correlation(int start, int end) {
  for (int i = end; i >= start; i--) {
    use(i);
  }
}

// CHECK: kernel_correlation
// CHECK-DAG:     %c-1_i32 = arith.constant -1 : i32
// CHECK-DAG:     %c1 = arith.constant 1 : index
// CHECK-NEXT:     %0 = arith.index_cast %arg1 : i32 to index
// CHECK-NEXT:     %1 = arith.addi %0, %c1 : index
// CHECK-NEXT:     %2 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:     affine.for %arg2 = %2 to %1 {
// CHECK-NEXT:       %3 = arith.index_cast %arg2 : index to i32
// CHECK-NEXT:       %4 = arith.subi %3, %arg0 : i32
// CHECK-NEXT:       %5 = arith.muli %4, %c-1_i32 : i32
// CHECK-NEXT:       %6 = arith.addi %arg1, %5 : i32
// CHECK-NEXT:       call @use(%6) : (i32) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
