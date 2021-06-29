// RUN: mlir-clang %s --function=kernel_correlation --raise-scf-to-affine | FileCheck %s

#define DATA_TYPE double

#define N 10

#define match(b1, b2) (((b1)+(b2)) == 3 ? 1 : 0)
#define max_score(s1, s2) ((s1 >= s2) ? s1 : s2)

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
void kernel_correlation(int table[N][N]) {
 for (int i = 9; i >= 0; i--) {
  for (int j=0; j<10; j++) {
      table[i][j] = i+j;
    }
  }
}

// CHECK:   func @kernel_correlation(%arg0: memref<?x10xi32>) {
// CHECK-NEXT:     %c9_i32 = constant 9 : i32
// CHECK-NEXT:     affine.for %arg1 = 0 to 10 iter_args(%arg2 = %c9_i32) {
// CHECK-NEXT:       %0 = index_cast %arg1 : i32 to index
// CHECK-NEXT:       %1 = subi %c9_i32, %0 : i32
// CHECK-NEXT:       affine.for %arg3 = 0 to 10 {
// CHECK-NEXT:         %3 = index_cast %arg3 : index to i32
// CHECK-NEXT:         %4 = addi %arg2, %3 : i32
// CHECK-NEXT:         memref.store %4, %arg0[%1, %arg3] : memref<?x10xi32>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }