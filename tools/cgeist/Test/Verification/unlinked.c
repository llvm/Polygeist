// RUN: cgeist %s --function=kernel_correlation --raise-scf-to-affine -S | FileCheck %s
// RUN: cgeist %s --function=kernel_correlation --raise-scf-to-affine -S -memref-fullrank | FileCheck %s --check-prefix=FULLRANK

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
// FULLRANK:   func @kernel_correlation(%{{.*}}: memref<10x10xi32>)

// CHECK-LABEL:   func.func @kernel_correlation(
// CHECK-SAME:                                  %[[VAL_0:[A-Za-z0-9_]*]]: memref<?x10xi32>)  
// CHECK:           %[[VAL_1:[A-Za-z0-9_]*]] = arith.constant 9 : index
// CHECK:           affine.for %[[VAL_2:[A-Za-z0-9_]*]] = 0 to 10 {
// CHECK:             %[[VAL_3:[A-Za-z0-9_]*]] = arith.subi %[[VAL_1]], %[[VAL_2]] : index
// CHECK:             %[[VAL_4:[A-Za-z0-9_]*]] = arith.index_cast %[[VAL_3]] : index to i32
// CHECK:             affine.for %[[VAL_5:[A-Za-z0-9_]*]] = 0 to 10 {
// CHECK:               %[[VAL_6:[A-Za-z0-9_]*]] = arith.index_cast %[[VAL_5]] : index to i32
// CHECK:               %[[VAL_7:[A-Za-z0-9_]*]] = arith.addi %[[VAL_4]], %[[VAL_6]] : i32
// CHECK:               affine.store %[[VAL_7]], %[[VAL_0]][-%[[VAL_2]] + 9, %[[VAL_5]]] : memref<?x10xi32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
