// RUN: mlir-clang %s --function=whiletofor -S | FileCheck %s

void use(int a[100][100]);

void whiletofor() {
  int a[100][100];
  int t = 7;
  int i, j;

  for (i = 0; i < 100; i++)
    for (j = 0; j < 100; j++) {
      if (t % 20 == 0)
        a[i][j] = 2;
      else
        a[i][j] = 3;
      t++;
    }

  use(a);
}

// TODO redundant for elim
// CHECK: func @whiletofor()
// CHECK-DAG:      %c1 = constant 1 : index
// CHECK-DAG:      %c0 = constant 0 : index
// CHECK-DAG:      %c100 = constant 100 : index
// CHECK-DAG:      %c7_i32 = constant 7 : i32
// CHECK-DAG:      %c0_i32 = constant 0 : i32
// CHECK-DAG:      %c20_i32 = constant 20 : i32
// CHECK-DAG:      %c2_i32 = constant 2 : i32
// CHECK-DAG:      %c3_i32 = constant 3 : i32
// CHECK-DAG:      %c1_i32 = constant 1 : i32
// CHECK-DAG:      %0 = memref.alloca() : memref<100x100xi32>
// CHECK-NEXT:      %1 = scf.for %arg0 = %c0 to %c100 step %c1 iter_args(%arg1 = %c7_i32) -> (i32) {
// CHECK-NEXT:        %3 = scf.for %arg2 = %c0 to %c100 step %c1 iter_args(%arg3 = %arg1) -> (i32) {
// CHECK-NEXT:          %4 = index_cast %arg2 : index to i32
// CHECK-NEXT:          %5 = addi %4, %arg1 : i32
// CHECK-NEXT:          %[[i4:.+]] = remi_signed %5, %c20_i32 : i32
// CHECK-NEXT:          %[[i5:.+]] = cmpi eq, %[[i4]], %c0_i32 : i32
// CHECK-NEXT:          scf.if %[[i5]] {
// CHECK-NEXT:            memref.store %c2_i32, %0[%arg0, %arg2] : memref<100x100xi32>
// CHECK-NEXT:          } else {
// CHECK-NEXT:            memref.store %c3_i32, %0[%arg0, %arg2] : memref<100x100xi32>
// CHECK-NEXT:          }
// CHECK-NEXT:          %[[i6:.+]] = addi %5, %c1_i32 : i32
// CHECK-NEXT:          scf.yield %[[i6]] : i32
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield %3 : i32
// CHECK-NEXT:      }
// CHECK-NEXT:      %2 = memref.cast %0 : memref<100x100xi32> to memref<?x100xi32>
// CHECK-NEXT:      call @use(%2) : (memref<?x100xi32>) -> ()
// CHECK-NEXT:      return
// CHECK-NEXT:    }
