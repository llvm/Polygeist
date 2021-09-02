// RUN: mlir-clang %s --function=kernel_nussinov -S | FileCheck %s
#define N 5500
#define max_score(s1, s2) ((s1 >= s2) ? s1 : s2)

// CHECK: @kernel_nussinov(%arg0: i32, %arg1: memref<?xi32>)
// CHECK-NEXT:    affine.for %arg2 = 1 to 5500 {
// CHECK-NEXT:      affine.if #set(%arg2) {
// CHECK-NEXT:        %0 = affine.load %arg1[%arg2] : memref<?xi32>
// CHECK-NEXT:        %1 = affine.load %arg1[%arg2 - 1] : memref<?xi32>
// CHECK-NEXT:        %2 = cmpi sge, %0, %1 : i32
// CHECK-NEXT:        %3 = select %2, %0, %1 : i32
// CHECK-NEXT:        affine.store %3, %arg1[%arg2] : memref<?xi32>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }

void kernel_nussinov(int n, int table[N])
{
  int j;

#pragma scop
  for (j=1; j<N; j++) {

   if (j-1>=0)
      table[j] = max_score(table[j], table[j-1]);

 }
#pragma endscop

}
