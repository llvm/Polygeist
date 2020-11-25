// RUN: mlir-clang %s | FileCheck %s

int main(void) {

  float A[1024][1024];
  float B[1024][1024];
  float C[1024][1024];
  float beta = 1.0;
  float alpha = 1.0;


  for (int i = 0; i < 1024; i++)
    for (int j = 0; j < 1024; j++)
      C[i][j] = 0;

#pragma scop
  for (int i = 0; i < 1024; i++) 
    for (int j = 0; j < 1024; j++)
      for (int k = 0; k < 1024; k++)
        C[i][j] += alpha * A[i][k] * B[k][j];
#pragma endscop

  return 0;
}

// CHECK:       %0 = alloca() : memref<1024x1024xf32>
// CHECK-NEXT:  %1 = alloca() : memref<1024x1024xf32>
// CHECK-NEXT:  %2 = alloca() : memref<1024x1024xf32>

// CHECK:       affine.for %arg0 = 0 to 1024 {
// CHECK-NEXT:    affine.for %arg1 = 0 to 1024 {
// CHECK-NEXT:      %13 = affine.load %2[%arg0, %arg1] : memref<1024x1024xf32>
// CHECK-NEXT:      affine.for %arg2 = 0 to 1024 {
// CHECK-NEXT:        %14 = affine.load %0[%arg0, %arg2] : memref<1024x1024xf32>
// CHECK-NEXT:        %15 = mulf %3, %14 : f32
// CHECK-NEXT:        %16 = affine.load %1[%arg2, %arg1] : memref<1024x1024xf32>
// CHECK-NEXT:        %17 = mulf %15, %16 : f32
// CHECK-NEXT:        %18 = addf %13, %17 : f32
// CHECK-NEXT:        affine.store %18, %2[%arg0, %arg1] : memref<1024x1024xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }

