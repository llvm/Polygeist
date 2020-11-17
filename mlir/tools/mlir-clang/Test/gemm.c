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

// CHECK: affine.for %arg0 = 0 to 1024 {
// CHECK-NEXT: affine.for %arg1 = 0 to 1024 {
// CHECK-NEXT: affine.for %arg2 = 0 to 1024 {
// CHECK: affine.store %{{.*}}, %{{.*}}[%arg0, %arg1] : memref<1024x1024xf32>
