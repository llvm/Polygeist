// RUN: mlir-clang %s main | FileCheck %s

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

// CHECK: #map0 = affine_map<(d0) -> (d0)>

// CHECK: affine.for %arg0 = 0 to 1024 {
// CHECK-NEXT: %{{.*}} = index_cast %arg0 : index to i32
// CHECK-NEXT: affine.for %arg1 = 0 to 1024 {
// CHECK-NEXT: %{{.*}} = index_cast %arg1 : index to i32
// CHECK-NEXT: affine.for %arg2 = 0 to 1024 {
// CHECK-NEXT: %{{.*}} = index_cast %arg2 : index to i32 

// CHECK: [[DIM1:%[0-9]+]] = affine.apply #map0(%arg0)
// CHECK-NEXT: [[DIM2:%[0-9]+]] = affine.apply #map0(%arg1)
// CHECK-NEXT: affine.store %{{.*}}, %{{.*}}[[DIM1]], [[DIM2]]
