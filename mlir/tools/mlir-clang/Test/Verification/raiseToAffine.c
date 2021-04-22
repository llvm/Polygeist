// RUN: mlir-clang %s --function=matmul --raise-scf-to-affine | FileCheck %s

#define N 200
#define M 300
#define K 400
#define DATA_TYPE float

void matmul(DATA_TYPE A[N][K], DATA_TYPE B[K][M], DATA_TYPE C[N][M]) {
  int i, j, k;
  // CHECK: affine.for
  for (i = 0; i < N; i++)
    // CHECK-NEXT: affine.for
    for (j = 0; j < M; j++)
      // CHECK-NEXT: affine.for
      for (k = 0; k < K; k++)
        // CHECK: {{.*}} = affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x400xf32>
        // CHECK: {{.*}} = affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x300xf32>
        // CHECK: {{.*}} = mulf
        // CHECK: {{.*}} = affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x300xf32>
        // CHECK: {{.*}} = addf
        // CHECK: affine.store {{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x300xf32>
        C[i][j] += A[i][k] * B[k][j];
}
