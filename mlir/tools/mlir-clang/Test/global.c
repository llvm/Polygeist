// RUN: mlir-clang %s %stdinclude | FileCheck %s

float A[64][32];

int main() {
    #pragma scop
    for (int i = 0; i < 64; i++)
    for (int j = 0; j < 32; j++)
        A[i][j] = 3.0;
    #pragma endscop
    return 0;
}

// CHECK:  global_memref "private" @A : memref<64x32xf32>
// CHECK-NEXT:  func @main() -> i32 {
// CHECK-NEXT:    %c0_i32 = constant 0 : i32
// CHECK-NEXT:    %cst = constant 3.000000e+00 : f64
// CHECK-NEXT:    affine.for %arg0 = 0 to 64 {
// CHECK-NEXT:      affine.for %arg1 = 0 to 32 {
// CHECK-NEXT:        %0 = get_global_memref @A : memref<64x32xf32>
// CHECK-NEXT:        %1 = fptrunc %cst : f64 to f32
// CHECK-NEXT:        affine.store %1, %0[%arg0, %arg1] : memref<64x32xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return %c0_i32 : i32
// CHECK-NEXT:  }