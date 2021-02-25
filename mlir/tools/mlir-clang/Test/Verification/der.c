// RUN: mlir-clang %s --function=kernel_deriche | FileCheck %s

float kernel_deriche() {
    float a2, a6;
    a2 = a6 = 2.0;//EXP_FUN(-alpha);
    return a2;
}

// CHECK:  func @kernel_deriche() -> f32 {
// CHECK-NEXT:    %cst = constant 2.000000e+00 : f64
// CHECK-NEXT:    %0 = fptrunc %cst : f64 to f32
// CHECK-NEXT:    return %0 : f32
// CHECK-NEXT:  }