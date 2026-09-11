// RUN: cgeist %s --function=apply -S | FileCheck %s

__attribute__((const)) float special_const(float);
__attribute__((pure)) float special_pure(float);

void apply(float *input, float *output) {
  output[0] = special_const(input[0]) + special_pure(input[1]);
}

// CHECK: func.func private @special_const(f32) -> f32
// CHECK-SAME: polygeist.pure
// CHECK: func.func private @special_pure(f32) -> f32
// CHECK-SAME: polygeist.pure
