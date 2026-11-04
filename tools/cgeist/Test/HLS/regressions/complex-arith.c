// Floating `_Complex` + - * / and their compound assignments on array
// elements, built the way Vitis's front end builds them for its FPGA targets
// (hls-llvm-project CGExprComplex.cpp under IsFPGAArch): the product is
// (ac - bd) + i(ad + bc) with no NaN-recovery libcall, the quotient by a
// complex ((ac + bd) + i(bc - ad)) / (cc + dd) with no __divsc3, and a real
// operand has no imaginary part. libstdc++'s std::complex<float> operators
// are these compound assignments on its `_ComplexT` member (a typedef), whose
// type the opcode cases missed: `a[i] = a[i] * a[j]` asserted "must have
// shape 1" (CIRCT W9 round 6 review, defect 2). Values: check_semantics.py
// probes against g++ (std::complex<float>/<double> +, -, *, compound forms,
// real operands): bit-exact. The quotient by a complex is Vitis's synthesis
// formula, not the C simulation's: g++ (and Vitis C simulation) call
// __divsc3, which scales the operands, so their results differ in the last
// bits (probe w9r6fix cxdiv: a mismatch, as expected).
// RUN: cgeist %s --function='*' -S --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s

typedef _Complex float CF;

// CHECK-LABEL: func.func @mul_assign(
// CHECK: affine.for %[[I:.*]] = 0 to 8 {
// CHECK-DAG: %[[A:.*]] = affine.load %arg0[%[[I]], 0] : memref<8x2xf32>
// CHECK-DAG: %[[B:.*]] = affine.load %arg0[%[[I]], 1] : memref<8x2xf32>
// CHECK-DAG: %[[C:.*]] = affine.load %arg1[%[[I]], 0] : memref<8x2xf32>
// CHECK-DAG: %[[D:.*]] = affine.load %arg1[%[[I]], 1] : memref<8x2xf32>
// CHECK-DAG: %[[AC:.*]] = arith.mulf %[[A]], %[[C]] : f32
// CHECK-DAG: %[[BD:.*]] = arith.mulf %[[B]], %[[D]] : f32
// CHECK-DAG: %[[RE:.*]] = arith.subf %[[AC]], %[[BD]] : f32
// CHECK-DAG: %[[AD:.*]] = arith.mulf %[[A]], %[[D]] : f32
// CHECK-DAG: %[[BC:.*]] = arith.mulf %[[B]], %[[C]] : f32
// CHECK-DAG: %[[IM:.*]] = arith.addf %[[AD]], %[[BC]] : f32
// CHECK: affine.store %[[RE]], %arg0[%[[I]], 0] : memref<8x2xf32>
// CHECK: affine.store %[[IM]], %arg0[%[[I]], 1] : memref<8x2xf32>
void mul_assign(CF a[8], CF b[8]) {
  for (int i = 0; i < 8; i++)
    a[i] *= b[i];
}

// `a[i] = a[i] + b[j]`: the sum is a temporary; the assignment copies its
// two parts.
// CHECK-LABEL: func.func @add_copy(
// CHECK: %[[RE:.*]] = arith.addf
// CHECK: %[[IM:.*]] = arith.addf
// CHECK: affine.store %[[RE]], %arg0[%{{.*}}, 0] : memref<8x2xf32>
// CHECK: affine.store %[[IM]], %arg0[%{{.*}}, 1] : memref<8x2xf32>
void add_copy(CF a[8], CF b[8]) {
  for (int i = 0; i < 8; i++)
    a[i] = a[i] + b[(i + 1) & 7];
}

// A real operand: `a[i] - s` keeps the imaginary part, `s - a[i]` negates
// it, `a[i] * s` scales both parts, `a[i] / s` divides both.
// CHECK-LABEL: func.func @real_ops(
// CHECK: %[[IM0:.*]] = affine.load %arg0[0, 1]
// CHECK: arith.subf %{{.*}}, %arg2 : f32
// CHECK-NOT: arith.negf
// CHECK: affine.store %[[IM0]], %arg0[0, 1]
// CHECK: arith.subf %arg2, %{{.*}} : f32
// CHECK-NEXT: arith.negf
// CHECK: arith.mulf
// CHECK-NEXT: arith.mulf
// CHECK: arith.divf
// CHECK-NEXT: arith.divf
void real_ops(CF a[4], CF b[4], float s) {
  a[0] -= s;
  a[1] = s - a[1];
  a[2] *= s;
  a[3] /= s;
}

// The quotient by a complex: no library call.
// CHECK-LABEL: func.func @div_complex(
// CHECK-NOT: call
// CHECK-COUNT-2: arith.divf
// CHECK: return
void div_complex(CF a[4], CF b[4]) {
  for (int i = 0; i < 4; i++)
    a[i] /= b[i];
}
