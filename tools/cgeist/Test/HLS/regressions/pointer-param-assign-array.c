// Assigning an array to a pointer variable (`output = out[1];` with a
// parameter `int output[16]`, which C treats as `int *output`): the decayed
// row is a memref of static shape, the variable holds a memref<?xi32>.
// cgeist stored the one into the other unchanged (assertion `expect same
// type` in ValueCategory::store, forgebench mult_op_module_mm / mmv:
// `output = out[0];`). The row is now cast to the variable's type (W8b).
// RUN: cgeist %s %resourcedir --function='*' -S --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s

// CHECK-LABEL: func.func @f(
// CHECK-NOT:     memref.store {{.*}}memref<1xmemref
// CHECK:         return %{{.*}} : i32
void g(int a[16][32]) {
  for (int i = 0; i < 16; i++)
    for (int k = 0; k < 32; k++)
      a[i][k] = i + k;
}
int f(int output[16]) {
  int out[16][32];
  g(out);
  output = out[1];
  return output[3];
}
