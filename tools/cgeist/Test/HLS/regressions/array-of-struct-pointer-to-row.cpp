// A pointer to a row of a 2-D array of structs (PE (*q)[3] = pes, the array
// stored as memref<3x3x3xi32> with the struct's fields as the last
// dimension) is not a pointer to an int array: its array-to-pointer decay
// must not take row 0 of the struct array (W9 round 8 review fixes: the
// decay of 6b7c55d6 matched memref<3x3x3xi32> against the extents [3, 3]
// and wrote every PE into pes[0]; probes ps1 / ps4 are bit-exact again).
// RUN: cgeist %s %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s
// CHECK-LABEL: func.func @_Z3topPiS_
// CHECK: affine.store %{{.*}}, %[[A:.*]][%arg2, %arg3, 0] : memref<3x3x3xi32>
// CHECK: affine.load %[[A]][%arg2, 1, 0] : memref<3x3x3xi32>
// CHECK: affine.load %[[A]][%arg2, 2, 2] : memref<3x3x3xi32>
struct PE { int a; int b; int c; };
int top(int in[9], int out[9]) {
  PE pes[3][3];
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      pes[i][j].a = in[i * 3 + j];
      pes[i][j].b = i;
      pes[i][j].c = j;
    }
  PE (*q)[3] = pes;
  for (int i = 0; i < 3; i++)
    out[i] = q[i][1].a + q[i][2].c;
  return 0;
}
