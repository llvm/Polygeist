// RUN: cgeist %s --function=* -S | FileCheck %s

typedef struct {
  double real;
  double imag;
} complex_pair;

void assign_vla_struct(int leading_dimension, void *storage,
                       complex_pair source[1], int row, int column) {
  complex_pair(*view)[leading_dimension] =
      (complex_pair(*)[leading_dimension])storage;
  view[row][column] = source[0];
}

int initialize_short_strings(void) {
  const char names[2][8] = {"a", "bc"};
  return names[0][7] + names[1][7];
}

// CHECK-LABEL: func.func @assign_vla_struct
// CHECK: memref.copy {{.*}}, {{.*}} : memref<2xf64> to memref<?xf64>

// CHECK-LABEL: func.func @initialize_short_strings
// CHECK: return {{.*}} : i32
