// RUN: cgeist %s --function=* -S | FileCheck %s

typedef unsigned char byte_t;
typedef signed char signed_byte_t;

float unsigned_scalar_to_float(byte_t value) { return (float)value; }
float unsigned_pointer_to_float(const byte_t *value) { return (float)value[0]; }
float signed_scalar_to_float(signed_byte_t value) { return (float)value; }
byte_t float_to_unsigned(float value) { return (byte_t)value; }
signed_byte_t float_to_signed(float value) { return (signed_byte_t)value; }
int load_by_unsigned_byte(const int *values, byte_t index) {
  return values[index];
}

int main(int argc, char **argv) {
  (void)argv;
  byte_t value = (byte_t)(argc + 199);
  return unsigned_scalar_to_float(value) == 200.0f ? 0 : 1;
}

// CHECK-LABEL: func.func @unsigned_scalar_to_float
// CHECK: arith.uitofp {{.*}} : i8 to f32
// CHECK-LABEL: func.func @unsigned_pointer_to_float
// CHECK: arith.uitofp {{.*}} : i8 to f32
// CHECK-LABEL: func.func @signed_scalar_to_float
// CHECK: arith.sitofp {{.*}} : i8 to f32
// CHECK-LABEL: func.func @float_to_unsigned
// CHECK: arith.fptoui {{.*}} : f32 to i8
// CHECK-LABEL: func.func @float_to_signed
// CHECK: arith.fptosi {{.*}} : f32 to i8
// CHECK-LABEL: func.func @load_by_unsigned_byte
// CHECK: arith.index_castui {{.*}} : i8 to index
