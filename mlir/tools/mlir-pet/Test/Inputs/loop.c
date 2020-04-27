void print_memref_f32(float a);

int main() {
#pragma scop
  float a = 0.0;
  for (int k = 0; k < 100; k++)
    a = a + 5.0;

  print_memref_f32(a);
#pragma endscop
  return a;
}
