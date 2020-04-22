void print_memref_f32(float a);
int main() {

#pragma scop

  float a = 0.0;

  for (int i = 0; i < 1024; i++) {
    a++;
  }

  print_memref_f32(a);

  int b = 0;
  b++;

#pragma endscop
  return 0;
}
