void print_memref_f32(float a);

int main() {
#pragma scop 
  int k;
  k = 23; 
  
  float i = 200.0;
  print_memref_f32(i);

#pragma endscop
  return 0;
}
