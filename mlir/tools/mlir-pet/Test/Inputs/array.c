void print_memref_f32(float a);

int main() {
#pragma scop 
  
  float A[2][2];
  float B[2][2];

  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
      A[i][j] = 3.0;

  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
      A[i][j] = A[i][j] + 2.0;

 for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
      B[i][j] = 3.0;

  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
      B[i][j] = A[i][j] + 2.0;

  float a = 0.0;
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
      a = a + B[i][j];

  print_memref_f32(a);

#pragma endscop
  return 0;
}
