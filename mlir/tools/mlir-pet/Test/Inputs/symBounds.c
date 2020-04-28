float A[1024][1024];
float B[1024][1024];
float C[1024][1024];

int main(void) {

#pragma scop
  for (int i = 0; i < 1024; i++) {
    for (int j = 0; j < 1024; j++)
      for (int k = 0; k < 1024; k++) {
        for (int j = k; j < i; j++)
          C[i][j] += A[i][k] * B[k][j];
      }
  }
#pragma endscop

  return 0;
}
