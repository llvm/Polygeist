float A[1024][1024];
float B[1024][1024];
float C[1024][1024];
float beta = 1.0;
float alpha = 1.0;

int main(void) {

#pragma scop
  for (int i = 0; i < 1024; i++) {
    for (int j = 0; j < 1022; j++)
      C[i][j] *= beta;
    for (int k = 0; k < 1024; k++) {
      for (int j = 0; j < 151; j++)
        C[i][j] += alpha * A[i][k] * B[k][j];
    }
  }
#pragma endscop

  return 0;
}
