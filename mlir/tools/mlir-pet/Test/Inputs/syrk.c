float C[2000][2000];
float A[2000][1200];
int main(){
float alpha, beta;
#pragma scop
  for (int i = 0; i < 2000; i++) {
    for (int j = 0; j <= i; j++)
      C[i][j] *= beta;
    for (int k = 0; k < 1200; k++) {
      for (int j = 0; j <= i; j++)
        C[i][j] += alpha * A[i][k] * A[j][k];
    }
  }
#pragma endscop
return 0;
}
