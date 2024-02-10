void kernel(float A[64][64], int n) {
  int i, j;
  #pragma scop
  for (i = 1; i < n; i ++) {
    for (j = 1; j < n; j ++) {
      A[i][j] = A[i-1][j] + A[i][j-1]; 
    }
  }
  #pragma endscop
}

void main() {
  int N = 64;
  float A[N][N];

  kernel(A, N);
}