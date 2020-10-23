float L[2000][2000];
float x[2000];
float b[2000];

int main() {

#pragma scop
  for (int i = 0; i < 2000; i++) {
    x[i] = b[i];
    for (int j = 0; j < i; j++)
      x[i] -= L[i][j] * x[j];
    x[i] /= L[i][i];
  }
#pragma endscop

  return 0;
}
