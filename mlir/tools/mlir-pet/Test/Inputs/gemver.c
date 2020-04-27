
float A[1024][1024];
float u1[1024];
float v1[1024];
float u2[1024];
float v2[1024];
float x[1024];
float w[1024];
float z[1024];
float y[1024];
float alpha = 1.0;
float beta = 1.0;

int main() {

#pragma scop

  for (int i = 0; i < 1024; i++)
    for (int j = 0; j < 1024; j++)
      A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];

  for (int i = 0; i < 1024; i++)
    for (int j = 0; j < 1024; j++)
      x[i] = x[i] + beta * A[j][i] * y[j];

  for (int i = 0; i < 1024; i++)
    x[i] = x[i] + z[i];

  for (int i = 0; i < 1024; i++)
    for (int j = 0; j < 1024; j++)
      w[i] = w[i] + alpha * A[i][j] * x[j];

#pragma endscop
  return 0;
}
