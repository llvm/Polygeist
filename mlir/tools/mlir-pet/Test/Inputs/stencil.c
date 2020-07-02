float A[2000];
float B[2000];

int main(void) {

#pragma scop
  for (int t = 0; t < 500; t++)
    for (int i = 1; i < 2000 - 1; i++)
      B[i] = 0.33333 * (A[i - 1] + A[i] + A[i + 1]);
#pragma endscop
}
