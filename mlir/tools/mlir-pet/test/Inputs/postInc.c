float A[1024];

int main(void) {

#pragma scop
  for (int i = 0; i < 1024; i++) {
    A[i]++;
  }
#pragma endscop
  return A[100];
}
