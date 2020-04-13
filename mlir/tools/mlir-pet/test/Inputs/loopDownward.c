float A[1024];

int main(void) {

#pragma scop
  for (int i = 1023; i >= 0; i--) {
    A[i]++;
  }
#pragma endscop
  return A[100];
}
