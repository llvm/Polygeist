const int N = 16;

void my_kernel(int a[N], int b[N], int c[N]) {
  for (int i = 0; i < N; i++) {
    c[i] = a[i] + b[i];
  }
}