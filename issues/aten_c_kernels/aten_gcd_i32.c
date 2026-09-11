/* Fixed-shape scalar specialization extracted from pinned ATen. */
#ifndef N
#define N 4096
#endif
#define ATEN_CONST __attribute__((const))
void aten_gcd_i32(int a[N], int b[N], int out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    int x = a[i] < 0 ? -a[i] : a[i];
    int y = b[i] < 0 ? -b[i] : b[i];
    while (y != 0) { int r = x % y; x = y; y = r; }
    out[i] = x;
  }
#pragma endscop
}
