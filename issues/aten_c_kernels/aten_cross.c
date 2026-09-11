/* aten::cross over the final dimension of length three.
 * Upstream family: aten/src/ATen/native/Cross.cpp.
 */
#define N 64

void aten_cross(float a[N][3], float b[N][3], float out[N][3]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    out[i][0] = a[i][1] * b[i][2] - a[i][2] * b[i][1];
    out[i][1] = a[i][2] * b[i][0] - a[i][0] * b[i][2];
    out[i][2] = a[i][0] * b[i][1] - a[i][1] * b[i][0];
  }
#pragma endscop
}
