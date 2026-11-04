// The callee of auto-inline-cross-unit.cpp, in a translation unit of its own.
void big(int a[256], int b[256]) {
  int loc[256];
  for (int i = 0; i < 256; i++) loc[i] = a[i] * 3 + i;
  for (int i = 0; i < 256; i++) b[i] = loc[255 - i] + loc[(i * 7) & 255];
  for (int i = 0; i < 16; i++) {
#pragma HLS unroll
    b[i] = loc[i] * a[i] - b[i];
  }
}
