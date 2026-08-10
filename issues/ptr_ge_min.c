void ptr_ge_min(float *buffer, int n) {
  float *buf2 = buffer;
  float *d = &buf2[n - 2];
  while (d >= buf2) {
    d[0] = 0.0f;
    d -= 2;
  }
}
