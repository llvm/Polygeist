void ptr_subassign_min(float *buffer, int n) {
  float *buf2 = buffer;
  float *d = &buf2[n - 2];
  d -= 2;
  d[0] = 0.0f;
}
