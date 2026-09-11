void aten_amp_update_scale_cpu(float scale[1], int tracker[1],
    float found_inf[1], float growth, float backoff, int interval) {
  if (found_inf[0] != 0.0f) { scale[0] *= backoff; tracker[0] = 0; }
  else {
    int successful = tracker[0] + 1;
    if (successful == interval) { scale[0] *= growth; tracker[0] = 0; }
    else tracker[0] = successful;
  }
}
