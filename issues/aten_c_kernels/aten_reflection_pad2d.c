/* aten::reflection_pad2d, pad=1. Upstream: ATen/native/ReflectionPad.cpp. */
#define C 3
#define H 8
#define W 8
void aten_reflection_pad2d(float input[C][H][W],
                           float output[C][H + 2][W + 2]) {
#pragma scop
  for (int c = 0; c < C; ++c)
    for (int oh = 0; oh < H + 2; ++oh)
      for (int ow = 0; ow < W + 2; ++ow) {
        int ih = oh == 0 ? 1 : (oh == H + 1 ? H - 2 : oh - 1);
        int iw = ow == 0 ? 1 : (ow == W + 1 ? W - 2 : ow - 1);
        output[c][oh][ow] = input[c][ih][iw];
      }
#pragma endscop
}
