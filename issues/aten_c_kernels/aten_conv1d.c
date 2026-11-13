/* aten::conv1d, stride 1, no padding. Upstream: ATen/native/Convolution.cpp. */
#define B 2
#define IC 3
#define OC 4
#define W 16
#define K 3
void aten_conv1d(float input[B][IC][W], float weight[OC][IC][K],
                 float bias[OC], float output[B][OC][W - K + 1]) {
#pragma scop
  for (int b = 0; b < B; ++b)
    for (int oc = 0; oc < OC; ++oc)
      for (int ow = 0; ow < W - K + 1; ++ow) {
        output[b][oc][ow] = bias[oc];
        for (int ic = 0; ic < IC; ++ic)
          for (int k = 0; k < K; ++k)
            output[b][oc][ow] += input[b][ic][ow + k] * weight[oc][ic][k];
      }
#pragma endscop
}
