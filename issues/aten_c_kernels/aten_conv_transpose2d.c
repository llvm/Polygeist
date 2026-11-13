/* aten::conv_transpose2d, stride 1. Upstream: ATen/native/NaiveConvolutionTranspose2d.cpp. */
#define B 1
#define IC 2
#define OC 3
#define H 6
#define W 6
#define K 3
void aten_conv_transpose2d(float input[B][IC][H][W],
                           float weight[IC][OC][K][K],
                           float output[B][OC][H+K-1][W+K-1]) {
#pragma scop
  for (int b = 0; b < B; ++b)
    for (int oc = 0; oc < OC; ++oc)
      for (int oh = 0; oh < H+K-1; ++oh)
        for (int ow = 0; ow < W+K-1; ++ow)
          output[b][oc][oh][ow] = 0.0f;
  for (int b = 0; b < B; ++b)
    for (int ic = 0; ic < IC; ++ic)
      for (int ih = 0; ih < H; ++ih)
        for (int iw = 0; iw < W; ++iw)
          for (int oc = 0; oc < OC; ++oc)
            for (int kh = 0; kh < K; ++kh)
              for (int kw = 0; kw < K; ++kw)
                output[b][oc][ih+kh][iw+kw] +=
                    input[b][ic][ih][iw] * weight[ic][oc][kh][kw];
#pragma endscop
}
