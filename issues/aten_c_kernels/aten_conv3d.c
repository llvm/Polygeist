/* aten::conv3d, 3^3 stride 1. Upstream: ATen/native/Convolution.cpp. */
#define B 1
#define IC 2
#define OC 3
#define D 6
#define H 6
#define W 6
#define K 3
void aten_conv3d(float input[B][IC][D][H][W],
                 float weight[OC][IC][K][K][K], float bias[OC],
                 float output[B][OC][D-K+1][H-K+1][W-K+1]) {
#pragma scop
  for (int b = 0; b < B; ++b)
    for (int oc = 0; oc < OC; ++oc)
      for (int od = 0; od < D-K+1; ++od)
        for (int oh = 0; oh < H-K+1; ++oh)
          for (int ow = 0; ow < W-K+1; ++ow) {
            output[b][oc][od][oh][ow] = bias[oc];
            for (int ic = 0; ic < IC; ++ic)
              for (int kd = 0; kd < K; ++kd)
                for (int kh = 0; kh < K; ++kh)
                  for (int kw = 0; kw < K; ++kw)
                    output[b][oc][od][oh][ow] +=
                        input[b][ic][od+kd][oh+kh][ow+kw] *
                        weight[oc][ic][kd][kh][kw];
          }
#pragma endscop
}
