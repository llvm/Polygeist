/* aten::conv2d, NCHW, stride 1, valid padding. */
#define B 2
#define IC 4
#define OC 8
#define H 16
#define W 16
#define KH 3
#define KW 3
#define OH (H - KH + 1)
#define OW (W - KW + 1)

void aten_conv2d(float input[B][IC][H][W],
                 float filter[OC][IC][KH][KW],
                 float output[B][OC][OH][OW]) {
#pragma scop
  for (int b = 0; b < B; ++b)
    for (int oc = 0; oc < OC; ++oc)
      for (int oh = 0; oh < OH; ++oh)
        for (int ow = 0; ow < OW; ++ow)
          output[b][oc][oh][ow] = 0.0f;
  for (int b = 0; b < B; ++b)
    for (int oc = 0; oc < OC; ++oc)
      for (int oh = 0; oh < OH; ++oh)
        for (int ow = 0; ow < OW; ++ow)
          for (int ic = 0; ic < IC; ++ic)
            for (int kh = 0; kh < KH; ++kh)
              for (int kw = 0; kw < KW; ++kw)
                output[b][oc][oh][ow] +=
                    input[b][ic][oh + kh][ow + kw] *
                    filter[oc][ic][kh][kw];
#pragma endscop
}
