#ifndef R
#define R 32
#endif
#ifndef K
#define K 64
#endif
void aten_xor_sum_cpu(int x[R][K], int out[R]) {
  for(int r=0;r<R;++r){int v=0;for(int k=0;k<K;++k)v^=x[r][k];out[r]=v;}
}
