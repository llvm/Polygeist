#ifndef R
#define R 32
#endif
#ifndef K
#define K 64
#endif
void aten_max_values_cpu(float x[R][K], float out[R]) {
  for(int r=0;r<R;++r){float v=x[r][0];for(int k=1;k<K;++k)v=x[r][k]>v?x[r][k]:v;out[r]=v;}
}
