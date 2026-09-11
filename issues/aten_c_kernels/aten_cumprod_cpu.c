#ifndef R
#define R 32
#endif
#ifndef K
#define K 64
#endif
void aten_cumprod_cpu(float x[R][K], float out[R][K]) {
  for(int r=0;r<R;++r){float v=1.0f;for(int k=0;k<K;++k){v*=x[r][k];out[r][k]=v;}}
}
