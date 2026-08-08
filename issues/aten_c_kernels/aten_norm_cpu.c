#ifndef R
#define R 32
#endif
#ifndef K
#define K 64
#endif
extern float sqrtf(float);
void aten_norm_cpu(float x[R][K], float out[R]) {
  for(int r=0;r<R;++r){float s=0.0f;for(int k=0;k<K;++k)s+=x[r][k]*x[r][k];out[r]=sqrtf(s);}
}
