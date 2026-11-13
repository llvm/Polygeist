#ifndef R
#define R 32
#endif
#ifndef K
#define K 64
#endif
extern float expf(float); extern float log1pf(float);
void aten_logcumsumexp_cpu(float x[R][K], float out[R][K]) {
  for(int r=0;r<R;++r){float v=x[r][0];out[r][0]=v;for(int k=1;k<K;++k){float m=v>x[r][k]?v:x[r][k];float d=v-x[r][k];if(d<0)d=-d;v=m+log1pf(expf(-d));out[r][k]=v;}}
}
