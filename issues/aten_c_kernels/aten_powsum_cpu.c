#ifndef R
#define R 32
#endif
#ifndef K
#define K 64
#endif
extern float powf(float,float);
void aten_powsum_cpu(float x[R][K], float p, float out[R]) {
  for(int r=0;r<R;++r){float s=0.0f;for(int k=0;k<K;++k){float a=x[r][k]<0?-x[r][k]:x[r][k];s+=powf(a,p);}out[r]=s;}
}
