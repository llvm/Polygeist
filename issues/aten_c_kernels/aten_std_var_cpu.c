#ifndef R
#define R 32
#endif
#ifndef K
#define K 64
#endif
extern float sqrtf(float);
void aten_std_var_cpu(float x[R][K], int correction, float out[R]) {
  for (int r=0;r<R;++r) { float sum=0.0f; for(int k=0;k<K;++k) sum+=x[r][k];
    float mean=sum/(float)K, sq=0.0f;
    for(int k=0;k<K;++k){float d=x[r][k]-mean;sq+=d*d;}
    out[r]=sqrtf(sq/(float)(K-correction)); }
}
