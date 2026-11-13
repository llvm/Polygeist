#ifndef R
#define R 16
#endif
#ifndef K
#define K 64
#endif
#ifndef TOP
#define TOP 8
#endif
void aten_nansum_cpu(float x[R][K],float out[R]){for(int r=0;r<R;++r){float s=0;for(int k=0;k<K;++k)if(x[r][k]==x[r][k])s+=x[r][k];out[r]=s;}}
