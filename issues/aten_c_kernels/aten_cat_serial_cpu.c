#ifndef R
#define R 16
#endif
#ifndef K
#define K 64
#endif
#ifndef TOP
#define TOP 8
#endif
#define M 12
void aten_cat_serial_cpu(float a[R][K],float b[M][K],float out[R+M][K]){for(int r=0;r<R;++r)for(int k=0;k<K;++k)out[r][k]=a[r][k];for(int r=0;r<M;++r)for(int k=0;k<K;++k)out[R+r][k]=b[r][k];}
