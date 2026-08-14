#ifndef T
#define T 4
#endif
#ifndef R
#define R 16
#endif
#ifndef K
#define K 32
#endif
void aten_stack_serial_cpu(float x[T][R][K],float out[R][T][K]){for(int t=0;t<T;++t)for(int r=0;r<R;++r)for(int k=0;k<K;++k)out[r][t][k]=x[t][r][k];}
