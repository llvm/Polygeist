#define R 32
#define C 64
#define S 8
#define L 16
void aten_narrow_copy_dense_cpu(float x[R][C],float out[R][L]){for(int i=0;i<R;++i)for(int j=0;j<L;++j)out[i][j]=x[i][S+j];}
