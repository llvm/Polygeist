#define B 8
#define I 16
#define J 20
#define K 24
void aten_trilinear_cpu(float a[B][I],float w[I][J][K],float b[B][J],float out[B][K]){for(int n=0;n<B;++n)for(int k=0;k<K;++k){float v=0;for(int i=0;i<I;++i)for(int j=0;j<J;++j)v+=a[n][i]*w[i][j][k]*b[n][j];out[n][k]=v;}}
