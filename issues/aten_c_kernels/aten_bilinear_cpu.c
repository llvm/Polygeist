#define B 8
#define I 16
#define J 20
#define O 24
void aten_bilinear_cpu(float a[B][I],float w[O][I][J],float b[B][J],float out[B][O]){for(int n=0;n<B;++n)for(int o=0;o<O;++o){float v=0;for(int i=0;i<I;++i)for(int j=0;j<J;++j)v+=a[n][i]*w[o][i][j]*b[n][j];out[n][o]=v;}}
