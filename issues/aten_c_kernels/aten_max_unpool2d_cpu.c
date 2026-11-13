#define C 2
#define N 64
#define O 256
void aten_max_unpool2d_cpu(float x[C][N],int index[C][N],float out[C][O]){for(int p=0;p<C*O;++p)((float*)out)[p]=0;for(int c=0;c<C;++c)for(int i=0;i<N;++i)out[c][index[c][i]]=x[c][i];}
