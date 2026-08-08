#define C 4
#define I 32
#define O 7
void aten_adaptive_max_pool1d_cpu(float x[C][I],float out[C][O],int index[C][O]){for(int c=0;c<C;++c)for(int o=0;o<O;++o){int s=o*I/O,e=((o+1)*I+O-1)/O,b=s;float v=x[c][s];for(int i=s+1;i<e;++i)if(x[c][i]>v){v=x[c][i];b=i;}out[c][o]=v;index[c][o]=b;}}
