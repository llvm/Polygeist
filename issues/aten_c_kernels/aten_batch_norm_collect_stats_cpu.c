#define B 4
#define C 8
#define S 32
extern float sqrtf(float);void aten_batch_norm_collect_stats_cpu(float x[B][C][S],float mean[C],float var[C]){for(int c=0;c<C;++c){float s=0;for(int b=0;b<B;++b)for(int i=0;i<S;++i)s+=x[b][c][i];mean[c]=s/(B*S);float q=0;for(int b=0;b<B;++b)for(int i=0;i<S;++i){float d=x[b][c][i]-mean[c];q+=d*d;}var[c]=q/(B*S);}}
