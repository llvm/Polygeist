#define B 1
#define H 2
#define Q 16
#define K 16
#define D 32
extern float expf(float);extern float sqrtf(float);
void aten_flash_attention_backward_cpu(float q[B][H][Q][D],float k[B][H][K][D],float v[B][H][K][D],float grad[B][H][Q][D],float dq[B][H][Q][D],float dk[B][H][K][D],float dv[B][H][K][D]){for(int p=0;p<B*H*Q*D;++p)((float*)dq)[p]=0;for(int p=0;p<B*H*K*D;++p){((float*)dk)[p]=0;((float*)dv)[p]=0;}float scale=1.0f/sqrtf((float)D);for(int b=0;b<B;++b)for(int h=0;h<H;++h)for(int i=0;i<Q;++i){float p[K],dp[K],m=-3.402823466e38f,z=0,sump=0;for(int j=0;j<K;++j){float s=0;for(int d=0;d<D;++d)s+=q[b][h][i][d]*k[b][h][j][d];p[j]=s*scale;m=p[j]>m?p[j]:m;}for(int j=0;j<K;++j){p[j]=expf(p[j]-m);z+=p[j];}for(int j=0;j<K;++j){p[j]/=z;float s=0;for(int d=0;d<D;++d){s+=grad[b][h][i][d]*v[b][h][j][d];dv[b][h][j][d]+=p[j]*grad[b][h][i][d];}dp[j]=s;sump+=s*p[j];}for(int j=0;j<K;++j){float ds=p[j]*(dp[j]-sump)*scale;for(int d=0;d<D;++d){dq[b][h][i][d]+=ds*k[b][h][j][d];dk[b][h][j][d]+=ds*q[b][h][i][d];}}}}
