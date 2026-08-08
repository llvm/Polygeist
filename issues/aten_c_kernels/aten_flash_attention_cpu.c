#define B 1
#define H 2
#define Q 16
#define K 16
#define D 32
extern float expf(float);extern float logf(float);extern float sqrtf(float);
void aten_flash_attention_cpu(float q[B][H][Q][D],float k[B][H][K][D],float v[B][H][K][D],float out[B][H][Q][D],float lse[B][H][Q]){float scale=1.0f/sqrtf((float)D);for(int b=0;b<B;++b)for(int h=0;h<H;++h)for(int i=0;i<Q;++i){float score[K],m=-3.402823466e38f;for(int j=0;j<K;++j){float s=0;for(int d=0;d<D;++d)s+=q[b][h][i][d]*k[b][h][j][d];score[j]=s*scale;m=score[j]>m?score[j]:m;}float z=0;for(int j=0;j<K;++j){score[j]=expf(score[j]-m);z+=score[j];}lse[b][h][i]=m+logf(z);for(int d=0;d<D;++d){float s=0;for(int j=0;j<K;++j)s+=score[j]/z*v[b][h][j][d];out[b][h][i][d]=s;}}}
