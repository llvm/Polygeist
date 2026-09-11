#define N 32
#define SIZE 64
#define STEP 2
void aten_unfold_backward_cpu(float grad[N][SIZE],float out[N*STEP+SIZE]){for(int i=0;i<N*STEP+SIZE;++i)out[i]=0;for(int w=0;w<N;++w)for(int k=0;k<SIZE;++k)out[w*STEP+k]+=grad[w][k];}
