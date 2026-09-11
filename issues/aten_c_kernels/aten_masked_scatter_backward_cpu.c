#define N 512
void aten_masked_scatter_backward_cpu(float grad[N],int mask[N],float out[N]){int p=0;for(int i=0;i<N;++i)if(mask[i])out[p++]=grad[i];}
