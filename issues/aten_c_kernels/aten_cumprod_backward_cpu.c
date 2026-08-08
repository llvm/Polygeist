#define N 128
void aten_cumprod_backward_cpu(float x[N],float prod[N],float grad[N],float out[N]){for(int i=0;i<N;++i){float v=0;for(int j=i;j<N;++j){float p=1;for(int k=0;k<=j;++k)if(k!=i)p*=x[k];v+=grad[j]*p;}out[i]=v;}}
