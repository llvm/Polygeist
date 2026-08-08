#define N 256
void aten_randperm_cpu(unsigned bits[N],int out[N]){for(int i=0;i<N;++i)out[i]=i;for(int i=N-1;i>0;--i){int j=bits[i]%(i+1);int t=out[i];out[i]=out[j];out[j]=t;}}
