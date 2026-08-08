#define N 2048
void aten_count_nonzero_cpu(float x[N],int out[1]){int n=0;for(int i=0;i<N;++i)n+=x[i]!=0;out[0]=n;}
