#define N 1024
void aten_unique_consecutive_cpu(int x[N],int out[N],int count[1]){int p=0;for(int i=0;i<N;++i)if(i==0||x[i]!=x[i-1])out[p++]=x[i];count[0]=p;}
