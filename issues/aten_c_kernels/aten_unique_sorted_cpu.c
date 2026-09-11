#define N 1024
void aten_unique_sorted_cpu(int x[N],int out[N],int count[1]){for(int i=0;i<N;++i)for(int j=i+1;j<N;++j)if(x[j]<x[i]){int t=x[i];x[i]=x[j];x[j]=t;}int p=0;for(int i=0;i<N;++i)if(i==0||x[i]!=x[i-1])out[p++]=x[i];count[0]=p;}
