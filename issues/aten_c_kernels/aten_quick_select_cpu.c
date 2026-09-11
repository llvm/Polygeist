#define N 127
void aten_quick_select_cpu(float x[N],int k,float out[1]){for(int i=0;i<=k;++i){int b=i;for(int j=i+1;j<N;++j)if(x[j]<x[b])b=j;float t=x[i];x[i]=x[b];x[b]=t;}out[0]=x[k];}
