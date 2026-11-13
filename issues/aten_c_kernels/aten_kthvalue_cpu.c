#define R 16
#define N 63
void aten_kthvalue_cpu(float x[R][N],int k,float out[R]){for(int r=0;r<R;++r){for(int i=0;i<=k;++i){int b=i;for(int j=i+1;j<N;++j)if(x[r][j]<x[r][b])b=j;float t=x[r][i];x[r][i]=x[r][b];x[r][b]=t;}out[r]=x[r][k];}}
