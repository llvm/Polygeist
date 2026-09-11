#define N 1024
void aten_equal_cpu(float a[N],float b[N],int out[1]){int v=1;for(int i=0;i<N;++i)v=v&&(a[i]==b[i]);out[0]=v;}
