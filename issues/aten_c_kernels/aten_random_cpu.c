#ifndef N
#define N 4096
#endif
void aten_random_cpu(unsigned bits[N],unsigned upper,int out[N]){for(int i=0;i<N;++i)out[i]=(int)(bits[i]%upper);}
