#ifndef N
#define N 4096
#endif
void aten_random_from_to_cpu(unsigned bits[N],int from,int to,int out[N]){unsigned range=(unsigned)(to-from);for(int i=0;i<N;++i)out[i]=from+(int)(bits[i]%range);}
