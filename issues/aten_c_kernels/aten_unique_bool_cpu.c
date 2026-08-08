#define N 1024
void aten_unique_bool_cpu(int x[N],int values[2],int count[2]){count[0]=count[1]=0;for(int i=0;i<N;++i)count[x[i]!=0]++;values[0]=0;values[1]=1;}
