#define B 8
#define N 64
void aten_nested_where_cpu(int cond[B][N],float a[B][N],float b[B][N],float out[B][N]){for(int q=0;q<B;++q)for(int i=0;i<N;++i)out[q][i]=cond[q][i]?a[q][i]:b[q][i];}
