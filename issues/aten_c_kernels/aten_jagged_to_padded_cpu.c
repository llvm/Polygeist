#define B 8
#define N 64
void aten_jagged_to_padded_cpu(float x[B*N],int off[B+1],float out[B][N]){for(int b=0;b<B;++b)for(int i=0;i<N;++i)out[b][i]=off[b]+i<off[b+1]?x[off[b]+i]:0;}
