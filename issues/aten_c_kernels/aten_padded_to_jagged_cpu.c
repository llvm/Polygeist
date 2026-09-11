#define B 8
#define N 64
void aten_padded_to_jagged_cpu(float x[B][N],int off[B+1],float out[B*N]){for(int b=0;b<B;++b)for(int i=0;off[b]+i<off[b+1];++i)out[off[b]+i]=x[b][i];}
