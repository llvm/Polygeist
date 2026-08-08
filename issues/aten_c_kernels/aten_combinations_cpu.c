#define N 32
#define K (N*(N-1)/2)
void aten_combinations_cpu(float x[N],float out[K][2]){for(int i=0;i<N;++i)for(int j=i+1;j<N;++j){int p=i*(2*N-i-1)/2+(j-i-1);out[p][0]=x[i];out[p][1]=x[j];}}
