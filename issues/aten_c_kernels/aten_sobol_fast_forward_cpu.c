#define N 256
#define D 8
void aten_sobol_fast_forward_cpu(unsigned state[D],unsigned dirs[D][32]){for(int n=0;n<N;++n){int bit=0,q=n;while(q&1){++bit;q>>=1;}for(int d=0;d<D;++d)state[d]^=dirs[d][bit];}}
