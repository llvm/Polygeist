#define N 256
#define D 8
void aten_sobol_draw_cpu(unsigned state[D],unsigned dirs[D][32],float out[N][D]){for(int n=0;n<N;++n){int bit=0,q=n;while(q&1){++bit;q>>=1;}for(int d=0;d<D;++d){state[d]^=dirs[d][bit];out[n][d]=state[d]*(1.0f/4294967296.0f);}}}
