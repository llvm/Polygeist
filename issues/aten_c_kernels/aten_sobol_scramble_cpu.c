#define D 8
void aten_sobol_scramble_cpu(unsigned dirs[D][32],unsigned shift[D]){for(int d=0;d<D;++d)for(int b=0;b<32;++b)dirs[d][b]^=shift[d]>>(b&7);}
