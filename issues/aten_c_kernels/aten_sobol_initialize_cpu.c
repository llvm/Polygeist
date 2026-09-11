#define D 8
void aten_sobol_initialize_cpu(unsigned dirs[D][32],unsigned state[D]){for(int d=0;d<D;++d)state[d]=dirs[d][0];}
