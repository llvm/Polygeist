#define ROWS 16
#define INNER 32
#define COLS 24
#define NNZ 96
void aten_spmm_reduce_backward_input_arg_cpu(int crow[ROWS+1],int col[NNZ],int arg[ROWS][COLS],float grad[ROWS][COLS],float other[INNER][COLS],float out[NNZ]){for(int p=0;p<NNZ;++p)out[p]=0;for(int r=0;r<ROWS;++r)for(int n=0;n<COLS;++n){int p=arg[r][n];if(p>=0)out[p]+=grad[r][n]*other[col[p]][n];}}
