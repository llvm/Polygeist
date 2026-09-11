#define ROWS 16
#define INNER 32
#define COLS 24
#define NNZ 96
void aten_spmm_reduce_backward_other_arg_cpu(int col[NNZ],float val[NNZ],int arg[ROWS][COLS],float grad[ROWS][COLS],float out[INNER][COLS]){for(int k=0;k<INNER;++k)for(int n=0;n<COLS;++n)out[k][n]=0;for(int r=0;r<ROWS;++r)for(int n=0;n<COLS;++n){int p=arg[r][n];if(p>=0)out[col[p]][n]+=val[p]*grad[r][n];}}
