#define ROWS 16
#define INNER 32
#define COLS 24
#define NNZ 96
void aten_spmm_reduce_backward_other_cpu(int crow[ROWS+1],int col[NNZ],float val[NNZ],float grad[ROWS][COLS],float out[INNER][COLS]){for(int k=0;k<INNER;++k)for(int n=0;n<COLS;++n)out[k][n]=0;for(int r=0;r<ROWS;++r)for(int p=crow[r];p<crow[r+1];++p)for(int n=0;n<COLS;++n)out[col[p]][n]+=val[p]*grad[r][n];}
