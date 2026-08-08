#define ROWS 16
#define INNER 32
#define COLS 24
#define NNZ 96
void aten_spmm_reduce_backward_input_cpu(int crow[ROWS+1],int col[NNZ],float grad[ROWS][COLS],float other[INNER][COLS],float out[NNZ]){for(int r=0;r<ROWS;++r)for(int p=crow[r];p<crow[r+1];++p){float s=0;for(int n=0;n<COLS;++n)s+=grad[r][n]*other[col[p]][n];out[p]=s;}}
