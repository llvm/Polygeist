#define ROWS 16
#define INNER 32
#define COLS 24
#define NNZ 96
void aten_spmm_reduce_arg_cpu(int crow[ROWS+1],int col[NNZ],float val[NNZ],float other[INNER][COLS],int choose_max,float out[ROWS][COLS],int arg[ROWS][COLS]){for(int r=0;r<ROWS;++r)for(int n=0;n<COLS;++n){float acc=choose_max?-3.402823466e38f:3.402823466e38f;int best=-1;for(int p=crow[r];p<crow[r+1];++p){float x=val[p]*other[col[p]][n];if((choose_max&&x>acc)||(!choose_max&&x<acc)){acc=x;best=p;}}out[r][n]=acc;arg[r][n]=best;}}
