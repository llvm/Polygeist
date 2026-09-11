#define ROWS 16
#define INNER 32
#define COLS 24
#define NNZ 96
void aten_spmm_reduce_cpu(int crow[ROWS+1],int col[NNZ],float val[NNZ],float other[INNER][COLS],int reduce,float out[ROWS][COLS]){for(int r=0;r<ROWS;++r)for(int n=0;n<COLS;++n){float acc=reduce==2?-3.402823466e38f:(reduce==3?3.402823466e38f:0.0f);for(int p=crow[r];p<crow[r+1];++p){float x=val[p]*other[col[p]][n];if(reduce==0||reduce==1)acc+=x;else if(reduce==2)acc=acc>x?acc:x;else acc=acc<x?acc:x;}if(reduce==1&&crow[r+1]>crow[r])acc/=(float)(crow[r+1]-crow[r]);out[r][n]=acc;}}
