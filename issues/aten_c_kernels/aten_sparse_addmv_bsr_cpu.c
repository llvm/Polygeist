#define R 16
#define C 16
#define BR 4
#define BC 4
#define N 64
void aten_sparse_addmv_bsr_cpu(int ptr[R+1],int col[N],float val[N][BR][BC],float x[C*BC],float out[R*BR]){for(int r=0;r<R;++r)for(int i=0;i<BR;++i){float v=0;for(int p=ptr[r];p<ptr[r+1];++p)for(int j=0;j<BC;++j)v+=val[p][i][j]*x[col[p]*BC+j];out[r*BR+i]=v;}}
