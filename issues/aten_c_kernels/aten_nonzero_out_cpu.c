#define R 32
#define C 64
void aten_nonzero_out_cpu(float x[R][C],int row[R*C],int col[R*C],int count[1]){int p=0;for(int r=0;r<R;++r)for(int c=0;c<C;++c)if(x[r][c]!=0){row[p]=r;col[p++]=c;}count[0]=p;}
