#define R 32
#define C 64
void aten_allany_dims_cpu(int x[R][C],int all,int out[R]){for(int r=0;r<R;++r){int v=all;for(int c=0;c<C;++c)v=all?(v&&x[r][c]):(v||x[r][c]);out[r]=v;}}
