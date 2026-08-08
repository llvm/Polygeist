#define R 128
#define C 16
void aten_unique_dim_impl_cpu(float x[R][C],int keep[R]){for(int r=0;r<R;++r){int unique=1;for(int q=0;q<r;++q){int same=1;for(int c=0;c<C;++c)same&=x[r][c]==x[q][c];unique&=!same;}keep[r]=unique;}}
