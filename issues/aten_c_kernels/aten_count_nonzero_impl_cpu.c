#define R 32
#define C 64
void aten_count_nonzero_impl_cpu(float x[R][C],int out[R]){for(int r=0;r<R;++r){int n=0;for(int c=0;c<C;++c)n+=x[r][c]!=0;out[r]=n;}}
