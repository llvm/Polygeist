#define R 64
#define C 32
void aten_rowwise_prune_cpu(float x[R][C],float threshold,int keep[R]){for(int r=0;r<R;++r){float v=0;for(int c=0;c<C;++c){float a=x[r][c];v+=a<0?-a:a;}keep[r]=v>threshold;}}
