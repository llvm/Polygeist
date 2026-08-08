#define R 32
#define C 64
extern float powf(float,float);void aten_linalg_powsum_cpu(float x[R][C],float p,float out[R]){for(int r=0;r<R;++r){float v=0;for(int c=0;c<C;++c)v+=powf(x[r][c]<0?-x[r][c]:x[r][c],p);out[r]=v;}}
