#define R 32
#define C 64
extern float sqrtf(float);void aten_vector_norm_out_cpu(float x[R][C],float out[R]){for(int r=0;r<R;++r){float v=0;for(int c=0;c<C;++c)v+=x[r][c]*x[r][c];out[r]=sqrtf(v);}}
