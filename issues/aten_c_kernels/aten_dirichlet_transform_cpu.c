#define R 64
#define C 16
void aten_dirichlet_transform_cpu(float gamma[R][C],float out[R][C]){for(int r=0;r<R;++r){float s=0;for(int c=0;c<C;++c)s+=gamma[r][c];for(int c=0;c<C;++c)out[r][c]=gamma[r][c]/s;}}
