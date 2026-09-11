#define R 64
#define C 64
#define BR 4
#define BC 4
void aten_compressed_block_convert_cpu(float x[R][C],float out[R/BR][C/BC][BR][BC]){for(int r=0;r<R;++r)for(int c=0;c<C;++c)out[r/BR][c/BC][r%BR][c%BC]=x[r][c];}
