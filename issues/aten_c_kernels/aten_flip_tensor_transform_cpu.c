#define R 32
#define C 64
void aten_flip_tensor_transform_cpu(float x[R][C],float out[R][C]){for(int i=0;i<R;++i)for(int j=0;j<C;++j)out[i][j]=x[R-1-i][C-1-j];}
