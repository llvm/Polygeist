#define N 64
void aten_eye_cpu(float out[N][N]){for(int i=0;i<N;++i)for(int j=0;j<N;++j)out[i][j]=i==j;}
