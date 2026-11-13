#define N 64
void aten_eig_complex_vectors_cpu(float real[N][N],float imag[N],float out_re[N][N],float out_im[N][N]){for(int i=0;i<N;++i)for(int j=0;j<N;++j){out_re[i][j]=real[i][j];out_im[i][j]=imag[j]==0?0:(j+1<N?real[i][j+1]:0);}}
