#define N 256
void aten_fft_conjugate_symmetry_cpu(float re[N],float im[N]){for(int i=N/2+1;i<N;++i){re[i]=re[N-i];im[i]=-im[N-i];}}
