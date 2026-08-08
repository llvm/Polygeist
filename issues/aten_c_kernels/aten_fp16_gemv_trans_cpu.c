#define M 64
#define K 128
void aten_fp16_gemv_trans_cpu(float matrix[M][K],float vector[M],float out[K]){for(int k=0;k<K;++k){float s=0;for(int m=0;m<M;++m)s+=matrix[m][k]*vector[m];out[k]=s;}}
