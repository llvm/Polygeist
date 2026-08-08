#define N 1024
void aten_sparse_sum_backward_cpu(float grad,float out[N]){for(int i=0;i<N;++i)out[i]=grad;}
