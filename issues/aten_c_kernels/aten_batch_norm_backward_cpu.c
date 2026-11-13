#define B 4
#define C 8
#define S 32
void aten_batch_norm_backward_cpu(float grad[B][C][S],float x[B][C][S],float mean[C],float invstd[C],float weight[C],float dx[B][C][S],float dweight[C],float dbias[C]){for(int c=0;c<C;++c){float sg=0,sgx=0;for(int b=0;b<B;++b)for(int i=0;i<S;++i){sg+=grad[b][c][i];sgx+=grad[b][c][i]*(x[b][c][i]-mean[c]);}dbias[c]=sg;dweight[c]=sgx*invstd[c];for(int b=0;b<B;++b)for(int i=0;i<S;++i)dx[b][c][i]=weight[c]*invstd[c]/(B*S)*((B*S)*grad[b][c][i]-sg-(x[b][c][i]-mean[c])*invstd[c]*invstd[c]*sgx);}}
