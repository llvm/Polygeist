#define B 32
#define C 16
void aten_nll_loss_backward_cpu(float grad[B],int target[B],float weight[C],float out[B][C]){for(int b=0;b<B;++b)for(int c=0;c<C;++c)out[b][c]=0;for(int b=0;b<B;++b)out[b][target[b]]=-grad[b]*weight[target[b]];}
