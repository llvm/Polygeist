#define B 32
#define C 16
void aten_multi_margin_loss_backward_cpu(float input[B][C],int target[B],float weight[C],float margin,int p,float grad[B],float out[B][C]){for(int b=0;b<B;++b){int t=target[b];float sum=0;for(int c=0;c<C;++c){out[b][c]=0;if(c!=t){float z=margin-input[b][t]+input[b][c];if(z>0){float g=grad[b]*weight[t]*(p==1?1.0f:2.0f*z)/C;out[b][c]=g;sum+=g;}}}out[b][t]=-sum;}}
