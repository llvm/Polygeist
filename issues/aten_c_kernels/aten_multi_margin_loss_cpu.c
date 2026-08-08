#define B 32
#define C 16
void aten_multi_margin_loss_cpu(float input[B][C],int target[B],float weight[C],float margin,int p,float out[B]){for(int b=0;b<B;++b){int t=target[b];float s=0;for(int c=0;c<C;++c)if(c!=t){float z=margin-input[b][t]+input[b][c];if(z>0)s+=p==1?z:z*z;}out[b]=s*weight[t]/C;}}
