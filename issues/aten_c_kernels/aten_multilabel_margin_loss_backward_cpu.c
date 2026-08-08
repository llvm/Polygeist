#define B 16
#define C 16
#define L 4
void aten_multilabel_margin_loss_backward_cpu(float input[B][C],int target[B][L],float grad[B],float out[B][C]){for(int b=0;b<B;++b){for(int c=0;c<C;++c)out[b][c]=0;for(int l=0;l<L;++l){int t=target[b][l];for(int c=0;c<C;++c){int used=0;for(int q=0;q<L;++q)used|=target[b][q]==c;if(!used&&1.0f-input[b][t]+input[b][c]>0){float g=grad[b]/C;out[b][c]+=g;out[b][t]-=g;}}}}}
