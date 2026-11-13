#define B 32
#define C 16
void aten_nll_loss_forward_cpu(float input[B][C],int target[B],float weight[C],float out[B],float total_weight[1]){float tw=0;for(int b=0;b<B;++b){int t=target[b];out[b]=-input[b][t]*weight[t];tw+=weight[t];}total_weight[0]=tw;}
