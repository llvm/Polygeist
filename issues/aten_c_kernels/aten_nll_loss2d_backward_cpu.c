#define B 4
#define C 8
#define H 16
#define W 16
void aten_nll_loss2d_backward_cpu(float grad[B][H][W],int target[B][H][W],float weight[C],float out[B][C][H][W]){for(int p=0;p<B*C*H*W;++p)((float*)out)[p]=0;for(int b=0;b<B;++b)for(int y=0;y<H;++y)for(int x=0;x<W;++x){int t=target[b][y][x];out[b][t][y][x]=-grad[b][y][x]*weight[t];}}
