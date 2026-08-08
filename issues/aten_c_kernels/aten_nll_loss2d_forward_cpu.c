#define B 4
#define C 8
#define H 16
#define W 16
void aten_nll_loss2d_forward_cpu(float input[B][C][H][W],int target[B][H][W],float weight[C],float out[B][H][W]){for(int b=0;b<B;++b)for(int y=0;y<H;++y)for(int x=0;x<W;++x){int t=target[b][y][x];out[b][y][x]=-input[b][t][y][x]*weight[t];}}
