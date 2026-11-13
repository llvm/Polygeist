#define B 2
#define C 3
#define IH 9
#define IW 10
#define OH 4
#define OW 5
#define KH 3
#define KW 3
void aten_fractional_max_pool2d_backward_cpu(float grad[B][C][OH][OW],int index[B][C][OH][OW],float out[B][C][IH][IW]){for(int p=0;p<B*C*IH*IW;++p)((float*)out)[p]=0;for(int b=0;b<B;++b)for(int c=0;c<C;++c)for(int y=0;y<OH;++y)for(int x=0;x<OW;++x){int q=index[b][c][y][x];out[b][c][q/IW][q%IW]+=grad[b][c][y][x];}}
