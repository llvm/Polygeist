#define B 1
#define C 2
#define ID 8
#define IH 9
#define IW 10
#define OD 3
#define OH 4
#define OW 5
#define KD 2
#define KH 3
#define KW 3
void aten_fractional_max_pool3d_backward_cpu(float grad[B][C][OD][OH][OW],int index[B][C][OD][OH][OW],float out[B][C][ID][IH][IW]){for(int p=0;p<B*C*ID*IH*IW;++p)((float*)out)[p]=0;for(int b=0;b<B;++b)for(int c=0;c<C;++c)for(int z=0;z<OD;++z)for(int y=0;y<OH;++y)for(int x=0;x<OW;++x){int q=index[b][c][z][y][x];out[b][c][q/(IH*IW)][(q/IW)%IH][q%IW]+=grad[b][c][z][y][x];}}
