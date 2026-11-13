#define C 2
#define ID 8
#define IH 9
#define IW 10
#define OD 3
#define OH 4
#define OW 5
void aten_adaptive_max_pool3d_legacy_backward_cpu(float g[C][OD][OH][OW],int idx[C][OD][OH][OW],float out[C][ID][IH][IW]){for(int p=0;p<C*ID*IH*IW;++p)((float*)out)[p]=0;for(int c=0;c<C;++c)for(int z=0;z<OD;++z)for(int y=0;y<OH;++y)for(int x=0;x<OW;++x){int q=idx[c][z][y][x];out[c][q/(IH*IW)][(q/IW)%IH][q%IW]+=g[c][z][y][x];}}
