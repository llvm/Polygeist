#define C 2
#define H 8
#define W 8
#define KH 3
#define KW 3
#define OH 6
#define OW 6
void aten_unfolded2d_acc_cpu(float col[C][KH][KW][OH][OW],float out[C][H][W]){for(int c=0;c<C;++c)for(int y=0;y<H;++y)for(int x=0;x<W;++x)out[c][y][x]=0;for(int c=0;c<C;++c)for(int ky=0;ky<KH;++ky)for(int kx=0;kx<KW;++kx)for(int oy=0;oy<OH;++oy)for(int ox=0;ox<OW;++ox)out[c][oy+ky][ox+kx]+=col[c][ky][kx][oy][ox];}
