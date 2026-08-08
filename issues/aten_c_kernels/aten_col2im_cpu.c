#define C 2
#define H 8
#define W 8
#define KH 3
#define KW 3
void aten_col2im_cpu(float col[C][KH][KW][H][W],float out[C][H+2][W+2]){for(int p=0;p<C*(H+2)*(W+2);++p)((float*)out)[p]=0;for(int c=0;c<C;++c)for(int ky=0;ky<KH;++ky)for(int kx=0;kx<KW;++kx)for(int y=0;y<H;++y)for(int x=0;x<W;++x)out[c][y+ky][x+kx]+=col[c][ky][kx][y][x];}
