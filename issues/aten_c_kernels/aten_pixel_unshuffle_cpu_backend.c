#ifndef B
#define B 1
#endif
#ifndef C
#define C 3
#endif
#ifndef H
#define H 8
#endif
#ifndef W
#define W 8
#endif
#ifndef RATIO
#define RATIO 2
#endif
void aten_pixel_unshuffle_cpu_backend(float x[B][C][H*RATIO][W*RATIO],float out[B][C*RATIO*RATIO][H][W]){for(int n=0;n<B;++n)for(int c=0;c<C;++c)for(int h=0;h<H;++h)for(int w=0;w<W;++w)for(int ry=0;ry<RATIO;++ry)for(int rx=0;rx<RATIO;++rx)out[n][c*RATIO*RATIO+ry*RATIO+rx][h][w]=x[n][c][h*RATIO+ry][w*RATIO+rx];}
