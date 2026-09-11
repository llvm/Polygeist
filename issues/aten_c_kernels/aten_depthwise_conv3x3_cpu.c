#define B 1
#define C 8
#define H 16
#define W 16
void aten_depthwise_conv3x3_cpu(float x[B][C][H][W],float weight[C][3][3],float bias[C],float out[B][C][H][W]){for(int n=0;n<B;++n)for(int c=0;c<C;++c)for(int y=0;y<H;++y)for(int z=0;z<W;++z){float s=bias[c];for(int ky=0;ky<3;++ky)for(int kx=0;kx<3;++kx){int iy=y+ky-1,ix=z+kx-1;if(iy>=0&&iy<H&&ix>=0&&ix<W)s+=x[n][c][iy][ix]*weight[c][ky][kx];}out[n][c][y][z]=s;}}
