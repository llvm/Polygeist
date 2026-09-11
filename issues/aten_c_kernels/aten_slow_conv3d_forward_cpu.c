#define C 2
#define O 3
#define D 8
#define H 9
#define W 10
#define K 3
void aten_slow_conv3d_forward_cpu(float x[C][D][H][W],float w[O][C][K][K][K],float out[O][D-2][H-2][W-2]){for(int o=0;o<O;++o)for(int z=0;z<D-2;++z)for(int y=0;y<H-2;++y)for(int q=0;q<W-2;++q){float v=0;for(int c=0;c<C;++c)for(int kz=0;kz<K;++kz)for(int ky=0;ky<K;++ky)for(int kx=0;kx<K;++kx)v+=x[c][z+kz][y+ky][q+kx]*w[o][c][kz][ky][kx];out[o][z][y][q]=v;}}
