#define C 2
#define O 3
#define H 16
#define W 16
#define K 3
#define D 2
void aten_dilated_convolution_cpu(float x[C][H][W],float w[O][C][K][K],float out[O][H-2*D][W-2*D]){for(int o=0;o<O;++o)for(int y=0;y<H-2*D;++y)for(int q=0;q<W-2*D;++q){float v=0;for(int c=0;c<C;++c)for(int ky=0;ky<K;++ky)for(int kx=0;kx<K;++kx)v+=x[c][y+ky*D][q+kx*D]*w[o][c][ky][kx];out[o][y][q]=v;}}
