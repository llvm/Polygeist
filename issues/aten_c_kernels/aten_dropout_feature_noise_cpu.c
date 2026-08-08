#define B 8
#define C 16
#define H 8
#define W 8
void aten_dropout_feature_noise_cpu(float x[B][C][H][W],float mask[B][C],float scale,float out[B][C][H][W]){for(int b=0;b<B;++b)for(int c=0;c<C;++c)for(int y=0;y<H;++y)for(int z=0;z<W;++z)out[b][c][y][z]=x[b][c][y][z]*mask[b][c]*scale;}
