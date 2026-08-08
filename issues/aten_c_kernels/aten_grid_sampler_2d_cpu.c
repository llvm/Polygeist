#define B 1
#define C 3
#define IH 8
#define IW 8
#define OH 6
#define OW 6
void aten_grid_sampler_2d_cpu(float x[B][C][IH][IW],float grid[B][OH][OW][2],float out[B][C][OH][OW]){for(int n=0;n<B;++n)for(int y=0;y<OH;++y)for(int z=0;z<OW;++z){float fx=(grid[n][y][z][0]+1)*0.5f*(IW-1),fy=(grid[n][y][z][1]+1)*0.5f*(IH-1);int x0=(int)fx,y0=(int)fy,x1=x0+1,y1=y0+1;float wx=fx-x0,wy=fy-y0;for(int c=0;c<C;++c){float v=0;if(x0>=0&&x0<IW&&y0>=0&&y0<IH)v+=(1-wx)*(1-wy)*x[n][c][y0][x0];if(x1>=0&&x1<IW&&y0>=0&&y0<IH)v+=wx*(1-wy)*x[n][c][y0][x1];if(x0>=0&&x0<IW&&y1>=0&&y1<IH)v+=(1-wx)*wy*x[n][c][y1][x0];if(x1>=0&&x1<IW&&y1>=0&&y1<IH)v+=wx*wy*x[n][c][y1][x1];out[n][c][y][z]=v;}}}
