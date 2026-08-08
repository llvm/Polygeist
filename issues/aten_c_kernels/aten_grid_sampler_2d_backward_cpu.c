#define B 1
#define C 3
#define IH 8
#define IW 8
#define OH 6
#define OW 6
void aten_grid_sampler_2d_backward_cpu(float x[B][C][IH][IW],float grid[B][OH][OW][2],float grad[B][C][OH][OW],float dx[B][C][IH][IW],float dgrid[B][OH][OW][2]){for(int p=0;p<B*C*IH*IW;++p)((float*)dx)[p]=0;for(int n=0;n<B;++n)for(int y=0;y<OH;++y)for(int z=0;z<OW;++z){float fx=(grid[n][y][z][0]+1)*0.5f*(IW-1),fy=(grid[n][y][z][1]+1)*0.5f*(IH-1);int x0=(int)fx,y0=(int)fy,x1=x0+1,y1=y0+1;float wx=fx-x0,wy=fy-y0,gx=0,gy=0;for(int c=0;c<C;++c){float g=grad[n][c][y][z],v00=0,v01=0,v10=0,v11=0;if(x0>=0&&x0<IW&&y0>=0&&y0<IH){v00=x[n][c][y0][x0];dx[n][c][y0][x0]+=g*(1-wx)*(1-wy);}if(x1>=0&&x1<IW&&y0>=0&&y0<IH){v01=x[n][c][y0][x1];dx[n][c][y0][x1]+=g*wx*(1-wy);}if(x0>=0&&x0<IW&&y1>=0&&y1<IH){v10=x[n][c][y1][x0];dx[n][c][y1][x0]+=g*(1-wx)*wy;}if(x1>=0&&x1<IW&&y1>=0&&y1<IH){v11=x[n][c][y1][x1];dx[n][c][y1][x1]+=g*wx*wy;}gx+=g*((v01-v00)*(1-wy)+(v11-v10)*wy);gy+=g*((v10-v00)*(1-wx)+(v11-v01)*wx);}dgrid[n][y][z][0]=gx*0.5f*(IW-1);dgrid[n][y][z][1]=gy*0.5f*(IH-1);}}
