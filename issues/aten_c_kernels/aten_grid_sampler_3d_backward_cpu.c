#ifndef B
#define B 1
#define C 2
#define ID 6
#define IH 7
#define IW 8
#define OD 4
#define OH 5
#define OW 6
#endif
void aten_grid_sampler_3d_backward_cpu(float grad[B][C][OD][OH][OW],float grid[B][OD][OH][OW][3],float out[B][C][ID][IH][IW]){for(int p=0;p<B*C*ID*IH*IW;++p)((float*)out)[p]=0;for(int b=0;b<B;++b)for(int z=0;z<OD;++z)for(int y=0;y<OH;++y)for(int w=0;w<OW;++w){float fx=(grid[b][z][y][w][0]+1)*.5f*(IW-1),fy=(grid[b][z][y][w][1]+1)*.5f*(IH-1),fz=(grid[b][z][y][w][2]+1)*.5f*(ID-1);int x0=(int)fx,y0=(int)fy,z0=(int)fz;float ax=fx-x0,ay=fy-y0,az=fz-z0;for(int c=0;c<C;++c)for(int dz=0;dz<2;++dz)for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx){int iz=z0+dz,iy=y0+dy,ix=x0+dx;if(iz>=0&&iz<ID&&iy>=0&&iy<IH&&ix>=0&&ix<IW)out[b][c][iz][iy][ix]+=grad[b][c][z][y][w]*(dz?az:1-az)*(dy?ay:1-ay)*(dx?ax:1-ax);}}}
