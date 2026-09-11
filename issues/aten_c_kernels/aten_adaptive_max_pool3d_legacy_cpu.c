#define C 2
#define ID 8
#define IH 9
#define IW 10
#define OD 3
#define OH 4
#define OW 5
void aten_adaptive_max_pool3d_legacy_cpu(float x[C][ID][IH][IW],float out[C][OD][OH][OW],int idx[C][OD][OH][OW]){for(int c=0;c<C;++c)for(int z=0;z<OD;++z)for(int y=0;y<OH;++y)for(int q=0;q<OW;++q){int zs=z*ID/OD,ze=((z+1)*ID+OD-1)/OD,ys=y*IH/OH,ye=((y+1)*IH+OH-1)/OH,xs=q*IW/OW,xe=((q+1)*IW+OW-1)/OW,b=(zs*IH+ys)*IW+xs;float v=x[c][zs][ys][xs];for(int iz=zs;iz<ze;++iz)for(int iy=ys;iy<ye;++iy)for(int ix=xs;ix<xe;++ix)if(x[c][iz][iy][ix]>v){v=x[c][iz][iy][ix];b=(iz*IH+iy)*IW+ix;}out[c][z][y][q]=v;idx[c][z][y][q]=b;}}
