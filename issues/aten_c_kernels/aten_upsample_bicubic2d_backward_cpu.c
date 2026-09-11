#define OH 8
#define OW 8
#define IH 5
#define IW 5
void aten_upsample_bicubic2d_backward_cpu(float grad[OH][OW],float out[IH][IW]){for(int p=0;p<IH*IW;++p)((float*)out)[p]=0;for(int y=0;y<OH;++y)for(int x=0;x<OW;++x){int iy=y*IH/OH,ix=x*IW/OW;out[iy][ix]+=grad[y][x];}}
