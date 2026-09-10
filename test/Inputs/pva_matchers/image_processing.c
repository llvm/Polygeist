// Independently authored semantic fixtures for PVA image-operation matching.
// These functions contain no PVA calls and recognition must not use names.
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

static int clampi(int x, int lo, int hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

void fixture_morphology_dilate3x3(int h, int w, const uint8_t *restrict in,
                                  uint8_t *restrict out) {
  for (int y = 1; y < h - 1; ++y)
    for (int x = 1; x < w - 1; ++x) {
      uint8_t v = 0;
      for (int ky = -1; ky <= 1; ++ky)
        for (int kx = -1; kx <= 1; ++kx) {
          uint8_t q = in[(y + ky) * w + x + kx];
          v = q > v ? q : v;
        }
      out[y * w + x] = v;
    }
}

void fixture_image_histogram_u8(int h, int w, const uint8_t *restrict in,
                                uint32_t (*restrict histogram)[256]) {
  for (int b = 0; b < 256; ++b) (*histogram)[b] = 0;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      (*histogram)[(uint32_t)in[y * w + x]]++;
}

// One weighted-centroid refinement step per supplied corner.
void fixture_corner_subpix(int h, int w, const uint8_t *restrict image,
                           int count, const float *restrict xy,
                           float *restrict refined_xy) {
  for (int p = 0; p < count; ++p) {
    int cx = (int)(xy[2 * p] + 0.5f), cy = (int)(xy[2 * p + 1] + 0.5f);
    float sx = 0.0f, sy = 0.0f, sw = 0.0f;
    for (int dy = -2; dy <= 2; ++dy)
      for (int dx = -2; dx <= 2; ++dx) {
        int x = clampi(cx + dx, 0, w - 1), y = clampi(cy + dy, 0, h - 1);
        float weight = (float)image[y * w + x] + 1.0f;
        sx += weight * (float)x;
        sy += weight * (float)y;
        sw += weight;
      }
    refined_xy[2 * p] = sx / sw;
    refined_xy[2 * p + 1] = sy / sw;
  }
}

void fixture_min_max_loc_f32(int n, const float *restrict in,
                             float *restrict min_value,
                             float *restrict max_value,
                             int *restrict min_index, int *restrict max_index) {
  float lo = in[0], hi = in[0];
  int li = 0, hi_i = 0;
  for (int i = 1; i < n; ++i) {
    if (in[i] < lo) { lo = in[i]; li = i; }
    if (in[i] > hi) { hi = in[i]; hi_i = i; }
  }
  *min_value = lo; *max_value = hi; *min_index = li; *max_index = hi_i;
}

// Sum-of-squared-differences template matching at every valid placement.
void fixture_template_matching_ssd(int h, int w, int th, int tw,
                                   const uint8_t *restrict image,
                                   const uint8_t *restrict templ,
                                   uint32_t *restrict score) {
  for (int y = 0; y <= h - th; ++y)
    for (int x = 0; x <= w - tw; ++x) {
      uint32_t sum = 0;
      for (int j = 0; j < th; ++j)
        for (int i = 0; i < tw; ++i) {
          int d = (int)image[(y + j) * w + x + i] - (int)templ[j * tw + i];
          sum += (uint32_t)(d * d);
        }
      score[y * (w - tw + 1) + x] = sum;
    }
}

void fixture_mix_channels_rgba_to_bgra(int pixels,
                                        const uint8_t *restrict in,
                                        uint8_t *restrict out) {
  for (int i = 0; i < pixels; ++i) {
    out[4*i] = in[4*i+2]; out[4*i+1] = in[4*i+1];
    out[4*i+2] = in[4*i]; out[4*i+3] = in[4*i+3];
  }
}

// 4x4 block-linear storage to ordinary pitch-linear row-major storage.
void fixture_block_linear_to_pitch_linear_4x4(int h, int w,
                                               const uint8_t *restrict in,
                                               uint8_t *restrict out) {
  int blocks_x = (w + 3) / 4;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      int block = (y / 4) * blocks_x + x / 4;
      int within = (y % 4) * 4 + x % 4;
      out[y * w + x] = in[block * 16 + within];
    }
}

void fixture_convert_rgb_to_gray(int pixels, const uint8_t *restrict rgb,
                                 uint8_t *restrict gray) {
  for (int i = 0; i < pixels; ++i)
    gray[i] = (uint8_t)((77u * rgb[3*i] + 150u * rgb[3*i+1] +
                         29u * rgb[3*i+2] + 128u) >> 8);
}

void fixture_image_blend_u8(int n, const uint8_t *restrict a,
                            const uint8_t *restrict b, uint8_t *restrict out,
                            uint16_t alpha_q8) {
  for (int i = 0; i < n; ++i)
    out[i] = (uint8_t)((alpha_q8 * a[i] + (256u - alpha_q8) * b[i] + 128u) >> 8);
}

void fixture_image_resize_bilinear(int ih, int iw, int oh, int ow,
                                   const uint8_t *restrict in,
                                   uint8_t *restrict out) {
  for (int y = 0; y < oh; ++y)
    for (int x = 0; x < ow; ++x) {
      float fy = ((float)y + 0.5f) * (float)ih / (float)oh - 0.5f;
      float fx = ((float)x + 0.5f) * (float)iw / (float)ow - 0.5f;
      int y0 = clampi((int)floorf(fy), 0, ih-1), x0 = clampi((int)floorf(fx), 0, iw-1);
      int y1 = clampi(y0+1, 0, ih-1), x1 = clampi(x0+1, 0, iw-1);
      float wy = fy-floorf(fy), wx = fx-floorf(fx);
      float v = (1-wy)*((1-wx)*in[y0*iw+x0]+wx*in[y0*iw+x1]) +
                wy*((1-wx)*in[y1*iw+x0]+wx*in[y1*iw+x1]);
      out[y*ow+x] = (uint8_t)clampi((int)(v+0.5f), 0, 255);
    }
}

void fixture_blur_filter_roi3x3(int h, int w, int y0, int x0, int rh, int rw,
                                const uint8_t *restrict in,
                                uint8_t *restrict out) {
  for (int y = y0+1; y < y0+rh-1 && y < h-1; ++y)
    for (int x = x0+1; x < x0+rw-1 && x < w-1; ++x) {
      int sum = 0;
      for (int ky=-1; ky<=1; ++ky) for (int kx=-1; kx<=1; ++kx)
        sum += in[(y+ky)*w+x+kx];
      out[y*w+x] = (uint8_t)((sum+4)/9);
    }
}

// Two-pass Manhattan distance transform; zero pixels are foreground seeds.
void fixture_distance_transform_l1(int h, int w, const uint8_t *restrict in,
                                   uint16_t *restrict distance) {
  const uint16_t inf = 32767;
  for (int y=0; y<h; ++y) for (int x=0; x<w; ++x) {
    uint16_t v = in[y*w+x] == 0 ? 0 : inf;
    if (y>0 && distance[(y-1)*w+x]+1 < v) v=distance[(y-1)*w+x]+1;
    if (x>0 && distance[y*w+x-1]+1 < v) v=distance[y*w+x-1]+1;
    distance[y*w+x]=v;
  }
  for (int y=h-1; y>=0; --y) for (int x=w-1; x>=0; --x) {
    uint16_t v=distance[y*w+x];
    if (y+1<h && distance[(y+1)*w+x]+1<v) v=distance[(y+1)*w+x]+1;
    if (x+1<w && distance[y*w+x+1]+1<v) v=distance[y*w+x+1]+1;
    distance[y*w+x]=v;
  }
}

void fixture_bilateral_filter3x3(int h, int w, const uint8_t *restrict in,
                                 uint8_t *restrict out, float sigma_r,
                                 float sigma_s) {
  float ir=-0.5f/(sigma_r*sigma_r), is=-0.5f/(sigma_s*sigma_s);
  for(int y=1;y<h-1;++y) for(int x=1;x<w-1;++x){
    float c=in[y*w+x], ws=0, vs=0;
    for(int ky=-1;ky<=1;++ky) for(int kx=-1;kx<=1;++kx){
      float q=in[(y+ky)*w+x+kx], d=q-c;
      float wt=expf((float)(kx*kx+ky*ky)*is+d*d*ir); ws+=wt; vs+=wt*q;
    }
    out[y*w+x]=(uint8_t)clampi((int)(vs/ws+0.5f),0,255);
  }
}

void fixture_brute_force_matcher_l2(int queries, int trains, int dims,
                                    const float *restrict query,
                                    const float *restrict train,
                                    int *restrict best_index,
                                    float *restrict best_distance) {
  for(int q=0;q<queries;++q){ float best=INFINITY; int bi=-1;
    for(int t=0;t<trains;++t){ float sum=0;
      for(int d=0;d<dims;++d){float z=query[q*dims+d]-train[t*dims+d];sum+=z*z;}
      if(sum<best){best=sum;bi=t;}
    } best_index[q]=bi; best_distance[q]=best;
  }
}

void fixture_warp_perspective_nearest(int ih,int iw,int oh,int ow,
                                      const uint8_t *restrict in,
                                      uint8_t *restrict out,
                                      const float *restrict m){
  for(int y=0;y<oh;++y) for(int x=0;x<ow;++x){
    float z=m[6]*x+m[7]*y+m[8];
    int sx=(int)floorf((m[0]*x+m[1]*y+m[2])/z+0.5f);
    int sy=(int)floorf((m[3]*x+m[4]*y+m[5])/z+0.5f);
    out[y*ow+x]=(sx>=0&&sx<iw&&sy>=0&&sy<ih)?in[sy*iw+sx]:0;
  }
}

void fixture_image_stats_f32(int n,const float *restrict in,float *restrict sum,
                             float *restrict sumsq,float *restrict lo,float *restrict hi){
  float s=0,q=0,mn=in[0],mx=in[0];
  for(int i=0;i<n;++i){float v=in[i];s+=v;q+=v*v;mn=v<mn?v:mn;mx=v>mx?v:mx;}
  *sum=s;*sumsq=q;*lo=mn;*hi=mx;
}

// Sobel magnitude plus dual threshold; hysteresis is intentionally excluded.
void fixture_canny_sobel_threshold(int h,int w,const uint8_t *restrict in,
                                   uint8_t *restrict edges,int low,int high){
  for(int y=1;y<h-1;++y) for(int x=1;x<w-1;++x){
    int gx=-in[(y-1)*w+x-1]+in[(y-1)*w+x+1]-2*in[y*w+x-1]+2*in[y*w+x+1]-in[(y+1)*w+x-1]+in[(y+1)*w+x+1];
    int gy=-in[(y-1)*w+x-1]-2*in[(y-1)*w+x]-in[(y-1)*w+x+1]+in[(y+1)*w+x-1]+2*in[(y+1)*w+x]+in[(y+1)*w+x+1];
    int mag=abs(gx)+abs(gy); edges[y*w+x]=(uint8_t)(mag>=high?255:(mag>=low?128:0));
  }
}

void fixture_median_filter3x3(int h,int w,const uint8_t *restrict in,uint8_t *restrict out){
  for(int y=1;y<h-1;++y) for(int x=1;x<w-1;++x){uint8_t a[9];int n=0;
    for(int ky=-1;ky<=1;++ky)for(int kx=-1;kx<=1;++kx)a[n++]=in[(y+ky)*w+x+kx];
    for(int i=1;i<9;++i){uint8_t v=a[i];int j=i;while(j>0&&a[j-1]>v){a[j]=a[j-1];--j;}a[j]=v;}
    out[y*w+x]=a[4];
  }
}

void fixture_histogram_equalization(int h,int w,const uint8_t *restrict in,uint8_t *restrict out){
  uint32_t hist[256],cdf[256];uint8_t lut[256];
  for(int b=0;b<256;++b)hist[b]=0;
  for(int i=0;i<h*w;++i)hist[in[i]]++;
  cdf[0]=hist[0];for(int b=1;b<256;++b)cdf[b]=cdf[b-1]+hist[b];
  uint32_t first=0;for(int b=0;b<256;++b)if(first==0&&cdf[b])first=cdf[b];
  uint32_t den=(uint32_t)(h*w)>first?(uint32_t)(h*w)-first:1;
  for(int b=0;b<256;++b)lut[b]=(uint8_t)clampi((int)((cdf[b]-first)*255/den),0,255);
  for(int i=0;i<h*w;++i)out[i]=lut[in[i]];
}

void fixture_background_subtractor(int n,const uint8_t *restrict frame,
                                   uint8_t *restrict background,uint8_t *restrict mask,
                                   int threshold,uint16_t alpha_q8){
  for(int i=0;i<n;++i){int d=(int)frame[i]-background[i];if(d<0)d=-d;
    mask[i]=(uint8_t)(d>threshold?255:0);
    background[i]=(uint8_t)((alpha_q8*frame[i]+(256-alpha_q8)*background[i]+128)>>8);
  }
}

void fixture_gaussian_pyramid_down2(int h,int w,const uint8_t *restrict in,uint8_t *restrict out){
  const int k[5]={1,4,6,4,1};
  for(int oy=1;oy<h/2-1;++oy)for(int ox=1;ox<w/2-1;++ox){int sum=0;
    for(int ky=-2;ky<=2;++ky)for(int kx=-2;kx<=2;++kx)sum+=k[ky+2]*k[kx+2]*in[(2*oy+ky)*w+2*ox+kx];
    out[oy*(w/2)+ox]=(uint8_t)((sum+128)>>8);
  }
}

void fixture_remap_bilinear(int ih,int iw,int oh,int ow,const uint8_t *restrict in,
                            const float *restrict mapx,const float *restrict mapy,
                            uint8_t *restrict out){
  for(int i=0;i<oh*ow;++i){float fx=mapx[i],fy=mapy[i];int x0=(int)floorf(fx),y0=(int)floorf(fy);
    if(x0<0||x0+1>=iw||y0<0||y0+1>=ih){out[i]=0;continue;}float wx=fx-x0,wy=fy-y0;
    float v=(1-wy)*((1-wx)*in[y0*iw+x0]+wx*in[y0*iw+x0+1])+wy*((1-wx)*in[(y0+1)*iw+x0]+wx*in[(y0+1)*iw+x0+1]);
    out[i]=(uint8_t)clampi((int)(v+0.5f),0,255);
  }
}

void fixture_orb_descriptor(int h,int w,const uint8_t *restrict in,int count,
                            const int *restrict xy,const int8_t *restrict pairs,
                            uint8_t *restrict desc){
  for(int p=0;p<count;++p)for(int byte=0;byte<32;++byte){uint8_t bits=0;
    for(int bit=0;bit<8;++bit){int q=(byte*8+bit)*4,x=xy[2*p],y=xy[2*p+1];
      int ax=clampi(x+pairs[q],0,w-1),ay=clampi(y+pairs[q+1],0,h-1);
      int bx=clampi(x+pairs[q+2],0,w-1),by=clampi(y+pairs[q+3],0,h-1);
      bits|=(uint8_t)((in[ay*w+ax]<in[by*w+bx])<<bit);
    }desc[p*32+byte]=bits;
  }
}

void fixture_image_flip_horizontal(int h,int w,int channels,const uint8_t *restrict in,uint8_t *restrict out){
  for(int y=0;y<h;++y)for(int x=0;x<w;++x)for(int c=0;c<channels;++c)
    out[(y*w+x)*channels+c]=in[(y*w+(w-1-x))*channels+c];
}

// Iterative label propagation for 4-connected components; max_steps is explicit.
void fixture_ccl_label_propagation(int h,int w,const uint8_t *restrict binary,
                                   int32_t *restrict labels,int max_steps){
  for(int i=0;i<h*w;++i)labels[i]=binary[i]?(i+1):0;
  for(int step=0;step<max_steps;++step)for(int y=0;y<h;++y)for(int x=0;x<w;++x){int i=y*w+x;
    if(!binary[i])continue;
    int v=labels[i];
    if(x>0&&labels[i-1]&&labels[i-1]<v)v=labels[i-1];
    if(y>0&&labels[i-w]&&labels[i-w]<v)v=labels[i-w];
    labels[i]=v;
  }
}

void fixture_hog_cells(int h,int w,int cell,const uint8_t *restrict in,float *restrict hist){
  int cells_x=w/cell,cells_y=h/cell;for(int i=0;i<cells_x*cells_y*9;++i)hist[i]=0;
  for(int y=1;y<h-1;++y)for(int x=1;x<w-1;++x){float gx=(float)in[y*w+x+1]-in[y*w+x-1];float gy=(float)in[(y+1)*w+x]-in[(y-1)*w+x];
    float angle=atan2f(gy,gx)+3.14159265f;int bin=clampi((int)(angle*(9.0f/6.2831853f)),0,8);
    hist[((y/cell)*cells_x+x/cell)*9+bin]+=sqrtf(gx*gx+gy*gy);
  }
}

void fixture_fast_corner(int h,int w,const uint8_t *restrict in,uint8_t *restrict corners,int threshold){
  const int dx[16]={0,1,2,3,3,3,2,1,0,-1,-2,-3,-3,-3,-2,-1};
  const int dy[16]={-3,-3,-2,-1,0,1,2,3,3,3,2,1,0,-1,-2,-3};
  for(int y=3;y<h-3;++y)for(int x=3;x<w-3;++x){int c=in[y*w+x],bright=0,dark=0;
    for(int k=0;k<16;++k){int q=in[(y+dy[k])*w+x+dx[k]];bright+=q>c+threshold;dark+=q<c-threshold;}
    corners[y*w+x]=(uint8_t)(bright>=9||dark>=9);
  }
}

void fixture_contour_boundary(int h,int w,const uint8_t *restrict binary,uint8_t *restrict boundary){
  for(int y=1;y<h-1;++y)for(int x=1;x<w-1;++x){int i=y*w+x;
    boundary[i]=(uint8_t)(binary[i]&&(!binary[i-1]||!binary[i+1]||!binary[i-w]||!binary[i+w]));
  }
}

void fixture_conv2d_f32(int h,int w,int kh,int kw,const float *restrict in,
                        const float *restrict kernel,float *restrict out){
  for(int y=0;y<=h-kh;++y)for(int x=0;x<=w-kw;++x){float sum=0;
    for(int ky=0;ky<kh;++ky)for(int kx=0;kx<kw;++kx)sum+=in[(y+ky)*w+x+kx]*kernel[ky*kw+kx];
    out[y*(w-kw+1)+x]=sum;
  }
}

void fixture_gaussian_filter3x3(int h,int w,const uint8_t *restrict in,uint8_t *restrict out){
  const int k[3]={1,2,1};for(int y=1;y<h-1;++y)for(int x=1;x<w-1;++x){int sum=0;
    for(int ky=-1;ky<=1;++ky)for(int kx=-1;kx<=1;++kx)sum+=k[ky+1]*k[kx+1]*in[(y+ky)*w+x+kx];
    out[y*w+x]=(uint8_t)((sum+8)>>4);
  }
}

void fixture_box_filter3x3(int h,int w,const uint8_t *restrict in,uint8_t *restrict out){
  for(int y=1;y<h-1;++y)for(int x=1;x<w-1;++x){int sum=0;
    for(int ky=-1;ky<=1;++ky)for(int kx=-1;kx<=1;++kx)sum+=in[(y+ky)*w+x+kx];
    out[y*w+x]=(uint8_t)((sum+4)/9);
  }
}
