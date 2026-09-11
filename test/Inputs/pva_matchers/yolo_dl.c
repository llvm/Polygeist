// Independently authored semantic fixtures for YOLO and common PVA-DL ops.
#include <math.h>
#include <stdint.h>

static float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }
static int clamp_index(int x, int lo, int hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

// Fused bilinear resize, letterbox fill, NHWC->NCHW, and affine normalization.
void fixture_nhwc_to_nchw_letterbox_normalize(
    int ih, int iw, int channels, int oh, int ow, int top, int left,
    int resized_h, int resized_w, const uint8_t *restrict input,
    float *restrict output, const float *restrict mean,
    const float *restrict inv_std, uint8_t fill) {
  for (int c = 0; c < channels; ++c)
    for (int y = 0; y < oh; ++y)
      for (int x = 0; x < ow; ++x) {
        float value = (float)fill;
        if (y >= top && y < top + resized_h && x >= left && x < left + resized_w) {
          float fy = ((float)(y-top)+0.5f)*(float)ih/(float)resized_h-0.5f;
          float fx = ((float)(x-left)+0.5f)*(float)iw/(float)resized_w-0.5f;
          int y0=clamp_index((int)floorf(fy),0,ih-1),x0=clamp_index((int)floorf(fx),0,iw-1);
          int y1=clamp_index(y0+1,0,ih-1),x1=clamp_index(x0+1,0,iw-1);
          float wy=fy-floorf(fy),wx=fx-floorf(fx);
          value=(1-wy)*((1-wx)*input[(y0*iw+x0)*channels+c]+wx*input[(y0*iw+x1)*channels+c])+
                wy*((1-wx)*input[(y1*iw+x0)*channels+c]+wx*input[(y1*iw+x1)*channels+c]);
        }
        output[(c*oh+y)*ow+x]=(value*(1.0f/255.0f)-mean[c])*inv_std[c];
      }
}

// Decode one YOLOv5 prediction tensor into xyxy boxes and object*class scores.
void fixture_yolov5_decode(int anchors, int classes, int gh, int gw,
                           const float *restrict logits,
                           const float *restrict anchor_wh, int stride,
                           float *restrict boxes, float *restrict scores) {
  int attrs=classes+5;
  for(int a=0;a<anchors;++a)for(int y=0;y<gh;++y)for(int x=0;x<gw;++x){
    int p=(a*gh*gw+y*gw+x)*attrs, o=(a*gh*gw+y*gw+x);
    float cx=(2*sigmoidf(logits[p])-0.5f+x)*stride;
    float cy=(2*sigmoidf(logits[p+1])-0.5f+y)*stride;
    float bw=2*sigmoidf(logits[p+2]);bw=bw*bw*anchor_wh[2*a];
    float bh=2*sigmoidf(logits[p+3]);bh=bh*bh*anchor_wh[2*a+1];
    boxes[4*o]=cx-0.5f*bw;boxes[4*o+1]=cy-0.5f*bh;
    boxes[4*o+2]=cx+0.5f*bw;boxes[4*o+3]=cy+0.5f*bh;
    float objectness=sigmoidf(logits[p+4]);
    for(int c=0;c<classes;++c)scores[o*classes+c]=objectness*sigmoidf(logits[p+5+c]);
  }
}

// Greedy class-agnostic NMS. Input order must already be descending by score.
int fixture_batched_nms(int n,float iou_threshold,const float *restrict boxes,
                        const float *restrict scores,int *restrict keep){
  uint8_t suppressed[n];for(int i=0;i<n;++i)suppressed[i]=0;int kept=0;
  for(int i=0;i<n;++i){if(suppressed[i])continue;keep[kept++]=i;
    for(int j=i+1;j<n;++j){if(suppressed[j]||scores[j]>scores[i])continue;
      float xl=boxes[4*i]>boxes[4*j]?boxes[4*i]:boxes[4*j];
      float yt=boxes[4*i+1]>boxes[4*j+1]?boxes[4*i+1]:boxes[4*j+1];
      float xr=boxes[4*i+2]<boxes[4*j+2]?boxes[4*i+2]:boxes[4*j+2];
      float yb=boxes[4*i+3]<boxes[4*j+3]?boxes[4*i+3]:boxes[4*j+3];
      float iw=xr>xl?xr-xl:0,ih=yb>yt?yb-yt:0,inter=iw*ih;
      float ai=(boxes[4*i+2]-boxes[4*i])*(boxes[4*i+3]-boxes[4*i+1]);
      float aj=(boxes[4*j+2]-boxes[4*j])*(boxes[4*j+3]-boxes[4*j+1]);
      if(inter/(ai+aj-inter)>iou_threshold)suppressed[j]=1;
    }
  }return kept;
}

void fixture_dl_convolution_nchw(int n,int ci,int hi,int wi,int co,int kh,int kw,
                                 const float *restrict x,const float *restrict k,
                                 const float *restrict bias,float *restrict y){
  int ho=hi-kh+1,wo=wi-kw+1;
  for(int b=0;b<n;++b)for(int o=0;o<co;++o)for(int yy=0;yy<ho;++yy)for(int xx=0;xx<wo;++xx){float s=bias[o];
    for(int c=0;c<ci;++c)for(int r=0;r<kh;++r)for(int q=0;q<kw;++q)
      s+=x[((b*ci+c)*hi+yy+r)*wi+xx+q]*k[((o*ci+c)*kh+r)*kw+q];
    y[((b*co+o)*ho+yy)*wo+xx]=s;
  }
}

void fixture_dl_depthwise_convolution(int channels,int h,int w,int kh,int kw,
                                      const float *restrict x,const float *restrict k,
                                      float *restrict y){int ho=h-kh+1,wo=w-kw+1;
  for(int c=0;c<channels;++c)for(int yy=0;yy<ho;++yy)for(int xx=0;xx<wo;++xx){float s=0;
    for(int r=0;r<kh;++r)for(int q=0;q<kw;++q)s+=x[(c*h+yy+r)*w+xx+q]*k[(c*kh+r)*kw+q];
    y[(c*ho+yy)*wo+xx]=s;}
}

void fixture_dl_gemm(int m,int n,int k,const float *restrict a,const float *restrict b,
                     const float *restrict bias,float *restrict c){
  for(int i=0;i<m;++i)for(int j=0;j<n;++j){float s=bias?bias[j]:0;
    for(int q=0;q<k;++q)s+=a[i*k+q]*b[q*n+j];
    c[i*n+j]=s;}
}

void fixture_dl_max_pool2x2(int c,int h,int w,const float *restrict x,float *restrict y){
  for(int ch=0;ch<c;++ch)for(int oy=0;oy<h/2;++oy)for(int ox=0;ox<w/2;++ox){float v=x[(ch*h+2*oy)*w+2*ox];
    for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx){float q=x[(ch*h+2*oy+dy)*w+2*ox+dx];v=q>v?q:v;}
    y[(ch*(h/2)+oy)*(w/2)+ox]=v;}
}

void fixture_dl_activation_relu(int n,const float *restrict x,float *restrict y){
  for(int i=0;i<n;++i)y[i]=x[i]>0?x[i]:0;
}

void fixture_dl_gather_rows(int rows,int cols,const float *restrict x,int count,
                            const int32_t *restrict indices,float *restrict y){
  (void)rows;
  for(int i=0;i<count;++i)for(int j=0;j<cols;++j)y[i*cols+j]=x[indices[i]*cols+j];
}

void fixture_dl_quantize_dequantize(int n,const float *restrict x,int8_t *restrict q,
                                    float *restrict y,float scale,int zero){
  for(int i=0;i<n;++i){int v=(int)nearbyintf(x[i]/scale)+zero;v=clamp_index(v,-128,127);
    q[i]=(int8_t)v;y[i]=((float)v-(float)zero)*scale;}
}

void fixture_dl_softmax(int rows,int cols,const float *restrict x,float *restrict y){
  for(int r=0;r<rows;++r){float mx=x[r*cols];for(int c=1;c<cols;++c)mx=x[r*cols+c]>mx?x[r*cols+c]:mx;
    float sum=0;for(int c=0;c<cols;++c){float e=expf(x[r*cols+c]-mx);y[r*cols+c]=e;sum+=e;}
    for(int c=0;c<cols;++c)y[r*cols+c]/=sum;}
}

void fixture_dl_topk(int n,int k,const float *restrict x,float *restrict values,int *restrict indices){
  for(int j=0;j<k;++j){float best=-INFINITY;int bi=-1;for(int i=0;i<n;++i){int used=0;
      for(int p=0;p<j;++p)used|=indices[p]==i;
      if(!used&&x[i]>best){best=x[i];bi=i;}}
    values[j]=best;indices[j]=bi;}
}

void fixture_dl_layer_norm(int rows,int cols,const float *restrict x,
                           const float *restrict gamma,const float *restrict beta,
                           float epsilon,float *restrict y){
  for(int r=0;r<rows;++r){float sum=0,sq=0;for(int c=0;c<cols;++c){float v=x[r*cols+c];sum+=v;sq+=v*v;}
    float mean=sum/cols,inv=1.0f/sqrtf(sq/cols-mean*mean+epsilon);
    for(int c=0;c<cols;++c)y[r*cols+c]=(x[r*cols+c]-mean)*inv*gamma[c]+beta[c];}
}
