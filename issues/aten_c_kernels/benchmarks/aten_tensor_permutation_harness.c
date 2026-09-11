#include <stdio.h>
#include <stdlib.h>

static float value(int i) { return (float)((i * 17) % 251 - 113) * 0.03125f; }

#if defined(TEST_CHANNEL5)
void aten_channel_shuffle(float *, float *);
int main(void){int n=B*G*CPG*H*W;float*x=malloc(n*4),*y=malloc(n*4);for(int i=0;i<n;i++)x[i]=value(i);aten_channel_shuffle(x,y);for(int b=0;b<B;b++)for(int g=0;g<G;g++)for(int c=0;c<CPG;c++)for(int h=0;h<H;h++)for(int w=0;w<W;w++){int si=((((b*G+g)*CPG+c)*H+h)*W+w),di=((((b*CPG+c)*G+g)*H+h)*W+w);if(y[di]!=x[si])return 1;}puts("PASS permutation channel5");return 0;}
#elif defined(TEST_CHANNEL4)
void aten_channel_shuffle_cpu(float *, float *);
int main(void){int n=B*G*CPG*S;float*x=malloc(n*4),*y=malloc(n*4);for(int i=0;i<n;i++)x[i]=value(i);aten_channel_shuffle_cpu(x,y);for(int b=0;b<B;b++)for(int g=0;g<G;g++)for(int c=0;c<CPG;c++)for(int s=0;s<S;s++){int si=((b*(G*CPG)+g*CPG+c)*S+s),di=((b*(G*CPG)+c*G+g)*S+s);if(y[di]!=x[si])return 1;}puts("PASS permutation channel4");return 0;}
#elif defined(TEST_PIXEL6)
void aten_pixel_shuffle(float *, float *);
int main(void){int n=B*C*R*R*H*W;float*x=malloc(n*4),*y=malloc(n*4);for(int i=0;i<n;i++)x[i]=value(i);aten_pixel_shuffle(x,y);for(int b=0;b<B;b++)for(int c=0;c<C;c++)for(int ry=0;ry<R;ry++)for(int rx=0;rx<R;rx++)for(int h=0;h<H;h++)for(int w=0;w<W;w++){int si=(((((b*C+c)*R+ry)*R+rx)*H+h)*W+w),di=(((b*C+c)*(H*R)+h*R+ry)*(W*R)+w*R+rx);if(y[di]!=x[si])return 1;}puts("PASS permutation pixel6");return 0;}
#elif defined(TEST_PIXEL_BACKEND) || defined(TEST_UNPIXEL_BACKEND)
#ifdef TEST_PIXEL_BACKEND
void aten_pixel_shuffle_cpu_backend(float *, float *);
#else
void aten_pixel_unshuffle_cpu_backend(float *, float *);
#endif
int main(void){int n=B*C*RATIO*RATIO*H*W;float*x=malloc(n*4),*y=malloc(n*4);for(int i=0;i<n;i++)x[i]=value(i);
#ifdef TEST_PIXEL_BACKEND
aten_pixel_shuffle_cpu_backend(x,y);
#else
aten_pixel_unshuffle_cpu_backend(x,y);
#endif
for(int b=0;b<B;b++)for(int c=0;c<C;c++)for(int h=0;h<H;h++)for(int w=0;w<W;w++)for(int ry=0;ry<RATIO;ry++)for(int rx=0;rx<RATIO;rx++){int packed=(((b*(C*RATIO*RATIO)+c*RATIO*RATIO+ry*RATIO+rx)*H+h)*W+w),expanded=(((b*C+c)*(H*RATIO)+h*RATIO+ry)*(W*RATIO)+w*RATIO+rx);
#ifdef TEST_PIXEL_BACKEND
if(y[expanded]!=x[packed])return 1;
#else
if(y[packed]!=x[expanded])return 1;
#endif
}puts("PASS permutation pixel-backend");return 0;}
#elif defined(TEST_TRANSPOSE)
void aten_transpose_copy(float *, float *);
int main(void){float*x=malloc((size_t)M*N*4),*y=malloc((size_t)M*N*4);for(int i=0;i<M*N;i++)x[i]=value(i);aten_transpose_copy(x,y);for(int i=0;i<M;i++)for(int j=0;j<N;j++)if(y[j*M+i]!=x[i*N+j])return 1;puts("PASS permutation transpose");return 0;}
#elif defined(TEST_STACK)
void aten_stack_serial_cpu(float *, float *);
int main(void){int n=T*R*K;float*x=malloc(n*4),*y=malloc(n*4);for(int i=0;i<n;i++)x[i]=value(i);aten_stack_serial_cpu(x,y);for(int t=0;t<T;t++)for(int r=0;r<R;r++)for(int k=0;k<K;k++)if(y[(r*T+t)*K+k]!=x[(t*R+r)*K+k])return 1;puts("PASS permutation stack");return 0;}
#else
#error select TEST mode
#endif
