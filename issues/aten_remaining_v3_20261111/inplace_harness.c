#define CU(call) do { cudaError_t e=(call); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));exit(2);} }while(0)
static double now_ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec*1e-6;}
static int equal(float a,float b,int exact){
  if(isnan(a)||isnan(b))return isnan(a)&&isnan(b);
  if(!isfinite(a)||!isfinite(b))return a==b;
  if(exact)return memcmp(&a,&b,sizeof(float))==0;
  if(a==0&&b==0)return signbit(a)==signbit(b);
  return fabs((double)a-b)<=1e-5+1e-4*fabs(b);
}
int main(int argc,char **argv){
  const int edge=argc>1&&!strcmp(argv[1],"edge");
  const int partial=argc>1&&!strcmp(argv[1],"partial");
  const int timing=argc>1&&!strcmp(argv[1],"timing");
  const size_t bytes=(N+8)*sizeof(float);
  float *initial[2],*h[2],*d[2],*r[2],*v[2];
  float edges[]={0.0f,-0.0f,1.0f,-1.0f,INFINITY,-INFINITY,NAN,0x1p-149f,-0x1p-149f,1e20f,-1e20f};
  const size_t ne=sizeof(edges)/sizeof(edges[0]);
  for(int a=0;a<2;a++){
    initial[a]=malloc(bytes);h[a]=malloc(bytes);r[a]=malloc(bytes);v[a]=malloc(bytes);
    CU(cudaMalloc((void**)&d[a],bytes));
    for(size_t i=0;i<N+8;i++)initial[a][i]=913.25f;
    for(size_t i=0;i<N;i++){
      float x=(float)(((i+a*613)*104729LL)%8192)/1024.0f-4.0f;
      if(edge)x=edges[(a?i/ne:i)%ne];
      initial[a][i+4]=x;
    }
    memcpy(h[a],initial[a],bytes);memcpy(r[a],initial[a],bytes);
    CU(cudaMemcpy(d[a],initial[a],bytes,cudaMemcpyHostToDevice));
    initial[a]+=4;h[a]+=4;r[a]+=4;v[a]+=4;d[a]+=4;
  }
  if(partial){float *p[]={d[1]+1,d[1]};invoke(p);CU(cudaDeviceSynchronize());puts("FAIL partial overlap accepted");return 1;}
  if(timing){
    for(int k=0;k<2;k++)for(int w=0;w<5;w++){if(k)direct(d);else invoke(d);CU(cudaDeviceSynchronize());}
    for(int i=0;i<5;i++)for(int z=0;z<2;z++){
      int k=z^(i&1);CU(cudaDeviceSynchronize());double t=now_ms();
      if(k)direct(d);else invoke(d);CU(cudaDeviceSynchronize());
      printf("SAMPLE path=%s sample=%d ms=%.9f\n",k?"direct_cudnn":"raised",i,now_ms()-t);
    }
    puts("PASS timing");return 0;
  }
  int anybad=0;
  for(int alias=0;alias<2;alias++){
    for(int a=0;a<2;a++){
      memcpy(r[a]-4,initial[a]-4,bytes);
      CU(cudaMemcpy(d[a]-4,initial[a]-4,bytes,cudaMemcpyHostToDevice));
    }
    float *rp[]={alias?r[1]:r[0],r[1]},*dp[]={alias?d[1]:d[0],d[1]};
    reference(rp);direct(dp);CU(cudaDeviceSynchronize());
    for(int a=0;a<2;a++)CU(cudaMemcpy(v[a]-4,d[a]-4,bytes,cudaMemcpyDeviceToHost));
    for(int resident=0;resident<2;resident++){
      for(int a=0;a<2;a++){
        memcpy(h[a]-4,initial[a]-4,bytes);
        CU(cudaMemcpy(d[a]-4,initial[a]-4,bytes,cudaMemcpyHostToDevice));
      }
      float **base=resident?d:h;
      float *args[]={alias?base[1]:base[0],base[1]};
      invoke(args);CU(cudaDeviceSynchronize());
      if(resident)for(int a=0;a<2;a++)CU(cudaMemcpy(h[a]-4,d[a]-4,bytes,cudaMemcpyDeviceToHost));
      size_t bad=0,vendorbad=0;double maxabs=0,maxrel=0;
      for(int a=0;a<2;a++)for(size_t i=0;i<N+8;i++){
        const int output=a==1&&i>=4&&i<N+4;
        float got=h[a][(long)i-4],ref=r[a][(long)i-4],vendor=v[a][(long)i-4];
        if(!equal(got,ref,!output)){if(bad<4)printf("MISMATCH array=%d i=%zu got=%a reference=%a\n",a,i,got,ref);bad++;}
        if(!equal(got,vendor,1))vendorbad++;
        if(output&&isfinite(got)&&isfinite(ref)){
          double diff=fabs((double)got-ref);if(diff>maxabs)maxabs=diff;
          double rel=diff/fmax(fabs(ref),1e-30);if(rel>maxrel)maxrel=rel;
        }
      }
      printf("CHECK edge=%d alias=%d resident=%d count=%d bad=%zu vendorbad=%zu maxabs=%.9g maxrel=%.9g\n",edge,alias,resident,N,bad,vendorbad,maxabs,maxrel);
      anybad|=(bad||vendorbad);
    }
  }
  if(!anybad)puts("PASS correctness");
  return anybad;
}
