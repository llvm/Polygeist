#define CU(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));exit(2);} }while(0)
static double now_ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec*1e-6;}
static int equal(float a,float b,int exact){
 if(isnan(a)||isnan(b))return isnan(a)&&isnan(b);
 if(!isfinite(a)||!isfinite(b))return a==b;
 if(exact)return memcmp(&a,&b,sizeof(float))==0;
 if(a==0&&b==0)return signbit(a)==signbit(b);
 return fabs((double)a-b)<=1e-5+1e-4*fabs(b);
}
int main(int argc,char **argv){
 float *h[NA],*d[NA],*r[NA],*v[NA];
 int anybad=0;
 int legacy_mode=argc>1&&!strcmp(argv[1],"legacy");
 int edge=argc>1&&!strcmp(argv[1],"edge");
 float edges[]={0.0f,-0.0f,1.0f,-1.0f,2.0f,-2.0f,0.125f,-0.125f,0.001f,1e-20f,-1e-20f,80.0f,-80.0f};
 const size_t ne=sizeof(edges)/sizeof(edges[0]);
 for(int k=0;k<NS;k++)parameters[k]=1.0e-5f;
 for(int a=0;a<NA;a++){
  size_t bytes=(sizes[a]+8)*sizeof(float);
  h[a]=malloc(bytes);r[a]=malloc(bytes);v[a]=malloc(bytes);CU(cudaMalloc((void**)&d[a],bytes));
  for(size_t i=0;i<sizes[a]+8;i++)h[a][i]=913.25f;
  for(size_t i=0;i<sizes[a];i++){
   float x=(float)(((i+a*613)*104729LL)%8192)/1024.0f-4.0f;
   if(edge)x=edges[(a?i/ne:i)%ne];
   h[a][i+4]=outputs[a]?127.25f:x;
  }
  memcpy(r[a],h[a],bytes);CU(cudaMemcpy(d[a],h[a],bytes,cudaMemcpyHostToDevice));
  d[a]+=4;h[a]+=4;r[a]+=4;v[a]+=4;
 }
 if(argc>1&&!strcmp(argv[1],"timing")){
  for(int k=0;k<2;k++)for(int w=0;w<5;w++){if(k)direct(d);else invoke(d);CU(cudaDeviceSynchronize());}
  for(int i=0;i<5;i++)for(int z=0;z<2;z++){int k=z^(i&1);CU(cudaDeviceSynchronize());double t=now_ms();if(k)direct(d);else invoke(d);CU(cudaDeviceSynchronize());printf("SAMPLE path=%s sample=%d ms=%.9f\n",k?"direct_cudnn":"raised",i,now_ms()-t);}
  puts("PASS timing");return 0;
 }
 for(int parameter_case=0;parameter_case<(legacy_mode?1:(NS?2:1));parameter_case++){
  for(int k=0;k<NS;k++)parameters[k]=parameter_case?0.125f:1.0e-5f;
  reference(r);direct(d);CU(cudaDeviceSynchronize());
  for(int a=0;a<NA;a++)CU(cudaMemcpy(v[a]-4,d[a]-4,(sizes[a]+8)*sizeof(float),cudaMemcpyDeviceToHost));
  for(int resident=0;resident<2;resident++){
   if(legacy_mode)legacy(resident?d:h);else invoke(resident?d:h);CU(cudaDeviceSynchronize());
   double maxabs=0,maxrel=0;size_t bad=0,vendorbad=0,count=0;
   for(int a=0;a<NA;a++){
    size_t bytes=(sizes[a]+8)*sizeof(float);float *got=malloc(bytes);
    if(resident)CU(cudaMemcpy(got,d[a]-4,bytes,cudaMemcpyDeviceToHost));else memcpy(got,h[a]-4,bytes);
    for(int j=0;j<4;j++)if(got[j]!=913.25f||got[sizes[a]+4+j]!=913.25f)bad++;
    for(size_t i=0;i<sizes[a];i++){
     if(outputs[a]){
      count++;
      if(isfinite(got[i+4])&&isfinite(r[a][i])){double e=fabs((double)got[i+4]-r[a][i]);maxabs=fmax(maxabs,e);maxrel=fmax(maxrel,e/fmax(1e-30,fabs(r[a][i])));}
      if(!equal(got[i+4],r[a][i],0)){if(bad<4)printf("MISMATCH i=%zu got=%a reference=%a\n",i,got[i+4],r[a][i]);bad++;}
      if(!equal(got[i+4],v[a][i],1))vendorbad++;
     }else if(memcmp(&got[i+4],&r[a][i],sizeof(float)))bad++;
    }
    free(got);
   }
   printf("CHECK edge=%d parameters=%d resident=%d count=%zu bad=%zu vendorbad=%zu maxabs=%.9g maxrel=%.9g\n",edge,parameter_case,resident,count,bad,vendorbad,maxabs,maxrel);
   if(bad||vendorbad)anybad=1;
  }
 }
 if(anybad)return 3;
 puts("PASS correctness");return 0;
}
