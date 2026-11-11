#include <stdint.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
extern int cudaMalloc(void **,unsigned long);
extern int cudaFree(void *);
extern int cudaMemcpy(void *,const void *,unsigned long,int);
extern int cudaDeviceSynchronize(void);
extern void polygeist_cub_prefix_max_lower_bounds_i32_sized(int,int,const int*,int,int*);
#define CUDA(x) do{if((x)!=0)return 2;}while(0)
int main(void){
 const int counts[]={0,1,17,513},row_counts[]={0,1,7,65};int cases=0;
 for(int ni=0;ni<4;++ni)for(int ri=0;ri<4;++ri){
  int n=counts[ni],rows=row_counts[ri];
  // Retain allocations for the runtime host-registration cache.
  int *x=calloc(n+1,sizeof(int)),*out=calloc(rows+1,sizeof(int));
  int *want=calloc(rows+1,sizeof(int)),*dx,*dy;
  CUDA(cudaMalloc((void**)&dx,(n+3)*sizeof(int)));
  CUDA(cudaMalloc((void**)&dy,(rows+4)*sizeof(int)));
  for(int scenario=0;scenario<8;++scenario){
   for(int i=0;i<n;++i){
    switch(scenario){
     case 0:x[i]=i*(rows+1)/(n?n:1);break;
     case 1:x[i]=(n-i-1)*(rows+1)/(n?n:1);break;
     case 2:x[i]=(int)(((unsigned)i*23411u+721u)%(rows+7))-3;break;
     case 3:x[i]=INT_MIN;break;
     case 4:x[i]=INT_MAX;break;
     case 5:x[i]=i==0?INT_MAX:INT_MIN;break;
     case 6:x[i]=i%2?INT_MAX:INT_MIN;break;
     default:x[i]=-1;break;
    }
   }
   int p=0;for(int r=0;r<=rows;++r){while(p<n && x[p]<r)++p;want[r]=p;}
   if(n)CUDA(cudaMemcpy(dx+3,x,n*sizeof(int),1));
   for(int mask=0;mask<4;++mask){
    for(int r=0;r<=rows;++r)out[r]=-7;
    polygeist_cub_prefix_max_lower_bounds_i32_sized(rows,n,mask&1?dx+3:x,rows+1,mask&2?dy+3:out);
    CUDA(cudaDeviceSynchronize());
    if(mask&2)CUDA(cudaMemcpy(out,dy+3,(rows+1)*sizeof(int),2));
    for(int r=0;r<=rows;++r)if(out[r]!=want[r]){fprintf(stderr,"prefix mismatch n=%d rows=%d scenario=%d mask=%d r=%d got=%d expected=%d\n",n,rows,scenario,mask,r,out[r],want[r]);return 1;}
    ++cases;
   }
  }
  CUDA(cudaFree(dx));CUDA(cudaFree(dy));free(want);
 }
 printf("PASS prefix lower bounds residency: %d cases\n",cases);return 0;
}
