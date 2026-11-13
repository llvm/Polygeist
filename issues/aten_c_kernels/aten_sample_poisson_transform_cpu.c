#define N 1024
extern float expf(float);void aten_sample_poisson_transform_cpu(float lambda[N],float uniform[N][32],int out[N]){for(int i=0;i<N;++i){float p=expf(-lambda[i]),s=p;int k=0;while(k<31&&uniform[i][k]>s){++k;p*=lambda[i]/k;s+=p;}out[i]=k;}}
