#define SEG 16
#define N 128
void aten_segment_reduce_lengths_backward_cpu(float x[N],int lengths[SEG],float result[SEG],float grad[SEG],int reduce,float out[N]){int p=0;for(int s=0;s<SEG;++s)for(int i=0;i<lengths[s];++i){float g=reduce==0?grad[s]:(reduce==1?grad[s]/lengths[s]:(x[p]==result[s]?grad[s]:0));out[p++]=g;}}
