#define SEG 16
#define N 128
void aten_segment_reduce_lengths_cpu(float x[N],int lengths[SEG],int reduce,float out[SEG]){int p=0;for(int s=0;s<SEG;++s){float v=reduce==2?-3.402823466e38f:(reduce==3?3.402823466e38f:0);for(int i=0;i<lengths[s];++i){float a=x[p++];if(reduce==0||reduce==1)v+=a;else if(reduce==2)v=v>a?v:a;else v=v<a?v:a;}if(reduce==1&&lengths[s])v/=lengths[s];out[s]=v;}}
