#define M 64
#define K 128
void aten_bf16_dot_cpu(float a[K],float b[K],float out[1]){float s=0;for(int k=0;k<K;++k)s+=a[k]*b[k];out[0]=s;}
