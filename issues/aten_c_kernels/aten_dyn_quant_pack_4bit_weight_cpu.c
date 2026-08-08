#define M 32
#define K 64
#define N 48
void aten_dyn_quant_pack_4bit_weight_cpu(float weight[N][K],unsigned char packed[N][K/2],float scale[N],float zero[N]){for(int n=0;n<N;++n){float lo=weight[n][0],hi=lo;for(int k=1;k<K;++k){lo=weight[n][k]<lo?weight[n][k]:lo;hi=weight[n][k]>hi?weight[n][k]:hi;}scale[n]=(hi-lo)/15.0f;zero[n]=-lo/scale[n];for(int k=0;k<K;k+=2){int q0=(int)(weight[n][k]/scale[n]+zero[n]+0.5f);int q1=(int)(weight[n][k+1]/scale[n]+zero[n]+0.5f);if(q0<0)q0=0;if(q0>15)q0=15;if(q1<0)q1=0;if(q1>15)q1=15;packed[n][k/2]=(unsigned char)(q0|(q1<<4));}}}
