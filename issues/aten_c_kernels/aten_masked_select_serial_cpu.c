#ifndef R
#define R 32
#endif
#ifndef K
#define K 64
#endif
#ifndef S
#define S 128
#endif
void aten_masked_select_serial_cpu(float input[S],int mask[S],float out[S],int count[1]){int p=0;for(int i=0;i<S;++i)if(mask[i])out[p++]=input[i];count[0]=p;}
