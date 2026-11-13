#ifndef B
#define B 64
#endif
void aten_nested_batch_offsets_cpu(int size[B],int out[B+1]){out[0]=0;for(int b=0;b<B;++b)out[b+1]=out[b]+size[b];}
