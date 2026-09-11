#define T 24
#define B 4
#define C 12
#define L 5
#define S (2*L+1)
extern float expf(float);extern float logf(float);
void aten_ctc_loss_cpu(float logp[T][B][C],int labels[B][L],int blank,float loss[B],float alpha[B][T][S]){for(int b=0;b<B;++b){for(int t=0;t<T;++t)for(int s=0;s<S;++s)alpha[b][t][s]=0;alpha[b][0][0]=expf(logp[0][b][blank]);alpha[b][0][1]=expf(logp[0][b][labels[b][0]]);for(int t=1;t<T;++t)for(int s=0;s<S;++s){int lab=(s&1)?labels[b][s/2]:blank;float a=alpha[b][t-1][s];if(s>0)a+=alpha[b][t-1][s-1];if(s>1&&lab!=blank&&lab!=((s-2)&1?labels[b][(s-2)/2]:blank))a+=alpha[b][t-1][s-2];alpha[b][t][s]=a*expf(logp[t][b][lab]);}float z=alpha[b][T-1][S-1]+alpha[b][T-1][S-2];loss[b]=-logf(z);}}
