#define T 24
#define B 4
#define C 12
#define L 5
#define S (2*L+1)
extern float expf(float);
void aten_ctc_loss_backward_cpu(float logp[T][B][C],int labels[B][L],int blank,float alpha[B][T][S],float grad_loss[B],float out[T][B][C]){for(int t=0;t<T;++t)for(int b=0;b<B;++b)for(int c=0;c<C;++c)out[t][b][c]=expf(logp[t][b][c])*grad_loss[b];for(int b=0;b<B;++b){float beta[T][S];for(int t=0;t<T;++t)for(int s=0;s<S;++s)beta[t][s]=0;beta[T-1][S-1]=beta[T-1][S-2]=1;for(int t=T-2;t>=0;--t)for(int s=0;s<S;++s){int lab=(s&1)?labels[b][s/2]:blank;float v=beta[t+1][s]*expf(logp[t+1][b][lab]);if(s+1<S){int l1=((s+1)&1)?labels[b][(s+1)/2]:blank;v+=beta[t+1][s+1]*expf(logp[t+1][b][l1]);}if(s+2<S){int l2=((s+2)&1)?labels[b][(s+2)/2]:blank;if(lab!=blank&&lab!=l2)v+=beta[t+1][s+2]*expf(logp[t+1][b][l2]);}beta[t][s]=v;}float z=alpha[b][T-1][S-1]+alpha[b][T-1][S-2];for(int t=0;t<T;++t)for(int s=0;s<S;++s){int lab=(s&1)?labels[b][s/2]:blank;out[t][b][lab]-=grad_loss[b]*alpha[b][t][s]*beta[t][s]/z;}}}
