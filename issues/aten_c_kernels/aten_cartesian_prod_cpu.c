#define A 16
#define B 12
void aten_cartesian_prod_cpu(float a[A],float b[B],float out[A*B][2]){for(int i=0;i<A;++i)for(int j=0;j<B;++j){int p=i*B+j;out[p][0]=a[i];out[p][1]=b[j];}}
