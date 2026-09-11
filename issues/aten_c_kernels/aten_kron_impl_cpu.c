#define A 16
#define B 12
#define C 8
#define D 10
void aten_kron_impl_cpu(float x[A][B],float y[C][D],float out[A*C][B*D]){for(int a=0;a<A;++a)for(int b=0;b<B;++b)for(int c=0;c<C;++c)for(int d=0;d<D;++d)out[a*C+c][b*D+d]=x[a][b]*y[c][d];}
