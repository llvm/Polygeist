#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void lhsinit(double lhs[][3][5][5], int size);

int main(void) {
  const int size=4;
  double (*lhs)[3][5][5]=malloc((size_t)(size+1)*sizeof(*lhs));
  if (!lhs) return 100;
  for (int i=0;i<=size;++i)
    for (int b=0;b<3;++b)
      for (int n=0;n<5;++n)
        for (int m=0;m<5;++m) lhs[i][b][n][m]=7.0+i+b+n+m;
  lhsinit(lhs,size);
  for (int i=0;i<=size;++i)
    for (int b=0;b<3;++b)
      for (int n=0;n<5;++n)
        for (int m=0;m<5;++m) {
          double expected=7.0+i+b+n+m;
          if (i==0 || i==size) expected=(b==1 && n==m)?1.0:0.0;
          if (lhs[i][b][n][m]!=expected) {
            fprintf(stderr,"BT lhsinit mismatch [%d][%d][%d][%d]: %g vs %g\n",
                    i,b,n,m,lhs[i][b][n][m],expected);
            return 1;
          }
        }
  puts("npb-bt-lhsinit-source: PASS");
  free(lhs);
  return 0;
}
