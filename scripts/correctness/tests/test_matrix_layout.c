#include <cblas.h>
#include <math.h>
#include "polygeist_v2_matrix_layout.inc"

int main(int argc, char **argv) {
  double a[128], before[128], x[5], y[5];
  for (int i=0; i<128; ++i) a[i]=before[i]=-777.0;
  for (int r=0; r<5; ++r) for (int c=0; c<5; ++c)
    a[r*20+c*4]=before[r*20+c*4]=(r*5+c-11)*0.125;
  for (int i=0; i<5; ++i) x[i]=(i-2)*0.5;
  if (argc>1 && !strcmp(argv[1],"capacity")) {
    (void)polygeist_v2_matrix_pack(a, 8, 5, 5, 20, 4, 8);
    return 99;
  }
  if (argc>1 && !strcmp(argv[1],"overflow")) {
    (void)polygeist_v2_matrix_pack(a, INT64_MAX, INT64_MAX, 5, 20, 4, 8);
    return 99;
  }
  if (argc>1 && !strcmp(argv[1],"overlap")) {
    polygeist_v2_matrix_require_disjoint(a,sizeof(a),a+1,16,1);
    return 99;
  }
  polygeist_v2_matrix_require_disjoint(a,sizeof(a),a+1,16,0);
  double *packed=polygeist_v2_matrix_pack(a,sizeof(a),5,5,20,4,8);
  cblas_dgemv(CblasRowMajor,CblasNoTrans,5,5,1.0,packed,5,x,1,0.0,y,1);
  for (int r=0; r<5; ++r) {
    double expected=0;
    for (int c=0; c<5; ++c) expected+=a[r*20+c*4]*x[c];
    if (fabs(y[r]-expected)>1e-14) return 1;
  }
  polygeist_v2_matrix_release(packed,a,sizeof(a),5,5,20,4,8,0);
  if (memcmp(a,before,sizeof(a))) return 2;
  double b[64], initial[64], left[8], right[6];
  for (int i=0; i<64; ++i) b[i]=initial[i]=-999.0;
  for (int r=0; r<4; ++r) for (int c=0; c<3; ++c)
    b[r+c*8]=initial[r+c*8]=(r+c)*0.25;
  for (int i=0; i<8; ++i) left[i]=(i-3)*0.5;
  for (int i=0; i<6; ++i) right[i]=(i-2)*0.125;
  packed=polygeist_v2_matrix_pack(b,sizeof(b),4,3,1,8,8);
  cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,4,3,2,
              1.0,left,2,right,3,1.0,packed,3);
  polygeist_v2_matrix_release(packed,b,sizeof(b),4,3,1,8,8,1);
  for (int i=0; i<64; ++i) {
    double expected=initial[i];
    int r=i%8,c=i/8;
    if (r<4 && c<3)
      for (int k=0; k<2; ++k) expected+=left[r*2+k]*right[k*3+c];
    if (fabs(b[i]-expected)>1e-14) return 3;
  }
  if (polygeist_v2_matrix_pack(a,sizeof(a),4,5,20,1,8)!=a) return 4;
  if (polygeist_v2_matrix_pack(a,sizeof(a),4,1,20,4,8)!=a) return 5;
  puts("PASS actual OpenBLAS strided GEMV, transposed output GEMM, guards, fast paths");
  return 0;
}
