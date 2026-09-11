// PolyBench-style extraction of NPB MG's `norm2u3` kernel.
// Original (NPB3.0-omp-C/MG/mg.c lines 806-860): computes L2 norm `rnm2` and
// L-infinity norm `rnmu` over interior of r. The L-infinity branch uses
// `fabs` + `max` (non-affine — likely won't lift); the L2 branch is a pure
// sum-of-squares reduction (should lift).

#define N1 34
#define N2 34
#define N3 34

double  my_fabs(double x) { return x < 0.0 ? -x : x; }
double  my_max(double a, double b) { return a > b ? a : b; }

void mg_norm2u3(double r[N3][N2][N1],
                int n1, int n2, int n3,
                double *rnm2, double *rnmu,
                int nx, int ny, int nz) {
  double s = 0.0;
  int i3, i2, i1, n;
  double a = 0.0, tmp = 0.0;

  n = nx * ny * nz;

  for (i3 = 1; i3 < n3 - 1; i3++) {
    for (i2 = 1; i2 < n2 - 1; i2++) {
      for (i1 = 1; i1 < n1 - 1; i1++) {
        s = s + r[i3][i2][i1] * r[i3][i2][i1];
        tmp = my_fabs(r[i3][i2][i1]);
        if (tmp > a) a = tmp;
      }
    }
  }

  *rnm2 = s / (double)n;  // NPB does a sqrt after; left as caller's job
  *rnmu = a;
}
