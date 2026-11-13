// conv3d.c — extracted from polybenchGpu/OpenMP/stencils/convolution-3d/.
// See conv2d.c in this directory for why extraction is needed (cgeist
// inlines main→init→kernel, optimizer constant-folds init's
// A[i,j,k] = f(i,j,k), conv body loses its ins).

#ifndef NI
#define NI 128
#endif
#ifndef NJ
#define NJ 128
#endif
#ifndef NK
#define NK 128
#endif

// 15-tap 3D stencil over a 3x3x3 neighbourhood, weights from
// polybenchGpu's original kernel_conv2d (yes, it's misnamed kernel_conv2d
// in conv3d.c upstream — sic). Note: the original has duplicated index
// expressions (`2 * A[i-1][j-1][k-1] + 5 * A[i-1][j-1][k-1]` etc.) — we
// preserve that here verbatim so the lifted body matches what the IR
// explorer's existing convolution-3d entry shows.
void kernel_conv2d(int ni, int nj, int nk,
                   double A[NI][NJ][NK], double B[NI][NJ][NK]) {
  int i, j, k;
  for (i = 1; i < ni - 1; ++i)
    for (j = 1; j < nj - 1; ++j)
      for (k = 1; k < nk - 1; ++k) {
        B[i][j][k] =  2 * A[i-1][j-1][k-1]  +  4 * A[i+1][j-1][k-1]
                   +  5 * A[i-1][j-1][k-1]  +  7 * A[i+1][j-1][k-1]
                   + -8 * A[i-1][j-1][k-1]  + 10 * A[i+1][j-1][k-1]
                   + -3 * A[ i ][j-1][ k ]
                   +  6 * A[ i ][ j ][ k ]
                   + -9 * A[ i ][j+1][ k ]
                   +  2 * A[i-1][j-1][k+1]  +  4 * A[i+1][j-1][k+1]
                   +  5 * A[i-1][ j ][k+1]  +  7 * A[i+1][ j ][k+1]
                   + -8 * A[i-1][j+1][k+1]  + 10 * A[i+1][j+1][k+1];
      }
}
