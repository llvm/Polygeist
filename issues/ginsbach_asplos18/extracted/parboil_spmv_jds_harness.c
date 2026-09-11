#include <math.h>
#include <stdio.h>

void parboil_spmv_jds_core(int dim, int depth, int len,
                            int h_nzcnt[dim], int h_ptr[depth],
                            int h_indices[len], float h_data[len],
                            float h_x_vector[dim], int h_perm[dim],
                            float h_Ax_vector[dim]);

int main(void) {
  enum { dim = 17, depth = 4, len = dim * depth };
  int nzcnt[dim], ptr[depth], indices[len], perm[dim];
  float data[len], x[dim], got[dim], expected[dim];
  for (int k = 0; k < depth; ++k)
    ptr[k] = k * dim;
  for (int i = 0; i < dim; ++i) {
    x[i] = 0.2f + (float)((i * 7) % 13) / 9.0f;
    got[i] = expected[i] = -99.0f;
  }
  for (int i = 0; i < dim; ++i) {
    nzcnt[i] = 1 + (i % depth);
    perm[i] = dim - 1 - i;
    float sum = 0.0f;
    for (int k = 0; k < depth; ++k) {
      int j = ptr[k] + i;
      indices[j] = (i + 2 * k + 1) % dim;
      data[j] = 0.13f * (float)(1 + k) + 0.01f * (float)i;
      if (k < nzcnt[i])
        sum += data[j] * x[indices[j]];
    }
    expected[perm[i]] = sum;
  }

  parboil_spmv_jds_core(dim, depth, len, nzcnt, ptr, indices, data, x,
                         perm, got);
  for (int i = 0; i < dim; ++i) {
    if (!isfinite(got[i]) || fabsf(got[i] - expected[i]) > 2.0e-6f) {
      fprintf(stderr,
              "parboil-spmv-jds-core: FAIL index=%d got=%g expected=%g\n",
              i, got[i], expected[i]);
      return 1;
    }
  }
  puts("parboil-spmv-jds-core: PASS");
  return 0;
}
