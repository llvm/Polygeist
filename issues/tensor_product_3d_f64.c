/* Float64 separable 3D tensor product.
 *
 *   out[qi,qj,qk] = sum(i,j,k)
 *       psi[qi,i] * psi[qj,j] * psi[qk,k] * u[i,j,k]
 *
 * This intentionally uses the source-level scalar accumulator form that
 * exercises nested affine iter_args and remove-iter-args.
 */
#define KP 4
#define KQ 5

void tensor_product_3d_f64(const double psi[KQ * KP],
                           const double u[KP * KP * KP],
                           double out[KQ * KQ * KQ]) {
  for (long qi = 0; qi < KQ; ++qi) {
    for (long qj = 0; qj < KQ; ++qj) {
      for (long qk = 0; qk < KQ; ++qk) {
        double acc = 0.0;
        for (long i = 0; i < KP; ++i) {
          for (long j = 0; j < KP; ++j) {
            for (long k = 0; k < KP; ++k) {
              acc += psi[qi * KP + i] * psi[qj * KP + j] *
                     psi[qk * KP + k] * u[(i * KP + j) * KP + k];
            }
          }
        }
        out[(qi * KQ + qj) * KQ + qk] = acc;
      }
    }
  }
}
