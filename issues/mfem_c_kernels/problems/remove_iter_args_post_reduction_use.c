/*
 * Minimal form of the remove-iter-args dominance bug found while normalizing
 * MFEM mass apply.  The reduction result and a load defined after the
 * reduction are used by the same multiply.  remove-iter-args incorrectly
 * moves the multiply into the reduction region without moving the load.
 */
enum { D=4, Q=5, E=2 };
void remove_iter_args_post_reduction_use(const double *B, const double *coeff,
                                         const double *x, double *scratch) {
  for (int e=0;e<E;++e)
    for (int q=0;q<Q;++q) {
      double acc=0.0;
      for (int d=0;d<D;++d)
        acc += B[q*D+d]*x[e*D+d];
      scratch[e*Q+q]=acc*coeff[e*Q+q];
    }
}
