// ExaSP2-style dense/SP2/CG fixture designed to raise cleanly.
//
// This keeps dense normalization, square, SP2 update/selection, diagonal trace
// extraction, SpMV, AXPBY, and CG-step shapes while avoiding
// BML/MPI/container-level code.

#define EN 16

void exasp2_pipeline_easy(double rho[EN][EN], const double h[EN][EN],
                          double x[EN][EN], double x2[EN][EN],
                          const double a[EN][EN], const double v[EN],
                          double y[EN], double cg_x[EN], double cg_r[EN],
                          double cg_p[EN], const double cg_Ap[EN],
                          double axpby_out[EN], const double axpby_a[EN],
                          const double axpby_b[EN], double trace_diag[EN],
                          double emax, double emin, double alpha,
                          double beta, int take_square) {
  double range = emax - emin;

  // Normalize dense Hamiltonian into SP2 domain.
  for (int i = 0; i < EN; i++)
    for (int j = 0; j < EN; j++)
      rho[i][j] = -h[i][j] / range;

  for (int i = 0; i < EN; i++)
    rho[i][i] += emax / range;

  // Dense square.
  for (int i = 0; i < EN; i++)
    for (int j = 0; j < EN; j++) {
      double sum = 0.0;
      for (int k = 0; k < EN; k++)
        sum += rho[i][k] * rho[k][j];
      x2[i][j] = sum;
    }

  // SP2 selection/update.
  for (int i = 0; i < EN; i++)
    for (int j = 0; j < EN; j++)
      x[i][j] = take_square ? x2[i][j] : 2.0 * rho[i][j] - x2[i][j];

  // Trace diagonal extraction. The scalar trace reduction is intentionally
  // left out because one-element scalar reductions currently need a lowering
  // fix before they are safe to use in composed pipeline fixtures.
  for (int i = 0; i < EN; i++)
    trace_diag[i] = x[i][i];

  // SpMV.
  for (int i = 0; i < EN; i++) {
    double sum = 0.0;
    for (int j = 0; j < EN; j++)
      sum += a[i][j] * v[j];
    y[i] = sum;
  }

  // AXPBY.
  for (int i = 0; i < EN; i++)
    axpby_out[i] = alpha * axpby_a[i] + beta * axpby_b[i];

  // CG step.
  for (int i = 0; i < EN; i++) {
    cg_x[i] += alpha * cg_p[i];
    cg_r[i] -= alpha * cg_Ap[i];
    cg_p[i] = cg_r[i] + beta * cg_p[i];
  }
}
