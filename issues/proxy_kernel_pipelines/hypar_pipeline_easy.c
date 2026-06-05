// HyPar-style reconstruction/flux/update fixture designed to raise cleanly.
//
// This keeps finite-difference, WENO, limiter, upwind/centered flux, and
// timestep-update shapes while avoiding macro-generated multidimensional
// iterator state and solver structs.

#define HL 32
#define HNV 4

#define PK_ABS(x) ((x) < 0.0 ? -(x) : (x))
#define PK_MAX(a, b) ((a) > (b) ? (a) : (b))
#define PK_MIN(a, b) ((a) < (b) ? (a) : (b))

void hypar_pipeline_easy(const double fC[HL + 5][HNV],
                         const double u[HL][HNV],
                         const double source[HL][HNV],
                         const double eigL[HL + 1][HNV],
                         const double eigR[HL + 1][HNV],
                         const double uL[HL + 1][HNV],
                         const double uR[HL + 1][HNV],
                         double df[HL][HNV], double w1[HL + 1][HNV],
                         double w2[HL + 1][HNV], double w3[HL + 1][HNV],
                         double fL[HL + 1][HNV],
                         double fR[HL + 1][HNV],
                         double limited[HL][HNV],
                         double flux[HL + 1][HNV],
                         double reaction[HL][HNV],
                         double u_next[HL][HNV], double eps, double lambda,
                         double dt) {
  // Fourth-order derivative.
  for (int i = 0; i < HL; i++)
    for (int v = 0; v < HNV; v++)
      df[i][v] = (fC[i][v] - 8.0 * fC[i + 1][v] +
                  8.0 * fC[i + 3][v] - fC[i + 4][v]) /
                 12.0;

  // Jiang-Shu WENO weights.
  for (int i = 0; i < HL + 1; i++)
    for (int v = 0; v < HNV; v++) {
      double fm2 = fC[i][v], fm1 = fC[i + 1][v], f0 = fC[i + 2][v];
      double fp1 = fC[i + 3][v], fp2 = fC[i + 4][v];
      double b1 = (13.0 / 12.0) * (fm2 - 2.0 * fm1 + f0) *
                      (fm2 - 2.0 * fm1 + f0) +
                  0.25 * (fm2 - 4.0 * fm1 + 3.0 * f0) *
                      (fm2 - 4.0 * fm1 + 3.0 * f0);
      double b2 = (13.0 / 12.0) * (fm1 - 2.0 * f0 + fp1) *
                      (fm1 - 2.0 * f0 + fp1) +
                  0.25 * (fm1 - fp1) * (fm1 - fp1);
      double b3 = (13.0 / 12.0) * (f0 - 2.0 * fp1 + fp2) *
                      (f0 - 2.0 * fp1 + fp2) +
                  0.25 * (3.0 * f0 - 4.0 * fp1 + fp2) *
                      (3.0 * f0 - 4.0 * fp1 + fp2);
      double a1 = 0.1 / ((b1 + eps) * (b1 + eps));
      double a2 = 0.6 / ((b2 + eps) * (b2 + eps));
      double a3 = 0.3 / ((b3 + eps) * (b3 + eps));
      double sum = a1 + a2 + a3;
      w1[i][v] = a1 / sum;
      w2[i][v] = a2 / sum;
      w3[i][v] = a3 / sum;
    }

  // Fifth-order WENO left/right interface reconstructions.
  for (int i = 0; i < HL + 1; i++)
    for (int v = 0; v < HNV; v++) {
      double q1 = (2.0 * fC[i][v] - 7.0 * fC[i + 1][v] +
                   11.0 * fC[i + 2][v]) /
                  6.0;
      double q2 = (-fC[i + 1][v] + 5.0 * fC[i + 2][v] +
                   2.0 * fC[i + 3][v]) /
                  6.0;
      double q3 = (2.0 * fC[i + 2][v] + 5.0 * fC[i + 3][v] -
                   fC[i + 4][v]) /
                  6.0;
      fL[i][v] = w1[i][v] * q1 + w2[i][v] * q2 + w3[i][v] * q3;
      fR[i][v] = 0.5 * (fC[i + 2][v] + fC[i + 1][v]);
    }

  // Minmod limiter.
  for (int i = 0; i < HL; i++)
    for (int v = 0; v < HNV; v++) {
      double a = fC[i + 2][v] - fC[i + 1][v];
      double b = fC[i + 3][v] - fC[i + 2][v];
      limited[i][v] =
          (a * b <= 0.0) ? 0.0 : ((PK_ABS(a) < PK_ABS(b)) ? a : b);
    }

  // Local Lax-Friedrichs/upwind flux.
  for (int i = 0; i < HL + 1; i++)
    for (int v = 0; v < HNV; v++) {
      if (eigL[i][v] > 0.0 && eigR[i][v] > 0.0) {
        flux[i][v] = fL[i][v];
      } else if (eigL[i][v] < 0.0 && eigR[i][v] < 0.0) {
        flux[i][v] = fR[i][v];
      } else {
        double alpha = PK_MAX(PK_ABS(eigL[i][v]), PK_ABS(eigR[i][v]));
        flux[i][v] =
            0.5 * (fL[i][v] + fR[i][v] - alpha * (uR[i][v] - uL[i][v]));
      }
    }

  // Reaction and conservative update.
  for (int i = 0; i < HL; i++)
    for (int v = 0; v < HNV; v++) {
      reaction[i][v] = source[i][v] - lambda * u[i][v];
      u_next[i][v] =
          u[i][v] - dt * (flux[i + 1][v] - flux[i][v]) + dt * reaction[i][v];
    }
}
