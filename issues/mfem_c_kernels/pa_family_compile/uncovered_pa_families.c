/*
 * Compile-only structural extracts for MFEM partial-assembly integrator
 * families not represented by the validated standalone corpus.
 *
 * These concrete FP64 fixtures retain the characteristic tensor-product,
 * face, transpose, and pointwise stages of the pinned MFEM kernels.  They are
 * inputs to a raising/matching coverage survey only: they have not yet been
 * linked to a numerical harness and must not be reported as correctness-
 * validated MFEM implementations.
 */
enum { D = 4, E = 3, Q = 5, NE = 2, NF = 2 };

#define S2(x,y,e) ((x) + D*((y) + D*(e)))
#define V2(x,y,c,e) ((x) + D*((y) + D*((c) + 2*(e))))
#define S3(x,y,z,e) ((x) + D*((y) + D*((z) + D*(e))))
#define V3(x,y,z,c,e) ((x) + D*((y) + D*((z) + D*((c) + 3*(e)))))
#define Q2I(x,y,e) ((x) + Q*((y) + Q*(e)))
#define Q3I(x,y,z,e) ((x) + Q*((y) + Q*((z) + Q*(e))))

/* fem/integ/bilininteg_dgtrace_kernels.hpp: PADGTraceApply2D. */
void mfem_pa_dg_trace_apply_2d(const double *B, const double *op,
                              const double *X, double *Y) {
  double flux[NF][2][Q];
  for (int f = 0; f < NF; ++f)
    for (int q = 0; q < Q; ++q) {
      double u0 = 0.0, u1 = 0.0;
      for (int d = 0; d < D; ++d) {
        u0 += X[d + D * (0 + 2 * f)] * B[q * D + d];
        u1 += X[d + D * (1 + 2 * f)] * B[q * D + d];
      }
      flux[f][0][q] = op[q + Q * (0 + 4 * f)] * u0 +
                      op[q + Q * (1 + 4 * f)] * u1;
      flux[f][1][q] = op[q + Q * (2 + 4 * f)] * u0 +
                      op[q + Q * (3 + 4 * f)] * u1;
    }
  for (int f = 0; f < NF; ++f)
    for (int side = 0; side < 2; ++side)
      for (int d = 0; d < D; ++d) {
        double a = 0.0;
        for (int q = 0; q < Q; ++q)
          a += flux[f][side][q] * B[q * D + d];
        Y[d + D * (side + 2 * f)] += a;
      }
}

/* fem/integ/bilininteg_dgdiffusion_kernels.hpp: PADGDiffusionApply2D. */
void mfem_pa_dg_diffusion_apply_2d(const double *B, const double *G,
                                  const double *op, const double *X,
                                  double *Y) {
  double value[NF][2][Q], deriv[NF][2][Q], fv[NF][2][Q], fd[NF][2][Q];
  for (int f = 0; f < NF; ++f)
    for (int side = 0; side < 2; ++side)
      for (int q = 0; q < Q; ++q) {
        double v = 0.0, g = 0.0;
        for (int d = 0; d < D; ++d) {
          double x = X[d + D * (side + 2 * f)];
          v += x * B[q * D + d];
          g += x * G[q * D + d];
        }
        value[f][side][q] = v;
        deriv[f][side][q] = g;
      }
  for (int f = 0; f < NF; ++f)
    for (int q = 0; q < Q; ++q) {
      double jump = value[f][0][q] - value[f][1][q];
      double avg = 0.5 * (deriv[f][0][q] + deriv[f][1][q]);
      double penalty = op[q + Q * f];
      fv[f][0][q] = penalty * jump - avg;
      fv[f][1][q] = -fv[f][0][q];
      fd[f][0][q] = -0.5 * jump;
      fd[f][1][q] = -0.5 * jump;
    }
  for (int f = 0; f < NF; ++f)
    for (int side = 0; side < 2; ++side)
      for (int d = 0; d < D; ++d) {
        double a = 0.0;
        for (int q = 0; q < Q; ++q)
          a += fv[f][side][q] * B[q * D + d] +
               fd[f][side][q] * G[q * D + d];
        Y[d + D * (side + 2 * f)] += a;
      }
}

/* fem/integ/bilininteg_elasticity_pa.cpp: ElasticityComponentIntegrator. */
void mfem_pa_elasticity_component_apply_2d(
    const double *B, const double *G, const double *op,
    const double *X, double *Y) {
  double h[NE][2][Q][Q];
  for (int e = 0; e < NE; ++e)
    for (int qy = 0; qy < Q; ++qy)
      for (int qx = 0; qx < Q; ++qx) {
        double gx = 0.0, gy = 0.0;
        for (int dy = 0; dy < D; ++dy)
          for (int dx = 0; dx < D; ++dx) {
            double x = X[S2(dx, dy, e)];
            gx += x * G[qx * D + dx] * B[qy * D + dy];
            gy += x * B[qx * D + dx] * G[qy * D + dy];
          }
        int p = Q2I(qx, qy, e);
        h[e][0][qy][qx] = op[p] * gx + op[p + Q*Q*NE] * gy;
        h[e][1][qy][qx] = op[p + Q*Q*NE] * gx +
                          op[p + 2*Q*Q*NE] * gy;
      }
  for (int e = 0; e < NE; ++e)
    for (int dy = 0; dy < D; ++dy)
      for (int dx = 0; dx < D; ++dx) {
        double a = 0.0;
        for (int qy = 0; qy < Q; ++qy)
          for (int qx = 0; qx < Q; ++qx)
            a += h[e][0][qy][qx] * G[qx * D + dx] * B[qy * D + dy] +
                 h[e][1][qy][qx] * B[qx * D + dx] * G[qy * D + dy];
        Y[S2(dx, dy, e)] += a;
      }
}

/* fem/integ/bilininteg_interp_pa.cpp: GradientInterpolator. */
void mfem_pa_gradient_interpolator_2d(const double *B, const double *G,
                                     const double *X, double *Y) {
  for (int e = 0; e < NE; ++e)
    for (int c = 0; c < 2; ++c)
      for (int qy = 0; qy < Q; ++qy)
        for (int qx = 0; qx < Q; ++qx) {
          double a = 0.0;
          for (int dy = 0; dy < D; ++dy)
            for (int dx = 0; dx < D; ++dx)
              a += X[S2(dx,dy,e)] *
                   (c == 0 ? G[qx*D+dx] : B[qx*D+dx]) *
                   (c == 1 ? G[qy*D+dy] : B[qy*D+dy]);
          Y[qx + Q * (qy + Q * (c + 2 * e))] = a;
        }
}

/* fem/integ/bilininteg_interp_pa.cpp: IdentityInterpolator. */
void mfem_pa_identity_interpolator_2d(const double *B,
                                     const double *X, double *Y) {
  for (int e = 0; e < NE; ++e)
    for (int c = 0; c < 2; ++c)
      for (int qy = 0; qy < Q; ++qy)
        for (int qx = 0; qx < Q; ++qx) {
          double a = 0.0;
          for (int dy = 0; dy < D; ++dy)
            for (int dx = 0; dx < D; ++dx)
              a += X[V2(dx,dy,c,e)] * B[qx*D+dx] * B[qy*D+dy];
          Y[qx + Q * (qy + Q * (c + 2 * e))] = a;
        }
}

/* fem/integ/bilininteg_mixedcurl_pa.cpp: MixedScalarCurlIntegrator. */
void mfem_pa_mixed_scalar_curl_2d(const double *B, const double *G,
                                 const double *X, double *Y) {
  double curl[NE][Q][Q];
  for (int e = 0; e < NE; ++e)
    for (int qy = 0; qy < Q; ++qy)
      for (int qx = 0; qx < Q; ++qx) {
        double a = 0.0;
        for (int dy = 0; dy < D; ++dy)
          for (int dx = 0; dx < D; ++dx)
            a += X[V2(dx,dy,1,e)] * G[qx*D+dx] * B[qy*D+dy] -
                 X[V2(dx,dy,0,e)] * B[qx*D+dx] * G[qy*D+dy];
        curl[e][qy][qx] = a;
      }
  for (int e = 0; e < NE; ++e)
    for (int dy = 0; dy < D; ++dy)
      for (int dx = 0; dx < D; ++dx) {
        double a = 0.0;
        for (int qy = 0; qy < Q; ++qy)
          for (int qx = 0; qx < Q; ++qx)
            a += curl[e][qy][qx] * B[qx*D+dx] * B[qy*D+dy];
        Y[S2(dx,dy,e)] += a;
      }
}

/* fem/integ/bilininteg_mixedcurl_pa.cpp: MixedVectorCurlIntegrator. */
void mfem_pa_mixed_vector_curl_3d(const double *B, const double *G,
                                 const double *X, double *Y) {
  double curl[NE][3][Q][Q][Q];
  for (int e=0;e<NE;++e) for (int qz=0;qz<Q;++qz)
    for (int qy=0;qy<Q;++qy) for (int qx=0;qx<Q;++qx) {
      double c0=0.0,c1=0.0,c2=0.0;
      for (int dz=0;dz<D;++dz) for (int dy=0;dy<D;++dy)
        for (int dx=0;dx<D;++dx) {
          double x0=X[V3(dx,dy,dz,0,e)], x1=X[V3(dx,dy,dz,1,e)];
          double x2=X[V3(dx,dy,dz,2,e)];
          c0 += (x2*B[qy*D+dy]*G[qz*D+dz]-x1*G[qy*D+dy]*B[qz*D+dz])*B[qx*D+dx];
          c1 += (x0*G[qz*D+dz]*B[qx*D+dx]-x2*G[qx*D+dx]*B[qz*D+dz])*B[qy*D+dy];
          c2 += (x1*G[qx*D+dx]*B[qy*D+dy]-x0*G[qy*D+dy]*B[qx*D+dx])*B[qz*D+dz];
        }
      curl[e][0][qz][qy][qx]=c0; curl[e][1][qz][qy][qx]=c1;
      curl[e][2][qz][qy][qx]=c2;
    }
  for (int e=0;e<NE;++e) for (int c=0;c<3;++c)
    for (int dz=0;dz<D;++dz) for (int dy=0;dy<D;++dy)
      for (int dx=0;dx<D;++dx) {
        double a=0.0;
        for (int qz=0;qz<Q;++qz) for (int qy=0;qy<Q;++qy)
          for (int qx=0;qx<Q;++qx)
            a += curl[e][c][qz][qy][qx]*B[qx*D+dx]*B[qy*D+dy]*B[qz*D+dz];
        Y[V3(dx,dy,dz,c,e)] += a;
      }
}

/* MixedVectorWeakCurlIntegrator: transpose of the mixed-vector curl map. */
void mfem_pa_mixed_vector_weak_curl_3d(const double *B, const double *G,
                                      const double *X, double *Y) {
  double q[NE][3][Q][Q][Q];
  for (int e=0;e<NE;++e) for (int c=0;c<3;++c)
    for (int qz=0;qz<Q;++qz) for (int qy=0;qy<Q;++qy)
      for (int qx=0;qx<Q;++qx) {
        double a=0.0;
        for (int dz=0;dz<D;++dz) for (int dy=0;dy<D;++dy)
          for (int dx=0;dx<D;++dx)
            a += X[V3(dx,dy,dz,c,e)]*B[qx*D+dx]*B[qy*D+dy]*B[qz*D+dz];
        q[e][c][qz][qy][qx]=a;
      }
  for (int e=0;e<NE;++e) for (int dz=0;dz<D;++dz)
    for (int dy=0;dy<D;++dy) for (int dx=0;dx<D;++dx) {
      double a0=0.0,a1=0.0,a2=0.0;
      for (int qz=0;qz<Q;++qz) for (int qy=0;qy<Q;++qy)
        for (int qx=0;qx<Q;++qx) {
          a0 += q[e][1][qz][qy][qx]*B[qx*D+dx]*B[qy*D+dy]*G[qz*D+dz]
              - q[e][2][qz][qy][qx]*B[qx*D+dx]*G[qy*D+dy]*B[qz*D+dz];
          a1 += q[e][2][qz][qy][qx]*G[qx*D+dx]*B[qy*D+dy]*B[qz*D+dz]
              - q[e][0][qz][qy][qx]*B[qx*D+dx]*B[qy*D+dy]*G[qz*D+dz];
          a2 += q[e][0][qz][qy][qx]*B[qx*D+dx]*G[qy*D+dy]*B[qz*D+dz]
              - q[e][1][qz][qy][qx]*G[qx*D+dx]*B[qy*D+dy]*B[qz*D+dz];
        }
      Y[V3(dx,dy,dz,0,e)]+=a0; Y[V3(dx,dy,dz,1,e)]+=a1;
      Y[V3(dx,dy,dz,2,e)]+=a2;
    }
}

/* fem/integ/bilininteg_mixedvecgrad_pa.cpp: MixedVectorGradientIntegrator. */
void mfem_pa_mixed_vector_gradient_2d(const double *B, const double *G,
                                     const double *X, double *Y) {
  for (int e=0;e<NE;++e) for (int out=0;out<2;++out)
    for (int qy=0;qy<Q;++qy) for (int qx=0;qx<Q;++qx) {
      double a=0.0;
      for (int in=0;in<2;++in) for (int dy=0;dy<D;++dy)
        for (int dx=0;dx<D;++dx) {
          double basis = out == 0 ? G[qx*D+dx]*B[qy*D+dy]
                                  : B[qx*D+dx]*G[qy*D+dy];
          a += X[V2(dx,dy,in,e)] * basis * (in == out ? 1.0 : 0.5);
        }
      Y[qx+Q*(qy+Q*(out+2*e))]=a;
    }
}

/* fem/integ/bilininteg_vectorfediv_pa.cpp: VectorFEDivergenceIntegrator. */
void mfem_pa_vector_fe_divergence_3d(const double *B, const double *G,
                                    const double *X, double *Y) {
  double div[NE][Q][Q][Q];
  for (int e=0;e<NE;++e) for (int qz=0;qz<Q;++qz)
    for (int qy=0;qy<Q;++qy) for (int qx=0;qx<Q;++qx) {
      double a=0.0;
      for (int dz=0;dz<D;++dz) for (int dy=0;dy<D;++dy)
        for (int dx=0;dx<D;++dx) {
          a += X[V3(dx,dy,dz,0,e)]*G[qx*D+dx]*B[qy*D+dy]*B[qz*D+dz];
          a += X[V3(dx,dy,dz,1,e)]*B[qx*D+dx]*G[qy*D+dy]*B[qz*D+dz];
          a += X[V3(dx,dy,dz,2,e)]*B[qx*D+dx]*B[qy*D+dy]*G[qz*D+dz];
        }
      div[e][qz][qy][qx]=a;
    }
  for (int e=0;e<NE;++e) for (int dz=0;dz<D;++dz)
    for (int dy=0;dy<D;++dy) for (int dx=0;dx<D;++dx) {
      double a=0.0;
      for (int qz=0;qz<Q;++qz) for (int qy=0;qy<Q;++qy)
        for (int qx=0;qx<Q;++qx)
          a += div[e][qz][qy][qx]*B[qx*D+dx]*B[qy*D+dy]*B[qz*D+dz];
      Y[S3(dx,dy,dz,e)] += a;
    }
}

#undef Q3I
#undef Q2I
#undef V3
#undef S3
#undef V2
#undef S2
