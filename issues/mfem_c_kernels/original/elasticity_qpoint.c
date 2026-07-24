/*
 * Concrete quadrature-point stage from MFEM ElasticityAddMultPA_.
 * Input/output Q layout is [element][vector component][derivative][point].
 */
enum { Q1D = 5, NE = 2, NQ2 = 25, NQ3 = 125 };
#define Q2(e,a,b,p) ((p) + NQ2*((b) + 2*((a) + 2*(e))))
#define J2(e,i,j,p) ((p) + NQ2*((j) + 2*((i) + 2*(e))))
#define Q3(e,a,b,p) ((p) + NQ3*((b) + 3*((a) + 3*(e))))
#define J3(e,i,j,p) ((p) + NQ3*((j) + 3*((i) + 3*(e))))

void mfem_elasticity_qpoint_2d(const double *lambda, const double *mu,
                               const double *J, const double *weights,
                               double *Q) {
  for (int e = 0; e < NE; ++e)
    for (int p = 0; p < NQ2; ++p) {
      double a=J[J2(e,0,0,p)], b=J[J2(e,0,1,p)];
      double c=J[J2(e,1,0,p)], d=J[J2(e,1,1,p)];
      double det=a*d-b*c;
      double inv[2][2]={{d/det,-b/det},{-c/det,a/det}};
      double grad[2][2], out[2][2];
      for (int i=0;i<2;++i) for (int j=0;j<2;++j)
        grad[i][j]=Q[Q2(e,i,j,p)];
      double div=grad[0][0]+grad[1][1];
      double w=weights[p]*det;
      for (int m=0;m<2;++m) for (int q=0;q<2;++q) {
        double contraction=0.0;
        for (int i=0;i<2;++i) for (int j=0;j<2;++j)
          contraction += ((i==q)*inv[m][j]+(j==q)*inv[m][i])
                       * (grad[i][j]+grad[j][i]);
        out[m][q]=w*(lambda[p+NQ2*e]*inv[m][q]*div
                     +0.5*mu[p+NQ2*e]*contraction);
      }
      for (int m=0;m<2;++m) for (int q=0;q<2;++q)
        Q[Q2(e,q,m,p)]=out[m][q];
    }
}

void mfem_elasticity_qpoint_3d(const double *lambda, const double *mu,
                               const double *J, const double *weights,
                               double *Q) {
  for (int e = 0; e < NE; ++e)
    for (int p = 0; p < NQ3; ++p) {
      double a=J[J3(e,0,0,p)], b=J[J3(e,0,1,p)], c=J[J3(e,0,2,p)];
      double d=J[J3(e,1,0,p)], f=J[J3(e,1,1,p)], g=J[J3(e,1,2,p)];
      double h=J[J3(e,2,0,p)], i=J[J3(e,2,1,p)], j=J[J3(e,2,2,p)];
      double det=a*(f*j-g*i)-b*(d*j-g*h)+c*(d*i-f*h);
      double inv[3][3];
      inv[0][0]=(f*j-g*i)/det; inv[0][1]=(c*i-b*j)/det;
      inv[0][2]=(b*g-c*f)/det; inv[1][0]=(g*h-d*j)/det;
      inv[1][1]=(a*j-c*h)/det; inv[1][2]=(c*d-a*g)/det;
      inv[2][0]=(d*i-f*h)/det; inv[2][1]=(b*h-a*i)/det;
      inv[2][2]=(a*f-b*d)/det;
      double grad[3][3], out[3][3];
      for (int x=0;x<3;++x) for (int y=0;y<3;++y)
        grad[x][y]=Q[Q3(e,x,y,p)];
      double div=grad[0][0]+grad[1][1]+grad[2][2];
      double w=weights[p]*det;
      for (int m=0;m<3;++m) for (int q=0;q<3;++q) {
        double contraction=0.0;
        for (int x=0;x<3;++x) for (int y=0;y<3;++y)
          contraction += ((x==q)*inv[m][y]+(y==q)*inv[m][x])
                       * (grad[x][y]+grad[y][x]);
        out[m][q]=w*(lambda[p+NQ3*e]*inv[m][q]*div
                     +0.5*mu[p+NQ3*e]*contraction);
      }
      for (int m=0;m<3;++m) for (int q=0;q<3;++q)
        Q[Q3(e,q,m,p)]=out[m][q];
    }
}
