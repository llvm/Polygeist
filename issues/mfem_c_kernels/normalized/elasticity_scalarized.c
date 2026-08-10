/* Fixed-dimension scalarization of MFEM's elasticity quadrature function. */
#ifndef MFEM_BENCH_NE
#define MFEM_BENCH_NE 2
#endif
enum { Q1D=5,NE=MFEM_BENCH_NE,NQ2=25,NQ3=125 };
#define Q2(e,a,b,p) ((p)+NQ2*((b)+2*((a)+2*(e))))
#define J2(e,i,j,p) ((p)+NQ2*((j)+2*((i)+2*(e))))
#define Q3(e,a,b,p) ((p)+NQ3*((b)+3*((a)+3*(e))))
#define J3(e,i,j,p) ((p)+NQ3*((j)+3*((i)+3*(e))))

// polygeist-arg-extents mfem_elasticity_qpoint_2d_scalarized: la=25*MFEM_BENCH_NE, mu=25*MFEM_BENCH_NE, J=100*MFEM_BENCH_NE, wt=25, Q=100*MFEM_BENCH_NE, R=100*MFEM_BENCH_NE
void mfem_elasticity_qpoint_2d_scalarized(const double *la,const double *mu,
                                           const double *J,const double *wt,
                                           const double *Q,double *R) {
#define LOOP2(OA,OB,VALUE) for(int e=0;e<NE;++e)for(int p=0;p<NQ2;++p){ \
    double a=J[J2(e,0,0,p)],b=J[J2(e,0,1,p)],c=J[J2(e,1,0,p)],d=J[J2(e,1,1,p)]; \
    double det=a*d-b*c,i00=d/det,i01=-b/det,i10=-c/det,i11=a/det; \
    double g00=Q[Q2(e,0,0,p)],g01=Q[Q2(e,0,1,p)],g10=Q[Q2(e,1,0,p)],g11=Q[Q2(e,1,1,p)]; \
    double div=g00+g11,w=wt[p]*det,l=la[p+NQ2*e],m=mu[p+NQ2*e]; \
    double s00=w*(l*i00*div+m*(i00*(g00+g00)+i01*(g01+g10))); \
    double s01=w*(l*i01*div+m*(i00*(g10+g01)+i01*(g11+g11))); \
    double s10=w*(l*i10*div+m*(i10*(g00+g00)+i11*(g01+g10))); \
    double s11=w*(l*i11*div+m*(i10*(g10+g01)+i11*(g11+g11))); \
    (void)s00;(void)s01;(void)s10;(void)s11;R[Q2(e,OA,OB,p)]=(VALUE); }
  LOOP2(0,0,s00) LOOP2(1,0,s01) LOOP2(0,1,s10) LOOP2(1,1,s11)
#undef LOOP2
}

// polygeist-arg-extents mfem_elasticity_qpoint_3d_scalarized: la=125*MFEM_BENCH_NE, mu=125*MFEM_BENCH_NE, J=1125*MFEM_BENCH_NE, wt=125, Q=1125*MFEM_BENCH_NE, R=1125*MFEM_BENCH_NE
void mfem_elasticity_qpoint_3d_scalarized(const double *la,const double *mu,
                                           const double *J,const double *wt,
                                           const double *Q,double *R) {
#define LOOP3(OA,OB,VALUE) for(int e=0;e<NE;++e)for(int p=0;p<NQ3;++p){ \
    double a=J[J3(e,0,0,p)],b=J[J3(e,0,1,p)],c=J[J3(e,0,2,p)]; \
    double d=J[J3(e,1,0,p)],f=J[J3(e,1,1,p)],g=J[J3(e,1,2,p)]; \
    double h=J[J3(e,2,0,p)],i=J[J3(e,2,1,p)],j=J[J3(e,2,2,p)]; \
    double det=a*(f*j-g*i)-b*(d*j-g*h)+c*(d*i-f*h); \
    double i00=(f*j-g*i)/det,i01=(c*i-b*j)/det,i02=(b*g-c*f)/det; \
    double i10=(g*h-d*j)/det,i11=(a*j-c*h)/det,i12=(c*d-a*g)/det; \
    double i20=(d*i-f*h)/det,i21=(b*h-a*i)/det,i22=(a*f-b*d)/det; \
    double g00=Q[Q3(e,0,0,p)],g01=Q[Q3(e,0,1,p)],g02=Q[Q3(e,0,2,p)]; \
    double g10=Q[Q3(e,1,0,p)],g11=Q[Q3(e,1,1,p)],g12=Q[Q3(e,1,2,p)]; \
    double g20=Q[Q3(e,2,0,p)],g21=Q[Q3(e,2,1,p)],g22=Q[Q3(e,2,2,p)]; \
    double div=g00+g11+g22,w=wt[p]*det,l=la[p+NQ3*e],m=mu[p+NQ3*e]; \
    double s00=w*(l*i00*div+m*(i00*(g00+g00)+i01*(g01+g10)+i02*(g02+g20))); \
    double s01=w*(l*i01*div+m*(i00*(g10+g01)+i01*(g11+g11)+i02*(g12+g21))); \
    double s02=w*(l*i02*div+m*(i00*(g20+g02)+i01*(g21+g12)+i02*(g22+g22))); \
    double s10=w*(l*i10*div+m*(i10*(g00+g00)+i11*(g01+g10)+i12*(g02+g20))); \
    double s11=w*(l*i11*div+m*(i10*(g10+g01)+i11*(g11+g11)+i12*(g12+g21))); \
    double s12=w*(l*i12*div+m*(i10*(g20+g02)+i11*(g21+g12)+i12*(g22+g22))); \
    double s20=w*(l*i20*div+m*(i20*(g00+g00)+i21*(g01+g10)+i22*(g02+g20))); \
    double s21=w*(l*i21*div+m*(i20*(g10+g01)+i21*(g11+g11)+i22*(g12+g21))); \
    double s22=w*(l*i22*div+m*(i20*(g20+g02)+i21*(g21+g12)+i22*(g22+g22))); \
    (void)s00;(void)s01;(void)s02;(void)s10;(void)s11;(void)s12; \
    (void)s20;(void)s21;(void)s22;R[Q3(e,OA,OB,p)]=(VALUE); }
  LOOP3(0,0,s00) LOOP3(1,0,s01) LOOP3(2,0,s02)
  LOOP3(0,1,s10) LOOP3(1,1,s11) LOOP3(2,1,s12)
  LOOP3(0,2,s20) LOOP3(1,2,s21) LOOP3(2,2,s22)
#undef LOOP3
}
