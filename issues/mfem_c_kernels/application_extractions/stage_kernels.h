#ifndef MFEM_APPLICATION_STAGE_KERNELS_H
#define MFEM_APPLICATION_STAGE_KERNELS_H

/* Give each included extraction private constant names.  The numerical
 * functions retain their manifest names and are inlined by cgeist into an
 * application entry point selected with --function. */

#define D1D MASS_D1D
#define Q1D MASS_Q1D
#define NE MASS_NE
#include "../normalized/mass_stage_sliced.c"
#undef X2
#undef D2
#undef X3
#undef D3
#undef NE
#undef Q1D
#undef D1D

#define D1D DIFF_D1D
#define Q1D DIFF_Q1D
#define NE DIFF_NE
#include "../normalized/diffusion_stage_sliced.c"
#undef X2
#undef X3
#undef O2
#undef O3
#undef NE
#undef Q1D
#undef D1D

#define D1D CONV_D1D
#define Q1D CONV_Q1D
#define NE CONV_NE
#include "../normalized/convection_stage_sliced.c"
#undef X2
#undef X3
#undef O2
#undef O3
#undef NE
#undef Q1D
#undef D1D

#define D1D DERHAM2_D1D
#define Q1D DERHAM2_Q1D
#define EDGE DERHAM2_EDGE
#define NE DERHAM2_NE
#define N2 DERHAM2_N2
#include "../normalized/de_rham2_stage_sliced.c"
#undef V
#undef O
#undef N2
#undef NE
#undef EDGE
#undef Q1D
#undef D1D

#define D1D CURL3_D1D
#define Q1D CURL3_Q1D
#define EDGE CURL3_EDGE
#define NE CURL3_NE
#define N3 CURL3_N3
#include "../normalized/curlcurl3_stage_sliced.c"
#undef V
#undef OP
#undef N3
#undef NE
#undef EDGE
#undef Q1D
#undef D1D

#define D1D DIV3_D1D
#define Q1D DIV3_Q1D
#define EDGE DIV3_EDGE
#define NE DIV3_NE
#define N3 DIV3_N3
#include "../normalized/divdiv3_stage_sliced.c"
#undef V
#undef O
#undef N3
#undef NE
#undef EDGE
#undef Q1D
#undef D1D

#define D1D GRAD_D1D
#define Q1D GRAD_Q1D
#define VDIM GRAD_VDIM
#include "../normalized/gradient_stage_sliced.c"
#undef F2
#undef F3
#undef Q2
#undef Q3
#undef VDIM
#undef Q1D
#undef D1D

#define Q1D ELAS_Q1D
#define NE ELAS_NE
#define NQ2 ELAS_NQ2
#define NQ3 ELAS_NQ3
#include "../normalized/elasticity_scalarized.c"
#undef J2
#undef J3
#undef Q2
#undef Q3
#undef NQ3
#undef NQ2
#undef NE
#undef Q1D

#include "missing_stage_kernels.c"

#endif
