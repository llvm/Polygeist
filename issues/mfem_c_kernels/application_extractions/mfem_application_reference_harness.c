/*
 * Independent direct-C reference plus the common correctness/timing harness
 * for one MFEM-derived application path selected by an MFEM_APP_* define.
 * The transformed entry is supplied by polygeist_build.sh; renaming below
 * ensures this normally compiled translation unit defines only the reference.
 */
#if defined(MFEM_APP_MTOP)
#define mfem_app_mtop_iso_elasticity_dfem_2d \
  mfem_app_mtop_iso_elasticity_dfem_2d_reference
#include "mtop_iso_elasticity_dfem_2d.c"
#undef mfem_app_mtop_iso_elasticity_dfem_2d
#elif defined(MFEM_APP_MINIMAL_SURFACE)
#define mfem_app_dfem_minimal_surface_2d \
  mfem_app_dfem_minimal_surface_2d_reference
#include "dfem_minimal_surface_2d.c"
#undef mfem_app_dfem_minimal_surface_2d
#elif defined(MFEM_APP_EX35P_H1)
#define mfem_app_ex35p_h1_3d mfem_app_ex35p_h1_3d_reference
#include "ex35p_pa_operators.c"
#undef mfem_app_ex35p_h1_3d
#elif defined(MFEM_APP_EX35P_HCURL)
#define mfem_app_ex35p_hcurl_3d mfem_app_ex35p_hcurl_3d_reference
#include "ex35p_pa_operators.c"
#undef mfem_app_ex35p_hcurl_3d
#elif defined(MFEM_APP_EX35P_HDIV)
#define mfem_app_ex35p_hdiv_3d mfem_app_ex35p_hdiv_3d_reference
#include "ex35p_pa_operators.c"
#undef mfem_app_ex35p_hdiv_3d
#elif defined(MFEM_APP_EX9P)
#define mfem_app_ex9p_mass_convection_2d \
  mfem_app_ex9p_mass_convection_2d_reference
#include "ex9p_mass_convection_2d.c"
#undef mfem_app_ex9p_mass_convection_2d
#elif defined(MFEM_APP_GRAD_DIV)
#define mfem_app_grad_div_3d mfem_app_grad_div_3d_reference
#include "grad_div_3d.c"
#undef mfem_app_grad_div_3d
#elif defined(MFEM_APP_ABS_MASS)
#define mfem_app_abs_l1_mass_3d mfem_app_abs_l1_mass_3d_reference
#include "abs_l1_jacobi_operators.c"
#undef mfem_app_abs_l1_mass_3d
#elif defined(MFEM_APP_ABS_DIFFUSION)
#define mfem_app_abs_l1_diffusion_3d \
  mfem_app_abs_l1_diffusion_3d_reference
#include "abs_l1_jacobi_operators.c"
#undef mfem_app_abs_l1_diffusion_3d
#elif defined(MFEM_APP_ABS_CURLCURL)
#define mfem_app_abs_l1_curlcurl_3d \
  mfem_app_abs_l1_curlcurl_3d_reference
#include "abs_l1_jacobi_operators.c"
#undef mfem_app_abs_l1_curlcurl_3d
#elif defined(MFEM_APP_NAVIER)
/* This extraction already provides a separately named direct reference. */
#define mfem_app_navier_tgv_pa_operators_3d \
  mfem_app_navier_tgv_pa_operators_3d_reference_unused
#include "navier_tgv_pressure_diffusion_3d.c"
#undef mfem_app_navier_tgv_pa_operators_3d
#else
#error "Select one MFEM_APP_* application path"
#endif

#include "mfem_application_jetson_harness.c"
