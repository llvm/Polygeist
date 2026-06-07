#ifndef POLYGEIST_CUSTOM_LIBRARY_CUDA_STENCIL3D_7PT_H
#define POLYGEIST_CUSTOM_LIBRARY_CUDA_STENCIL3D_7PT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Structured 3D 7-point affine stencil over device-accessible pointers.
 *
 * Pointers are interpreted as already pointing at the logical center cell
 * for (i=0,j=0,k=0). Neighbor accesses use the input strides:
 *   xm = input - input_stride_i
 *   xp = input + input_stride_i
 *   ym = input - input_stride_j
 *   yp = input + input_stride_j
 *   zm = input - input_stride_k
 *   zp = input + input_stride_k
 *
 * For each output cell:
 *
 *   base = base_center * center
 *   if extra != NULL:
 *     base += base_extra * extra[i,j,k]
 *
 *   inner = coeff_center * center
 *         + coeff_xm * xm + coeff_xp * xp
 *         + coeff_ym * ym + coeff_yp * yp
 *         + coeff_zm * zm + coeff_zp * zp
 *   if extra != NULL:
 *     inner += coeff_extra * extra[i,j,k]
 *
 *   out = base + (coeff ? coeff[i,j,k] : 1) * inner
 *
 * This single form covers:
 *   - MiniAMR average7:
 *       coeff=NULL, base*=0, all coeff_* taps = 1/7, coeff_extra=0.
 *   - MiniAMR weighted7:
 *       base_center=1, coeff=cell coefficient,
 *       coeff_center=-6, neighbor coeffs=1.
 *   - HPGMG apply-op:
 *       coeff=NULL, coeff_center=a+6*b, neighbor coeffs=-b.
 *   - HPGMG residual:
 *       extra=rhs, base_extra=1,
 *       coeff_center=-(a+6*b), neighbor coeffs=b.
 *   - HPGMG weighted Jacobi-like smoother:
 *       base_center=1, extra=rhs, coeff=dinv,
 *       coeff_extra=weight,
 *       coeff_center=-weight*(a+6*b), neighbor coeffs=weight*b.
 *
 * Strides are in elements, not bytes. cuda_stream may be NULL for the default
 * stream, otherwise it must be a cudaStream_t cast to void*.
 */
void polygeist_custom_stencil3d_7pt_f64(
    int32_t nx, int32_t ny, int32_t nz,
    const double *input_center,
    int64_t input_stride_i, int64_t input_stride_j, int64_t input_stride_k,
    const double *extra,
    int64_t extra_stride_i, int64_t extra_stride_j, int64_t extra_stride_k,
    const double *coeff,
    int64_t coeff_stride_i, int64_t coeff_stride_j, int64_t coeff_stride_k,
    double *output,
    int64_t output_stride_i, int64_t output_stride_j, int64_t output_stride_k,
    double base_center, double base_extra, double coeff_extra,
    double coeff_center,
    double coeff_xm, double coeff_xp,
    double coeff_ym, double coeff_yp,
    double coeff_zm, double coeff_zp,
    void *cuda_stream);

void polygeist_custom_stencil3d_7pt_f32(
    int32_t nx, int32_t ny, int32_t nz,
    const float *input_center,
    int64_t input_stride_i, int64_t input_stride_j, int64_t input_stride_k,
    const float *extra,
    int64_t extra_stride_i, int64_t extra_stride_j, int64_t extra_stride_k,
    const float *coeff,
    int64_t coeff_stride_i, int64_t coeff_stride_j, int64_t coeff_stride_k,
    float *output,
    int64_t output_stride_i, int64_t output_stride_j, int64_t output_stride_k,
    float base_center, float base_extra, float coeff_extra,
    float coeff_center,
    float coeff_xm, float coeff_xp,
    float coeff_ym, float coeff_yp,
    float coeff_zm, float coeff_zp,
    void *cuda_stream);

/*
 * Flat seven-tensor ABI used by current raised Linalg lowering. `extra` and
 * `coeff` may be NULL. This mirrors the formula above but each tap is already
 * presented as a same-shaped tensor:
 *   out = base0*a0 + base_extra*extra
 *       + (coeff ? coeff[i] : 1) *
 *         (c0*a0 + ... + c6*a6 + coeff_extra*extra)
 */
void polygeist_custom_stencil3d_7pt_flat_f64(
    int32_t N,
    const double *a0, const double *a1, const double *a2,
    const double *a3, const double *a4, const double *a5,
    const double *a6, const double *extra, const double *coeff,
    double *out,
    double base0, double base_extra, double coeff_extra,
    double c0, double c1, double c2, double c3,
    double c4, double c5, double c6);

void polygeist_custom_stencil3d_7pt_flat_f32(
    int32_t N,
    const float *a0, const float *a1, const float *a2,
    const float *a3, const float *a4, const float *a5,
    const float *a6, const float *extra, const float *coeff,
    float *out,
    float base0, float base_extra, float coeff_extra,
    float c0, float c1, float c2, float c3,
    float c4, float c5, float c6);

#ifdef __cplusplus
}
#endif

#endif
