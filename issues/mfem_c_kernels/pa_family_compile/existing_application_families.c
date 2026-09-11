/*
 * Compile-only standalone entry points already used by the larger MFEM
 * application extractions.  stage_kernels.h supplies their concrete FP64,
 * D1D=4, Q1D=5 implementations.  This translation unit deliberately has no
 * harness or main: the PA-family survey selects one numerical function at a
 * time with cgeist --function.
 */
#include "../application_extractions/stage_kernels.h"
