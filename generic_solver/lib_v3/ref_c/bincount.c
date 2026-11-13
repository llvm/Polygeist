/*
 * bincount.c -- faithful C transliteration of NumPy's bincount core loops.
 *
 * Source: numpy/numpy commit 9e2409c6bb6ee5814d139f436f93f587e03cf9ed,
 *         numpy/_core/src/multiarray/compiled_base.c, arr_bincount:
 *         counting loop L178-L188 (PyArray_ZEROS result, ians[numbers[i]] += 1),
 *         weighted loop L189-L213 (dans[numbers[i]] += weights[i]).
 *
 *   counts[b] = #{ i : numbers[i] == b }            (unweighted)
 *   counts[b] = sum { weights[i] : numbers[i] == b } (weighted)
 *
 * Transliteration rules (generic_solver/lib_v2/README.md):
 *   - only the counting kernels are kept; argument conversion, the min/max
 *     scan that sizes the result and the error checks are dropped, so the
 *     caller passes nbins (the reference's ans_size) and guarantees
 *     0 <= numbers[i] < nbins (the reference raises otherwise);
 *   - PyArray_ZEROS becomes an explicit zero-fill loop over the nbins counts;
 *   - npy_intp numbers/counts are instantiated as int (CUB counter type);
 *   - WEIGHTED selects the weighted loop (double weights and counts);
 *   - ZERO_FILL = 0 drops the zero-fill, i.e. the counts are accumulated
 *     into the caller's existing counts (a deviation, listed in the manifest).
 *
 * Licence of the source:
 *   Copyright (c) 2005-2025, NumPy Developers.
 *   All rights reserved.
 *
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions are
 *   met:
 *
 *       * Redistributions of source code must retain the above copyright
 *          notice, this list of conditions and the following disclaimer.
 *
 *       * Redistributions in binary form must reproduce the above
 *          copyright notice, this list of conditions and the following
 *          disclaimer in the documentation and/or other materials provided
 *          with the distribution.
 *
 *       * Neither the name of the NumPy Developers nor the names of any
 *          contributors may be used to endorse or promote products derived
 *          from this software without specific prior written permission.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Macros:
 *   NETLIB_SPEC  name of the emitted C function (= spec name)
 *   WEIGHTED     0: count occurrences, 1: sum weights
 *   ZERO_FILL    1: reference (zero-filled result), 0: accumulate
 */

#ifndef NETLIB_SPEC
#error "NETLIB_SPEC must name the specialised function"
#endif
#ifndef WEIGHTED
#error "WEIGHTED must be 0 or 1"
#endif
#ifndef ZERO_FILL
#error "ZERO_FILL must be 0 or 1"
#endif

#if WEIGHTED
void NETLIB_SPEC(const int len, const int numbers[], const double weights[],
                 const int nbins, double dans[]) {
#if ZERO_FILL
  for (int b = 0; b < nbins; b++)
    dans[b] = 0.0;
#else
  (void)nbins;
#endif
  /* L208-L209 */
  for (int i = 0; i < len; i++)
    dans[numbers[i]] += weights[i];
}
#else
void NETLIB_SPEC(const int len, const int numbers[], const int nbins,
                 int ians[]) {
#if ZERO_FILL
  for (int b = 0; b < nbins; b++)
    ians[b] = 0;
#else
  (void)nbins;
#endif
  /* L185-L186 */
  for (int i = 0; i < len; i++)
    ians[numbers[i]] += 1;
}
#endif
