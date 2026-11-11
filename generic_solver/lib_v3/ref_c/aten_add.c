/* ATen native/ufunc/add.h: add<float>(self, other, alpha), alpha=1.
 * The scalar ufunc is applied to one contiguous row, as TensorIterator does.
 * No arithmetic/type changes. Upstream path/hash are pinned in the manifest.
 */
#ifndef NETLIB_SPEC
#error "NETLIB_SPEC must name the imported function"
#endif
void NETLIB_SPEC(int n, float x[], float y[], float output[]) {
  for (int i = 0; i < n; ++i) output[i] = x[i] + 1.0f * y[i];
}
