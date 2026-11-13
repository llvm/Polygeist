/* Pinned CUB seeded ThreadReduce, sequential scalar implementation.
 * Inline its prefix-array construction; preserve scalar order. Static length
 * becomes a runtime extent. The parallel DeviceReduce backend is separately
 * validated within a declared tolerance; bitwise parallel equivalence is not
 * implied. SCALAR and REDUCE are template specializations from the manifest. */
#ifndef NETLIB_SPEC
#error "NETLIB_SPEC must name the imported function"
#endif
#if REDUCE_PRODUCT
#define REDUCE(a,b) ((a) * (b))
#else
#define REDUCE(a,b) ((a) + (b))
#endif
SCALAR NETLIB_SPEC(int n, SCALAR seed, SCALAR input[])
{
  SCALAR retval = seed;
  for (int i = 0; i < n; ++i)
  {
    retval = REDUCE(retval, input[i]);
  }
  return retval;
}
