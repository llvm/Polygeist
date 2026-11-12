/* C specialization of pinned CUB ThreadScanInclusive with explicit seed.
 * T=float, ScanOp is plus<float> or multiplies<float>, LENGTH is a runtime extent.
 * The unused returned aggregate is discarded; each output and operation order
 * are preserved. DeviceScan::InclusiveScanInit is the corresponding API;
 * its parallel FP association is separately checked within declared tolerance. */
#ifndef NETLIB_SPEC
#error "NETLIB_SPEC must name the imported function"
#endif
#if SCAN_PRODUCT
#define SCAN_OP *
#else
#define SCAN_OP +
#endif
void NETLIB_SPEC(int n, float inclusive, float input[], float output[]) {

  for (int i = 0; i < n; ++i)
  {
    inclusive = inclusive SCAN_OP input[i];
    output[i] = inclusive;
  }

}
