/* Source-derived FP32 Thrust binary transform specialization.
 * Inline sequential for_each over the zip iterator and the actual
 * cuda::std::divides<float> source operator. Pointer and constant iterators
 * become their exact C reads; CUDA execution is validated independently.
 * Public Apache-2.0/LLVM source files and hashes are pinned in the manifest. */
#ifndef NETLIB_SPEC
#error "NETLIB_SPEC must name the imported function"
#endif
void NETLIB_SPEC(int n,
#if LEFT_SCALAR
float x,
#else
float x[],
#endif
#if RIGHT_SCALAR
float y,
#else
float y[],
#endif
float output[]) {
 for(int i=0;i<n;++i){
#if LEFT_SCALAR
  float left=x;
#else
  float left=x[i];
#endif
#if RIGHT_SCALAR
  float right=y;
#else
  float right=y[i];
#endif
  output[i] = left / right;
 }
}
