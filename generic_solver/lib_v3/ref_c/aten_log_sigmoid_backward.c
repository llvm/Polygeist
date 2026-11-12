/* Parameterized contiguous FP32 specialization of the pinned ATen scalar
 * log_sigmoid_backward_cpu lambda. TensorIterator traversal becomes a loop;
 * scalar_t=float, auto types become their exact C equivalents. No arithmetic
 * order changes. Source/licensing/provenance are recorded in the manifest. */
#ifndef NETLIB_SPEC
#error "NETLIB_SPEC must name the imported function"
#endif
void NETLIB_SPEC(int n,float input[],float buffer[],float grad[],float output[]){
 for(int i=0;i<n;++i){
int in_negative = input[i] < 0.0f;
        float max_deriv = in_negative ? 1.0f : 0.0f;
        float sign = in_negative ? 1.0f : -1.0f;
        output[i] = (max_deriv - sign * (buffer[i] / (1.0f + buffer[i]))) * grad[i];
 }
}
