// Integer pragma values that name a static constexpr member brought in by a
// using-declaration from a base class template (Vitis_Libraries vision
// xf_structs.hpp `COLS_ADDRBOUND`), a member of a type template parameter
// (`#pragma HLS PIPELINE II = CholeskyTraits::INNER_II`, solver cholesky.hpp,
// qrf.hpp, svd.hpp) and a member of a typedef of one resolve like C++ name
// lookup. CIRCT OPEN_ISSUES O219 C2.
// RUN: cgeist %s %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s

// CHECK-DAG: hls.loop_tripcount = [{avg = 16 : i32, max = 32 : i32, min = 1 : i32}]
// CHECK-DAG: hls.pipeline = [{II = 2 : i32
template <int C> struct Base { static constexpr int B = C / 2; };
template <int C> struct Iter : public Base<C> {
  using Base<C>::B;
  static void run(const int *in, int *out, int n) {
    for (int i = 0; i < n; i++) {
#pragma HLS loop_tripcount min = 1 max = B
      out[i] = in[i] + 1;
    }
  }
};
struct Traits { static const int INNER_II = 2; static const int N = 64; };
template <typename T> void work(const int *in, int *out) {
  typedef T TT;
  for (int k = 0; k < TT::N; k++) {
#pragma HLS PIPELINE II = T::INNER_II
    out[k] = in[k] * 3;
  }
}
void top(const int in[64], int out[64], int n) {
  Iter<64>::run(in, out, n);
  work<Traits>(in, out);
}
