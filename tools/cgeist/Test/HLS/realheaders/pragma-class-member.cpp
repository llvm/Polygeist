// A pragma naming a data member inside an instance member function is about
// `this->member` (Vitis parses `variable=` as an expression of the function:
// hls-llvm-project Parser::ParseHLSVariableExpression). An array_partition /
// array_reshape keeps the member path (`this.m_Val`, member_path) on the
// constructor's `this` and is carried to the object when the constructor is
// inlined (Vitis_Libraries blas types.hpp `WindowRm::m_Val`, security aes.hpp
// `key_list`). Any other pragma on a member is reported and dropped. CIRCT
// OPEN_ISSUES O219 C1.
// RUN: cgeist %s %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - 2>/dev/null | FileCheck %s
// RUN: cgeist %s %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o /dev/null 2>&1 | FileCheck %s --check-prefix=WARN

// CHECK: memref.alloca() {hls.array_partition = [{dim = 1 : i32, factor = 4 : i32, kind = "cyclic", member_path = [0 : i32], variable = "this.m_Val"}], polygeist.varname = "w"}
// WARN: warning: 'm_Buf' in '#pragma HLS bind_storage' names a data member of the object; the pragma is ignored
template <int N> struct Win {
  int m_Val[N];
  int m_Buf[N];
  Win() {
#pragma HLS array_partition variable = m_Val cyclic factor = 4
#pragma HLS bind_storage variable = m_Buf type = ram_2p impl = lutram
  }
};
void top(const int in[2048], int out[512]) {
  Win<2048> w;
  for (int i = 0; i < 2048; i++) { w.m_Val[i] = in[i]; w.m_Buf[i] = in[i]; }
  for (int i = 0; i < 512; i++)
    out[i] = w.m_Val[4 * i] + w.m_Val[4 * i + 1] + w.m_Buf[4 * i];
}
