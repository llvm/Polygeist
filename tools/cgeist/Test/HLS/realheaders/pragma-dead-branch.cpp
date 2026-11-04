// The pragmas in an arm of an if whose condition is a constant (a template
// argument) that the arm does not take are not emitted, as clang's
// CodeGenFunction::EmitIfStmt skips the dead arm: only the live arm's
// bind_storage applies, no conflict (Vitis_Libraries vision
// `if (USE_URAM) { ... impl=URAM } else { ... impl=BRAM }`, xf_box_filter,
// xf_demosaicing, xf_remap, xf_3dlut). CIRCT OPEN_ISSUES O219 C4.
// RUN: cgeist %s %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s

// CHECK: hls.bind_storage = [{impl = "lutram", type = "ram_s2p", variable = "buf"}]
// CHECK-NOT: impl = "uram"
template <int USE_URAM> void work(const int in[1024], int out[1024]) {
  int buf[1024];
  if (USE_URAM) {
#pragma HLS bind_storage variable = buf type = ram_s2p impl = uram
  } else {
#pragma HLS bind_storage variable = buf type = ram_s2p impl = lutram
  }
  for (int i = 0; i < 1024; i++) buf[i] = in[i];
  for (int i = 0; i < 1024; i++) out[i] = buf[1023 - i];
}
void top(const int in[1024], int out[1024]) { work<0>(in, out); }
