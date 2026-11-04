// --xlx-headers: a scalar wrapper constructed from a string literal through a
// constructor that passes the string to _ssdm_string2bits (the shape of the
// Vitis HLS headers' ap_fixed_base(const char*), etc/ap_fixed_base.h:458-471)
// is folded at compile time (XlxStringBits.cc). The declarations below are
// this test's own stand-ins, not the Vitis headers. Expected bits: the Vitis
// C simulation of ap_fixed<16,5> / ap_ufixed<8,4,AP_RND,AP_SAT> /
// ap_fixed<12,3,AP_RND_CONV,AP_WRAP> from the same strings, as signed
// decimals. Without --xlx-headers the constructor stays a call.
// RUN: cgeist %s --xlx-headers --scalar-wrapper-abi --function='*' -S -o - | FileCheck %s
// RUN: cgeist %s --scalar-wrapper-abi --function='*' -S -o - | FileCheck %s --check-prefix=OFF

enum ap_q_mode { AP_RND, AP_RND_ZERO, AP_RND_MIN_INF, AP_RND_INF, AP_RND_CONV,
                 AP_TRN, AP_TRN_ZERO };
enum ap_o_mode { AP_SAT, AP_SAT_ZERO, AP_SAT_SYM, AP_WRAP, AP_WRAP_SM };
extern "C" void _ssdm_string2bits(...);

template <int W, int I, ap_q_mode Q = AP_TRN, ap_o_mode O = AP_WRAP>
struct sfx {
  signed _BitInt(W) V;
  sfx() {}
  sfx(const char *s) {
    signed _BitInt(W) t;
    _ssdm_string2bits((void *)(&t), (const char *)(s), 10, I, true, Q, O, 0,
                      false);
    V = t;
  }
};
template <int W, int I, ap_q_mode Q = AP_TRN, ap_o_mode O = AP_WRAP>
struct ufx {
  unsigned _BitInt(W) V;
  ufx() {}
  ufx(const char *s) {
    unsigned _BitInt(W) t;
    _ssdm_string2bits((void *)(&t), (const char *)(s), 10, I, false, Q, O, 0,
                      false);
    V = t;
  }
};
// A derived class whose constructor forwards the string to its base, as
// ap_fixed(const char*) : Base(s) does.
template <int W, int I, ap_q_mode Q = AP_TRN, ap_o_mode O = AP_WRAP>
struct dfx : sfx<W, I, Q, O> {
  dfx(const char *s) : sfx<W, I, Q, O>(s) {}
};

// CHECK-DAG: memref.global "private" @"tbl@static@{{.*}}" : memref<6xi16> = dense<[3072, -5120, 10752, 6433, 768, 25600]>
// CHECK-DAG: memref.global "private" @"utbl@static@{{.*}}" : memref<3xi8> = dense<[-1, 0, 32]>
// OFF-NOT: dense<[3072
// OFF: _ssdm_string2bits
extern "C" {
int tbl(int i) {
  const static sfx<16, 5> tbl[6] = {"1.5",   "-0x2.8p0", "0b101.01",
                                    "3.14159", "0x1.8p-2", "1.25e1"};
  return (int)tbl[i % 6].V;
}

int utbl(int i) {
  const static ufx<8, 4, AP_RND, AP_SAT> utbl[3] = {"100", "-1", "2.0"};
  return (int)utbl[i % 3].V;
}

// CHECK-LABEL: func.func @scal
// CHECK-NOT: call
// CHECK: arith.constant -1312 : i32
int scal() {
  dfx<12, 3, AP_RND_CONV, AP_WRAP> c = "-2.5615234375";
  return (int)c.V;
}
}
