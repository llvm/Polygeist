// A braced aggregate initialiser that leaves a member with a braced default
// member initialiser (`bool r{false}`) to its default: the member's
// initialiser is an InitListExpr of a scalar, which hit "unknown inner type"
// (OPEN_ISSUES O152 in the hls-est tree, reduced from FocusedACO three-opt
// `Segment seg[3] = {{z_1, x, len, 0}, ...}`). It is the scalar's one element.
// RUN: cgeist %s %stdinclude --function='*' -S -o - | FileCheck %s

struct P {
  int x{0};
  bool r{false};
};

int top(int a) {
  P p = {a};
  return p.x + p.r;
}

// CHECK-LABEL: func.func @_Z3topi
// CHECK: return
