// RUN: cgeist %s --function='*' -S | FileCheck %s
// TODO:
// XFAIL: *

struct Int {
  int divisor;
};

struct IntArray {
  Int sizes_[4];
};

IntArray foo(IntArray &o) {
    return o;
}
