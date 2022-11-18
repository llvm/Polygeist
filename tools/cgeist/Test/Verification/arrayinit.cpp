// RUN: cgeist %s --function='*' -S | FileCheck %s

struct IntDivider {
  int divisor;
};

struct OffsetCalculator {
  IntDivider sizes_[4];
};

OffsetCalculator foo(OffsetCalculator &o) {
    return o;
}
