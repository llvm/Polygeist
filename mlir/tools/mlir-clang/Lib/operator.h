#ifndef PETMLIR_OPERATOR_H
#define PETMLIR_OPERATOR_H

namespace operators {

// isl ids are pointer-comparable.
bool operator==(const isl::id &left, const isl::id &right) {
  return left.get() == right.get();
}

} // end namespace operators.

#endif
