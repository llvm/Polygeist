// std::nullptr_t values (a parameter of type decltype(nullptr), the nullptr
// literal passed to it) are null pointers. cgeist stopped with "unhandled
// type" on the type (auto_ntt fwd_load_inv_store_poly_1_limbs and
// fwd_store_inv_load_poly_1_limbs through the TAPA headers, W8b).
// RUN: cgeist %s %resourcedir --function='*' -S -O0 -o - | FileCheck %s

// CHECK-LABEL: func.func @_Z4pickDn(
// CHECK-SAME:    %{{.*}}: !llvm.ptr)
int pick(decltype(nullptr) p) { return p == nullptr ? 3 : 4; }

// CHECK-LABEL: func.func @_Z3topv(
// CHECK:         llvm.mlir.zero : !llvm.ptr
// CHECK:         call @_Z4pickDn(
int top() { return pick(nullptr); }
