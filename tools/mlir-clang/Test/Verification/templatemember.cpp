// RUN: mlir-clang %s --function=* -S | FileCheck %s

class House;

template <typename T>
class Info;

template <>
class Info<House>{
public:
  static constexpr bool has_infinity = true;
};

bool add_kernel_cuda() {
  return Info<House>::has_infinity;
}

//  CHECK:   llvm.mlir.global external @_ZN4InfoI5HouseE12has_infinityE() : i8 {
//  CHECK-NEXT:     %c1_i8 = arith.constant 1 : i8
//  CHECK-NEXT:     llvm.return %c1_i8 : i8
//  CHECK-NEXT:   }
//  CHECK:   func @_Z15add_kernel_cudav() -> i8 attributes {llvm.linkage = #llvm.linkage<external>} {
//  CHECK-NEXT:     %0 = llvm.mlir.addressof @_ZN4InfoI5HouseE12has_infinityE : !llvm.ptr<i8>
//  CHECK-NEXT:     %1 = llvm.load %0 : !llvm.ptr<i8>
//  CHECK-NEXT:     return %1 : i8
//  CHECK-NEXT:   }
