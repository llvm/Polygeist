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
