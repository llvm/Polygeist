// RUN: cgeist %s --function=* -S | FileCheck %s

struct MOperandInfo {
  char device;
  char dtype;
};

template <typename T>
class MSmallVector {
 public:
  struct MOperandInfo *BeginX;

  const struct MOperandInfo& operator[](int idx) const {
    return BeginX[idx];
  }
};


struct MTensorIterator {
  char input_dtype() const { return operands_[0].dtype; }
  char device() const { return operands_[0].device; }
  MSmallVector<MOperandInfo> operands_;
};

template <typename func_t>
void igpu_kernel(MTensorIterator& iter, const func_t& f) {
  iter.device();
}

extern "C" {
void lt_kernel_cuda(MTensorIterator& iter) {
  if (iter.input_dtype()) {
    ([&]() { igpu_kernel(iter, []() -> bool { return false; }); })();
  }
}
}


// CHECK-LABEL:   func.func @lt_kernel_cuda(
// CHECK-SAME:                              %[[VAL_0:[A-Za-z0-9_]*]]: memref<?x!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>)
// CHECK:           %[[VAL_1:[A-Za-z0-9_]*]] = arith.constant 0 : i8
// CHECK:           %[[VAL_2:[A-Za-z0-9_]*]] = memref.alloca() : memref<1x1xmemref<?x!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>>
// CHECK:           %[[VAL_3:[A-Za-z0-9_]*]] = memref.cast %[[VAL_2]] : memref<1x1xmemref<?x!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>> to memref<?x1xmemref<?x!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>>
// CHECK:           %[[VAL_4:[A-Za-z0-9_]*]] = call @_ZNK15MTensorIterator11input_dtypeEv(%[[VAL_0]]) : (memref<?x!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>) -> i8
// CHECK:           %[[VAL_5:[A-Za-z0-9_]*]] = arith.cmpi ne, %[[VAL_4]], %[[VAL_1]] : i8
// CHECK:           scf.if %[[VAL_5]] {
// CHECK:             affine.store %[[VAL_0]], %[[VAL_2]][0, 0] : memref<1x1xmemref<?x!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>>
// CHECK:             func.call @_ZZ14lt_kernel_cudaENK3$_0clEv(%[[VAL_3]]) : (memref<?x1xmemref<?x!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>>) -> ()
// CHECK:           }
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @_ZNK15MTensorIterator11input_dtypeEv(
// CHECK-SAME:                                                    %[[VAL_0:[A-Za-z0-9_]*]]: memref<?x!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>) -> i8
// CHECK:           %[[VAL_1:[A-Za-z0-9_]*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_2:[A-Za-z0-9_]*]] = "polygeist.memref2pointer"(%[[VAL_0]]) : (memref<?x!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>) -> !llvm.ptr
// CHECK:           %[[VAL_3:[A-Za-z0-9_]*]] = "polygeist.pointer2memref"(%[[VAL_2]]) : (!llvm.ptr) -> memref<?x1xmemref<?x2xi8>>
// CHECK:           %[[VAL_4:[A-Za-z0-9_]*]] = call @_ZNK12MSmallVectorI12MOperandInfoEixEi(%[[VAL_3]], %[[VAL_1]]) : (memref<?x1xmemref<?x2xi8>>, i32) -> memref<?x2xi8>
// CHECK:           %[[VAL_5:[A-Za-z0-9_]*]] = affine.load %[[VAL_4]][0, 1] : memref<?x2xi8>
// CHECK:           return %[[VAL_5]] : i8
// CHECK:         }

// CHECK-LABEL:   func.func private @_ZZ14lt_kernel_cudaENK3$_0clEv(
// CHECK-SAME:                                                      %[[VAL_0:[A-Za-z0-9_]*]]: memref<?x1xmemref<?x!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>>)
// CHECK:           %[[VAL_1:[A-Za-z0-9_]*]] = affine.load %[[VAL_0]][0, 0] : memref<?x1xmemref<?x!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>>
// CHECK:           %[[VAL_2:[A-Za-z0-9_]*]] = call @_ZNK15MTensorIterator6deviceEv(%[[VAL_1]]) : (memref<?x!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>) -> i8
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @_ZNK12MSmallVectorI12MOperandInfoEixEi(
// CHECK-SAME:                                                      %[[VAL_0:[A-Za-z0-9_]*]]: memref<?x1xmemref<?x2xi8>>,
// CHECK-SAME:                                                      %[[VAL_1:[A-Za-z0-9_]*]]: i32) -> memref<?x2xi8>
// CHECK:           %[[VAL_2:[A-Za-z0-9_]*]] = affine.load %[[VAL_0]][0, 0] : memref<?x1xmemref<?x2xi8>>
// CHECK:           %[[VAL_3:[A-Za-z0-9_]*]] = arith.index_cast %[[VAL_1]] : i32 to index
// CHECK:           %[[VAL_4:[A-Za-z0-9_]*]] = "polygeist.subindex"(%[[VAL_2]], %[[VAL_3]]) : (memref<?x2xi8>, index) -> memref<?x2xi8>
// CHECK:           return %[[VAL_4]] : memref<?x2xi8>
// CHECK:         }

// CHECK-LABEL:   func.func @_ZNK15MTensorIterator6deviceEv(
// CHECK-SAME:                                              %[[VAL_0:[A-Za-z0-9_]*]]: memref<?x!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>) -> i8
// CHECK:           %[[VAL_1:[A-Za-z0-9_]*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_2:[A-Za-z0-9_]*]] = "polygeist.memref2pointer"(%[[VAL_0]]) : (memref<?x!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>) -> !llvm.ptr
// CHECK:           %[[VAL_3:[A-Za-z0-9_]*]] = "polygeist.pointer2memref"(%[[VAL_2]]) : (!llvm.ptr) -> memref<?x1xmemref<?x2xi8>>
// CHECK:           %[[VAL_4:[A-Za-z0-9_]*]] = call @_ZNK12MSmallVectorI12MOperandInfoEixEi(%[[VAL_3]], %[[VAL_1]]) : (memref<?x1xmemref<?x2xi8>>, i32) -> memref<?x2xi8>
// CHECK:           %[[VAL_5:[A-Za-z0-9_]*]] = affine.load %[[VAL_4]][0, 0] : memref<?x2xi8>
// CHECK:           return %[[VAL_5]] : i8
// CHECK:         }

