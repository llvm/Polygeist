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

// CHECK:   func @lt_kernel_cuda(%arg0: memref<?x!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-DAG:     %c0_i8 = arith.constant 0 : i8
// CHECK-NEXT:     %0 = memref.alloca() : memref<1x1xmemref<?x!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>>
// CHECK-NEXT:     %1 = memref.cast %0 : memref<1x1xmemref<?x!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>> to memref<?x1xmemref<?x!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>>
// CHECK-NEXT:     %2 = call @_ZNK15MTensorIterator11input_dtypeEv(%arg0) : (memref<?x!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>) -> i8
// CHECK-NEXT:     %3 = arith.cmpi ne, %2, %c0_i8 : i8
// CHECK-NEXT:     scf.if %3 {
// CHECK-NEXT:     affine.store %arg0, %0[0, 0] : memref<1x1xmemref<?x!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>>
// CHECK-NEXT:       func.call @_ZZ14lt_kernel_cudaENK3$_0clEv(%1) : (memref<?x1xmemref<?x!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func @_ZNK15MTensorIterator11input_dtypeEv(%arg0: memref<?x!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>) -> i8 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-DAG:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %0 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>) -> !llvm.ptr<!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>
// CHECK-NEXT:     %1 = "polygeist.pointer2memref"(%0) : (!llvm.ptr<!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>) -> memref<?x1xmemref<?x2xi8>>
// CHECK-NEXT:     %2 = call @_ZNK12MSmallVectorI12MOperandInfoEixEi(%1, %c0_i32) : (memref<?x1xmemref<?x2xi8>>, i32) -> memref<?x2xi8>
// CHECK-NEXT:     %3 = affine.load %2[0, 1] : memref<?x2xi8>
// CHECK-NEXT:     return %3 : i8
// CHECK-NEXT:   }
// CHECK-NEXT:   func private @_ZZ14lt_kernel_cudaENK3$_0clEv(%arg0: memref<?x1xmemref<?x!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>>) attributes {llvm.linkage = #llvm.linkage<internal>} {
// CHECK-NEXT:     %0 = affine.load %arg0[0, 0] : memref<?x1xmemref<?x!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>>
// CHECK-NEXT:     %1 = call @_ZNK15MTensorIterator6deviceEv(%0) : (memref<?x!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>) -> i8
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func @_ZNK12MSmallVectorI12MOperandInfoEixEi(%arg0: memref<?x1xmemref<?x2xi8>>, %arg1: i32) -> memref<?x2xi8> attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:     %0 = affine.load %arg0[0, 0] : memref<?x1xmemref<?x2xi8>>
// CHECK-NEXT:     %1 = arith.index_cast %arg1 : i32 to index
// CHECK-NEXT:     %2 = "polygeist.subindex"(%0, %1) : (memref<?x2xi8>, index) -> memref<?x2xi8>
// CHECK-NEXT:     return %2 : memref<?x2xi8>
// CHECK-NEXT:   }
// CHECK:   func @_ZNK15MTensorIterator6deviceEv(%arg0: memref<?x!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>) -> i8 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-DAG:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %0 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>) -> !llvm.ptr<!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>
// CHECK-NEXT:     %1 = "polygeist.pointer2memref"(%0) : (!llvm.ptr<!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>) -> memref<?x1xmemref<?x2xi8>>
// CHECK-NEXT:     %2 = call @_ZNK12MSmallVectorI12MOperandInfoEixEi(%1, %c0_i32) : (memref<?x1xmemref<?x2xi8>>, i32) -> memref<?x2xi8>
// CHECK-NEXT:     %3 = affine.load %2[0, 0] : memref<?x2xi8>
// CHECK-NEXT:     return %3 : i8
// CHECK-NEXT:   }
