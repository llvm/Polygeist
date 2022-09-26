// RUN: cgeist %s --function=* -S | FileCheck %s

struct AOperandInfo {
  void* data;

  bool is_output;

  bool is_read_write;
};


/// This is all the non-templated stuff common to all SmallVectors.

/// This is the part of SmallVectorTemplateBase which does not depend on whether
/// the type T is a POD. The extra dummy template argument is used by ArrayRef
/// to avoid unnecessarily requiring T to be complete.
template <typename T>
class ASmallVectorTemplateCommon {
 public:
  void *BeginX, *EndX;

  // forward iterator creation methods.
  const T* begin() const {
    return (const T*)this->BeginX;
  }
};

unsigned long long int div_kernel_cuda(ASmallVectorTemplateCommon<AOperandInfo> &operands) {
  return (const AOperandInfo*)operands.EndX - operands.begin();
}

// CHECK:  func.func
// @_Z15div_kernel_cudaR26ASmallVectorTemplateCommonI12AOperandInfoE(%arg0:
// memref<?x!llvm.struct<(ptr<i8>, ptr<i8>)>>) -> i64 attributes {llvm.linkage =
// #llvm.linkage<external>} { CHECK-DAG:    %c16_i64 = arith.constant 16 : i64
// CHECK-NEXT:    %0 = "polygeist.memref2pointer"(%arg0) :
// (memref<?x!llvm.struct<(ptr<i8>, ptr<i8>)>>) -> !llvm.ptr<struct<(ptr<i8>,
// ptr<i8>)>> CHECK-NEXT:    %1 = llvm.getelementptr %0[0, 1] :
// (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>) -> !llvm.ptr<ptr<i8>> CHECK-NEXT: %2
// = llvm.load %1 : !llvm.ptr<ptr<i8>> CHECK-NEXT:    %3 = call
// @_ZNK26ASmallVectorTemplateCommonI12AOperandInfoE5beginEv(%arg0) :
// (memref<?x!llvm.struct<(ptr<i8>, ptr<i8>)>>) ->
// memref<?x!llvm.struct<(ptr<i8>, i8, i8)>> CHECK-NEXT:    %4 = llvm.bitcast %2
// : !llvm.ptr<i8> to !llvm.ptr<struct<(ptr<i8>, i8, i8)>> CHECK-NEXT:    %5 =
// "polygeist.memref2pointer"(%3) : (memref<?x!llvm.struct<(ptr<i8>, i8, i8)>>)
// -> !llvm.ptr<struct<(ptr<i8>, i8, i8)>> CHECK-NEXT:    %6 = llvm.ptrtoint %5
// : !llvm.ptr<struct<(ptr<i8>, i8, i8)>> to i64 CHECK-NEXT:    %7 =
// llvm.ptrtoint %4 : !llvm.ptr<struct<(ptr<i8>, i8, i8)>> to i64 CHECK-NEXT: %8
// = arith.subi %7, %6 : i64 CHECK-NEXT:    %9 = arith.divsi %8, %c16_i64 : i64
// CHECK-NEXT:    return %9 : i64
// CHECK-NEXT:  }
// CHECK:  func.func
// @_ZNK26ASmallVectorTemplateCommonI12AOperandInfoE5beginEv(%arg0:
// memref<?x!llvm.struct<(ptr<i8>, ptr<i8>)>>) ->
// memref<?x!llvm.struct<(ptr<i8>, i8, i8)>> attributes {llvm.linkage =
// #llvm.linkage<linkonce_odr>} { CHECK-NEXT:    %0 =
// "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(ptr<i8>,
// ptr<i8>)>>) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> CHECK-NEXT:    %1 =
// llvm.getelementptr %0[0, 0] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>) ->
// !llvm.ptr<ptr<i8>> CHECK-NEXT:    %2 = llvm.load %1 : !llvm.ptr<ptr<i8>>
// CHECK-NEXT:    %3 = "polygeist.pointer2memref"(%2) : (!llvm.ptr<i8>) ->
// memref<?x!llvm.struct<(ptr<i8>, i8, i8)>> CHECK-NEXT:    return %3 :
// memref<?x!llvm.struct<(ptr<i8>, i8, i8)>> CHECK-NEXT:  }
