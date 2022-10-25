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

// CHECK:   func.func @_Z15div_kernel_cudaR26ASmallVectorTemplateCommonI12AOperandInfoE(%[[arg0:.+]]: memref<?x2xmemref<?xi8>>) -> i64
// CHECK-NEXT:     %[[V0:.+]] = affine.load %[[arg0]][0, 1] : memref<?x2xmemref<?xi8>>
// CHECK-NEXT:     %[[V1:.+]] = "polygeist.memref2pointer"(%[[V0]]) : (memref<?xi8>) -> !llvm.ptr<i8>
// CHECK-NEXT:     %[[V2:.+]] = call @_ZNK26ASmallVectorTemplateCommonI12AOperandInfoE5beginEv(%[[arg0]]) : (memref<?x2xmemref<?xi8>>) -> memref<?x!llvm.struct<(memref<?xi8>, i8, i8)>>
// CHECK-NEXT:     %[[V3:.+]] = llvm.bitcast %[[V1]] : !llvm.ptr<i8> to !llvm.ptr<!llvm.struct<(memref<?xi8>, i8, i8)>>
// CHECK-NEXT:     %[[V4:.+]] = "polygeist.memref2pointer"(%[[V2]]) : (memref<?x!llvm.struct<(memref<?xi8>, i8, i8)>>) -> !llvm.ptr<!llvm.struct<(memref<?xi8>, i8, i8)>>
// CHECK-DAG:     %[[i5:.+]] = llvm.ptrtoint %[[V4]] : !llvm.ptr<!llvm.struct<(memref<?xi8>, i8, i8)>> to i64
// CHECK-DAG:     %[[i6:.+]] = llvm.ptrtoint %[[V3]] : !llvm.ptr<!llvm.struct<(memref<?xi8>, i8, i8)>> to i64
// CHECK-NEXT:     %[[V7:.+]] = arith.subi %[[i6]], %[[i5]] : i64
// CHECK-NEXT:     %[[V8:.+]] = "polygeist.typeSize"() {source = !llvm.struct<(memref<?xi8>, i8, i8)>} : () -> index
// CHECK-NEXT:     %[[V9:.+]] = arith.index_cast %[[V8]] : index to i64
// CHECK-NEXT:     %[[V10:.+]] = arith.divsi %[[V7]], %[[V9]] : i64
// CHECK-NEXT:     return %[[V10]] : i64
// CHECK-NEXT:   }
// CHECK:   func.func @_ZNK26ASmallVectorTemplateCommonI12AOperandInfoE5beginEv(%[[arg0:.+]]: memref<?x2xmemref<?xi8>>) -> memref<?x!llvm.struct<(memref<?xi8>, i8, i8)>>
// CHECK-NEXT:     %[[V0:.+]] = affine.load %[[arg0]][0, 0] : memref<?x2xmemref<?xi8>>
// CHECK-NEXT:     %[[V1:.+]] = "polygeist.memref2pointer"(%[[V0]]) : (memref<?xi8>) -> !llvm.ptr<i8>
// CHECK-NEXT:     %[[V2:.+]] = "polygeist.pointer2memref"(%[[V1]]) : (!llvm.ptr<i8>) -> memref<?x!llvm.struct<(memref<?xi8>, i8, i8)>>
// CHECK-NEXT:     return %[[V2]] : memref<?x!llvm.struct<(memref<?xi8>, i8, i8)>>
// CHECK-NEXT:   }
