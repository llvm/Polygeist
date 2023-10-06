// RUN: polygeist-opt --convert-polygeist-to-llvm --split-input-file %s --allow-unregistered-dialect | FileCheck %s

module {
  func.func private @wow()
  func.func private @foo(%40 : memref<?xi8>) {
    %41 = "polygeist.stream2token"(%40) : (memref<?xi8>) -> !async.token
    %c1 = arith.constant 1 : index
    %token = async.execute [%41] {
      %c10 = arith.constant 10 : index
      scf.for %arg2 = %c1 to %c1 step %c1 {
        func.call @wow() : () -> ()
      }
      async.yield
    }
    return
  }
  llvm.func @_Z3runP11CUstream_stPii(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i32) {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.mlir.constant(20 : index) : i64
    %3 = llvm.mlir.constant(10 : index) : i64
    %4 = llvm.bitcast %arg0 : !llvm.ptr to !llvm.ptr
    %5 = llvm.bitcast %4 : !llvm.ptr to !llvm.ptr
    %6 = builtin.unrealized_conversion_cast %5 : !llvm.ptr to memref<?xi8>
    %7 = "polygeist.stream2token"(%6) : (memref<?xi8>) -> !async.token
    %token = async.execute [%7] {
      omp.parallel   {
        omp.wsloop   for  (%arg3, %arg4) : i64 = (%0, %0) to (%3, %2) step (%1, %1) {
          llvm.call @_Z9somethingPii(%arg1, %arg2) : (!llvm.ptr, i32) -> ()
          omp.yield
        }
        omp.terminator
      }
      async.yield
    }
    llvm.return
  }
  llvm.func @_Z9somethingPii(!llvm.ptr, i32) attributes {sym_visibility = "private"}
}

// CHECK-LABEL:   llvm.func @foo(
// CHECK-SAME:                   %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr) attributes {sym_visibility = "private"} {
// CHECK:           %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.mlir.addressof @kernelbody.{{[0-9\.]+}} : !llvm.ptr<func<void (ptr)>>
// CHECK:           %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.bitcast %[[VAL_2]] : !llvm.ptr<func<void (ptr)>> to !llvm.ptr
// CHECK:           llvm.call @fake_cuda_dispatch(%[[VAL_1]], %[[VAL_3]], %[[VAL_0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK:           llvm.return
// CHECK:         }

// CHECK-LABEL:   llvm.func @_Z3runP11CUstream_stPii(
// CHECK-SAME:                                       %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr,
// CHECK-SAME:                                       %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr,
// CHECK-SAME:                                       %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.mlir.constant(16 : i64) : i64
// CHECK:           %[[VAL_4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.call @malloc(%[[VAL_3]]) : (i64) -> !llvm.ptr
// CHECK:           %[[VAL_5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.getelementptr %[[VAL_4]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
// CHECK:           llvm.store %[[VAL_1]], %[[VAL_5]] : !llvm.ptr, !llvm.ptr
// CHECK:           %[[VAL_6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.getelementptr %[[VAL_4]][0, 1] : (!llvm.ptr) -> !llvm.ptr, i32
// CHECK:           llvm.store %[[VAL_2]], %[[VAL_6]] : i32, !llvm.ptr
// CHECK:           %[[VAL_7:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.mlir.addressof @kernelbody.{{[0-9\.]+}} : !llvm.ptr<func<void (ptr)>>
// CHECK:           %[[VAL_8:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.bitcast %[[VAL_7]] : !llvm.ptr<func<void (ptr)>> to !llvm.ptr
// CHECK:           llvm.call @fake_cuda_dispatch(%[[VAL_4]], %[[VAL_8]], %[[VAL_0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK:           llvm.return
// CHECK:         }

// CHECK-LABEL:   llvm.func @kernelbody.{{[0-9\.]+}}(
// CHECK-SAME:                                            %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr) {
// CHECK:           %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           llvm.br ^bb1
// CHECK:         ^bb1:
// CHECK:           llvm.br ^bb2(%[[VAL_1]] : i64)
// CHECK:         ^bb2(%[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64):
// CHECK:           %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.icmp "slt" %[[VAL_2]], %[[VAL_1]] : i64
// CHECK:           llvm.cond_br %[[VAL_3]], ^bb3, ^bb4
// CHECK:         ^bb3:
// CHECK:           llvm.call @wow() : () -> ()
// CHECK:           %[[VAL_4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.add %[[VAL_2]], %[[VAL_1]]  : i64
// CHECK:           llvm.br ^bb2(%[[VAL_4]] : i64)
// CHECK:         ^bb4:
// CHECK:           llvm.return
// CHECK:         }

// CHECK:         llvm.func @fake_cuda_dispatch(!llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}

// CHECK-LABEL:   llvm.func @kernelbody.{{[0-9\.]+}}(
// CHECK-SAME:                                            %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr) {
// CHECK:           %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.mlir.constant(10 : index) : i64
// CHECK:           %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.mlir.constant(20 : index) : i64
// CHECK:           %[[VAL_4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           %[[VAL_5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.getelementptr %[[VAL_0]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
// CHECK:           %[[VAL_6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.load %[[VAL_5]] : !llvm.ptr -> !llvm.ptr
// CHECK:           %[[VAL_7:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.getelementptr %[[VAL_0]][0, 1] : (!llvm.ptr) -> !llvm.ptr, i32
// CHECK:           %[[VAL_8:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.load %[[VAL_7]] : !llvm.ptr -> i32
// CHECK:           llvm.call @free(%[[VAL_0]]) : (!llvm.ptr) -> ()
// CHECK:           llvm.br ^bb1
// CHECK:         ^bb1:
// CHECK:           omp.parallel   {
// CHECK:             omp.wsloop   for  (%[[VAL_9:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]], %[[VAL_10:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]) : i64 = (%[[VAL_1]], %[[VAL_1]]) to (%[[VAL_2]], %[[VAL_3]]) step (%[[VAL_4]], %[[VAL_4]]) {
// CHECK:               llvm.call @_Z9somethingPii(%[[VAL_6]], %[[VAL_8]]) : (!llvm.ptr, i32) -> ()
// CHECK:               omp.yield
// CHECK:             }
// CHECK:             omp.terminator
// CHECK:           }
// CHECK:           llvm.return
// CHECK:         }

