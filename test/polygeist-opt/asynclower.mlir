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
  llvm.func @_Z3runP11CUstream_stPii(%arg0: !llvm.ptr<struct<()>>, %arg1: !llvm.ptr<i32>, %arg2: i32) {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.mlir.constant(20 : index) : i64
    %3 = llvm.mlir.constant(10 : index) : i64
    %4 = llvm.bitcast %arg0 : !llvm.ptr<struct<()>> to !llvm.ptr<i8>
    %5 = llvm.bitcast %4 : !llvm.ptr<i8> to !llvm.ptr<i8>
    %6 = builtin.unrealized_conversion_cast %5 : !llvm.ptr<i8> to memref<?xi8>
    %7 = "polygeist.stream2token"(%6) : (memref<?xi8>) -> !async.token
    %token = async.execute [%7] {
      omp.parallel   {
        omp.wsloop   for  (%arg3, %arg4) : i64 = (%0, %0) to (%3, %2) step (%1, %1) {
          llvm.call @_Z9somethingPii(%arg1, %arg2) : (!llvm.ptr<i32>, i32) -> ()
          omp.yield
        }
        omp.terminator
      }
      async.yield
    }
    llvm.return
  }
  llvm.func @_Z9somethingPii(!llvm.ptr<i32>, i32) attributes {sym_visibility = "private"}
}

// CHECK-LABEL:   llvm.func @foo(
// CHECK-SAME:                   %[[VAL_0:.*]]: !llvm.ptr<i8>) attributes {sym_visibility = "private"} {
// CHECK:           %[[VAL_1:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : !llvm.ptr<i8> to memref<?xi8>
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.null : !llvm.ptr<i8>
// CHECK:           %[[VAL_4:.*]] = llvm.mlir.addressof @kernelbody.{{[0-9\.]+}} : !llvm.ptr<func<void (ptr<i8>)>>
// CHECK:           %[[VAL_5:.*]] = llvm.bitcast %[[VAL_0]] : !llvm.ptr<i8> to !llvm.ptr<i8>
// CHECK:           llvm.call @fake_cuda_dispatch(%[[VAL_3]], %[[VAL_4]], %[[VAL_5]]) : (!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>, !llvm.ptr<i8>) -> ()
// CHECK:           llvm.return

// CHECK-LABEL:   llvm.func @_Z3runP11CUstream_stPii(
// CHECK-SAME:                                       %[[VAL_0:.*]]: !llvm.ptr<struct<()>>,
// CHECK-SAME:                                       %[[VAL_1:.*]]: !llvm.ptr<i32>,
// CHECK-SAME:                                       %[[VAL_2:.*]]: i32) {
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           %[[VAL_4:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           %[[VAL_5:.*]] = llvm.mlir.constant(20 : index) : i64
// CHECK:           %[[VAL_6:.*]] = llvm.mlir.constant(10 : index) : i64
// CHECK:           %[[VAL_7:.*]] = llvm.bitcast %[[VAL_0]] : !llvm.ptr<struct<()>> to !llvm.ptr<i8>
// CHECK:           %[[VAL_8:.*]] = llvm.bitcast %[[VAL_7]] : !llvm.ptr<i8> to !llvm.ptr<i8>
// CHECK:           %[[VAL_9:.*]] = builtin.unrealized_conversion_cast %[[VAL_8]] : !llvm.ptr<i8> to memref<?xi8>
// CHECK:           %[[VAL_10:.*]] = llvm.mlir.constant(16 : i64) : i64
// CHECK:           %[[VAL_11:.*]] = llvm.call @malloc(%[[VAL_10]]) : (i64) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_12:.*]] = llvm.bitcast %[[VAL_11]] : !llvm.ptr<i8> to !llvm.ptr<struct<(ptr<i32>, i32)>>
// CHECK:           %[[VAL_13:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_14:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_15:.*]] = llvm.getelementptr %[[VAL_12]]{{\[}}%[[VAL_13]], 0] : (!llvm.ptr<struct<(ptr<i32>, i32)>>, i32) -> !llvm.ptr<ptr<i32>>
// CHECK:           llvm.store %[[VAL_1]], %[[VAL_15]] : !llvm.ptr<ptr<i32>>
// CHECK:           %[[VAL_16:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_17:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_18:.*]] = llvm.getelementptr %[[VAL_12]]{{\[}}%[[VAL_16]], 1] : (!llvm.ptr<struct<(ptr<i32>, i32)>>, i32) -> !llvm.ptr<i32>
// CHECK:           llvm.store %[[VAL_2]], %[[VAL_18]] : !llvm.ptr<i32>
// CHECK:           %[[VAL_19:.*]] = llvm.bitcast %[[VAL_12]] : !llvm.ptr<struct<(ptr<i32>, i32)>> to !llvm.ptr<i8>
// CHECK:           %[[VAL_20:.*]] = llvm.mlir.addressof @kernelbody.{{[0-9\.]+}} : !llvm.ptr<func<void (ptr<i8>)>>
// CHECK:           %[[VAL_21:.*]] = llvm.bitcast %[[VAL_8]] : !llvm.ptr<i8> to !llvm.ptr<i8>
// CHECK:           llvm.call @fake_cuda_dispatch(%[[VAL_19]], %[[VAL_20]], %[[VAL_21]]) : (!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>, !llvm.ptr<i8>) -> ()
// CHECK:           llvm.return

// CHECK-LABEL:   llvm.func @kernelbody.{{[0-9\.]+}}(
// CHECK-SAME:                                            %[[VAL_0:.*]]: !llvm.ptr<i8>) {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           llvm.br ^bb1
// CHECK:         ^bb1:
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(10 : index) : i64
// CHECK:           llvm.br ^bb2(%[[VAL_1]] : i64)
// CHECK:         ^bb2(%[[VAL_3:.*]]: i64):
// CHECK:           %[[VAL_4:.*]] = llvm.icmp "slt" %[[VAL_3]], %[[VAL_1]] : i64
// CHECK:           llvm.cond_br %[[VAL_4]], ^bb3, ^bb4
// CHECK:         ^bb3:
// CHECK:           llvm.call @wow() : () -> ()
// CHECK:           %[[VAL_5:.*]] = llvm.add %[[VAL_3]], %[[VAL_1]]  : i64
// CHECK:           llvm.br ^bb2(%[[VAL_5]] : i64)
// CHECK:         ^bb4:
// CHECK:           llvm.return
// CHECK:         }

// CHECK:         llvm.func @fake_cuda_dispatch(!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>, !llvm.ptr<i8>) attributes {sym_visibility = "private"}

// CHECK-LABEL:   llvm.func @kernelbody.{{[0-9\.]+}}(
// CHECK-SAME:                                            %[[VAL_0:.*]]: !llvm.ptr<i8>) {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(10 : index) : i64
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(20 : index) : i64
// CHECK:           %[[VAL_4:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           %[[VAL_5:.*]] = llvm.bitcast %[[VAL_0]] : !llvm.ptr<i8> to !llvm.ptr<struct<(ptr<i32>, i32)>>
// CHECK:           %[[VAL_6:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_7:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_8:.*]] = llvm.getelementptr %[[VAL_5]]{{\[}}%[[VAL_6]], 0] : (!llvm.ptr<struct<(ptr<i32>, i32)>>, i32) -> !llvm.ptr<ptr<i32>>
// CHECK:           %[[VAL_9:.*]] = llvm.load %[[VAL_8]] : !llvm.ptr<ptr<i32>>
// CHECK:           %[[VAL_10:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_11:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_12:.*]] = llvm.getelementptr %[[VAL_5]]{{\[}}%[[VAL_10]], 1] : (!llvm.ptr<struct<(ptr<i32>, i32)>>, i32) -> !llvm.ptr<i32>
// CHECK:           %[[VAL_13:.*]] = llvm.load %[[VAL_12]] : !llvm.ptr<i32>
// CHECK:           llvm.call @free(%[[VAL_0]]) : (!llvm.ptr<i8>) -> ()
// CHECK:           llvm.br ^bb1
// CHECK:         ^bb1:
// CHECK:           omp.parallel   {
// CHECK:             omp.wsloop   for  (%[[VAL_14:.*]], %[[VAL_15:.*]]) : i64 = (%[[VAL_1]], %[[VAL_1]]) to (%[[VAL_2]], %[[VAL_3]]) step (%[[VAL_4]], %[[VAL_4]]) {
// CHECK:               llvm.call @_Z9somethingPii(%[[VAL_9]], %[[VAL_13]]) : (!llvm.ptr<i32>, i32) -> ()
// CHECK:               omp.yield
// CHECK:             }
// CHECK:             omp.terminator
// CHECK:           }
// CHECK:           llvm.return
