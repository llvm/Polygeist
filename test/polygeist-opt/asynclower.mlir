// RUN: polygeist-opt --convert-polygeist-to-llvm --split-input-file %s --allow-unregistered-dialect | FileCheck %s

module {
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

// CHECK:   llvm.func @_Z3runP11CUstream_stPii(%arg0: !llvm.ptr<struct<()>>, %arg1: !llvm.ptr<i32>, %arg2: i32) {
// CHECK-NEXT:     %0 = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:     %1 = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:     %2 = llvm.mlir.constant(20 : index) : i64
// CHECK-NEXT:     %3 = llvm.mlir.constant(10 : index) : i64
// CHECK-NEXT:     %4 = llvm.bitcast %arg0 : !llvm.ptr<struct<()>> to !llvm.ptr<i8>
// CHECK-NEXT:     %5 = llvm.bitcast %4 : !llvm.ptr<i8> to !llvm.ptr<i8>
// CHECK-NEXT:     %6 = builtin.unrealized_conversion_cast %5 : !llvm.ptr<i8> to memref<?xi8>
// CHECK-NEXT:     %7 = llvm.mlir.constant(16 : i64) : i64
// CHECK-NEXT:     %8 = llvm.call @malloc(%7) : (i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %9 = llvm.bitcast %8 : !llvm.ptr<i8> to !llvm.ptr<struct<(ptr<i32>, i32)>>
// CHECK-NEXT:     %10 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:     %11 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:     %12 = llvm.getelementptr %9[%10, 0] : (!llvm.ptr<struct<(ptr<i32>, i32)>>, i32) -> !llvm.ptr<ptr<i32>>
// CHECK-NEXT:     llvm.store %arg1, %12 : !llvm.ptr<ptr<i32>>
// CHECK-NEXT:     %13 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:     %14 = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:     %15 = llvm.getelementptr %9[%13, 1] : (!llvm.ptr<struct<(ptr<i32>, i32)>>, i32) -> !llvm.ptr<i32>
// CHECK-NEXT:     llvm.store %arg2, %15 : !llvm.ptr<i32>
// CHECK-NEXT:     %16 = llvm.bitcast %9 : !llvm.ptr<struct<(ptr<i32>, i32)>> to !llvm.ptr<i8>
// CHECK-NEXT:     %17 = llvm.mlir.addressof @kernelbody.{{[0-9\.]+}} : !llvm.ptr<func<void (ptr<i8>)>>
// CHECK-NEXT:     %18 = llvm.bitcast %5 : !llvm.ptr<i8> to !llvm.ptr<i8>
// CHECK-NEXT:     llvm.call @fake_cuda_dispatch(%16, %17, %18) : (!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>, !llvm.ptr<i8>) -> ()
// CHECK-NEXT:     llvm.return
// CHECK-NEXT:   }
// CHECK:   llvm.func @kernelbody.{{[0-9\.]+}}(%arg0: !llvm.ptr<i8>) {
// CHECK-NEXT:     %0 = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:     %1 = llvm.mlir.constant(10 : index) : i64
// CHECK-NEXT:     %2 = llvm.mlir.constant(20 : index) : i64
// CHECK-NEXT:     %3 = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:     %4 = llvm.bitcast %arg0 : !llvm.ptr<i8> to !llvm.ptr<struct<(ptr<i32>, i32)>>
// CHECK-NEXT:     %5 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:     %6 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:     %7 = llvm.getelementptr %4[%5, 0] : (!llvm.ptr<struct<(ptr<i32>, i32)>>, i32) -> !llvm.ptr<ptr<i32>>
// CHECK-NEXT:     %8 = llvm.load %7 : !llvm.ptr<ptr<i32>>
// CHECK-NEXT:     %9 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:     %10 = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:     %11 = llvm.getelementptr %4[%9, 1] : (!llvm.ptr<struct<(ptr<i32>, i32)>>, i32) -> !llvm.ptr<i32>
// CHECK-NEXT:     %12 = llvm.load %11 : !llvm.ptr<i32>
// CHECK-NEXT:     llvm.call @free(%arg0) : (!llvm.ptr<i8>) -> ()
// CHECK-NEXT:     omp.parallel   {
// CHECK-NEXT:       omp.wsloop   for  (%arg1, %arg2) : i64 = (%0, %0) to (%1, %2) step (%3, %3) {
// CHECK-NEXT:         llvm.call @_Z9somethingPii(%8, %12) : (!llvm.ptr<i32>, i32) -> ()
// CHECK-NEXT:         omp.yield
// CHECK-NEXT:       }
// CHECK-NEXT:       omp.terminator
// CHECK-NEXT:     }
// CHECK-NEXT:     llvm.return
// CHECK-NEXT:   }
