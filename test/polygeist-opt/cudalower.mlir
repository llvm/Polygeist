// RUN: polygeist-opt --parallel-lower --split-input-file %s | FileCheck %s

module attributes {llvm.data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64", llvm.target_triple = "nvptx64-nvidia-cuda"}  {
  llvm.func @cudaMemcpy(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i32) -> i32
  func @_Z1aPiS_(%arg0: memref<?xi32>, %arg1: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %c64_i64 = arith.constant 64 : i64
    %0 = "polygeist.memref2pointer"(%arg0) : (memref<?xi32>) -> !llvm.ptr<i8>
    %1 = "polygeist.memref2pointer"(%arg1) : (memref<?xi32>) -> !llvm.ptr<i8>
    %2 = llvm.call @cudaMemcpy(%0, %1, %c64_i64, %c1_i32) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i32) -> i32
    return %2 : i32
  }
}

// CHECK:   func @_Z1aPiS_(%arg0: memref<?xi32>, %arg1: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-DAG:     %c64_i64 = arith.constant 64 : i64
// CHECK-DAG:     %false = arith.constant false
// CHECK-DAG:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %0 = "polygeist.memref2pointer"(%arg0) : (memref<?xi32>) -> !llvm.ptr<i8>
// CHECK-NEXT:     %1 = "polygeist.memref2pointer"(%arg1) : (memref<?xi32>) -> !llvm.ptr<i8>
// CHECK-NEXT:     "llvm.intr.memcpy"(%0, %1, %c64_i64, %false) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
// CHECK-NEXT:     return %c0_i32 : i32
// CHECK-NEXT:   }

// -----

module {
  func private @S(%arg0: i8, %arg1: !llvm.ptr<i8>) -> i8 {
    cf.switch %arg0 : i8, [
      default: ^bb10(%arg0 : i8),
      0: ^bb1
    ]
  ^bb1:  // 2 preds: ^bb0, ^bb0
    %6 = llvm.load %arg1 : !llvm.ptr<i8>
    cf.br ^bb10(%6 : i8)
  ^bb10(%50: i8):  // 10 preds: ^bb0, ^bb1, ^bb2, ^bb3, ^bb4, ^bb5, ^bb6, ^bb7, ^bb8, ^bb9
    return %50 : i8
  }
  func @meta(%arg2: !llvm.ptr<i8>, %arg3: i8) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    gpu.launch blocks(%arg4, %arg5, %arg6) in (%arg10 = %c2, %arg11 = %c1, %arg12 = %c1) threads(%arg7, %arg8, %arg9) in (%arg13 = %c1, %arg14 = %c1, %arg15 = %c1) {
      call @S(%arg3, %arg2) : (i8, !llvm.ptr<i8>) -> (i8)
      gpu.terminator
    }
    return
  }
}
// CHECK:   func @meta(%arg0: !llvm.ptr<i8>, %arg1: i8) {
// CHECK-DAG:     %c1 = arith.constant 1 : index
// CHECK-DAG:     %c2 = arith.constant 2 : index
// CHECK-DAG:     %c0 = arith.constant 0 : index
// CHECK-NEXT:     scf.parallel (%arg2, %arg3, %arg4) = (%c0, %c0, %c0) to (%c2, %c1, %c1) step (%c1, %c1, %c1) {
// CHECK-NEXT:       scf.parallel (%arg5, %arg6, %arg7) = (%c0, %c0, %c0) to (%c1, %c1, %c1) step (%c1, %c1, %c1) {
// CHECK-NEXT:         %0 = memref.alloca_scope -> (i8) {
// CHECK-NEXT:         %1 = scf.execute_region -> i8 {
// CHECK-NEXT:           cf.switch %arg1 : i8, [
// CHECK-NEXT:             default: ^bb2(%arg1 : i8),
// CHECK-NEXT:             0: ^bb1
// CHECK-NEXT:           ]
// CHECK-NEXT:         ^bb1:  // pred: ^bb0
// CHECK-NEXT:           %2 = llvm.load %arg0 : !llvm.ptr<i8>
// CHECK-NEXT:           cf.br ^bb2(%2 : i8)
// CHECK-NEXT:         ^bb2(%3: i8):  // 2 preds: ^bb0, ^bb1
// CHECK-NEXT:           cf.br ^bb3(%3 : i8)
// CHECK-NEXT:         ^bb3(%4: i8):  // pred: ^bb2
// CHECK-NEXT:           scf.yield %4 : i8
// CHECK-NEXT:         }
// CHECK-NEXT:         memref.alloca_scope.return %1 : i8
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.yield
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// -----

module {
  func private @somethingA() -> () 
  func private @somethingB() -> ()
  func private @S(%arg0: i1) {
    call @somethingA() : () -> ()
    scf.if %arg0 {
        nvvm.barrier0
    }
    call @somethingB() : () -> ()
    return 
  }
  func @meta(%arg: i1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    gpu.launch blocks(%arg4, %arg5, %arg6) in (%arg10 = %c2, %arg11 = %c1, %arg12 = %c1) threads(%arg7, %arg8, %arg9) in (%arg13 = %c2, %arg14 = %c1, %arg15 = %c1) {
      call @S(%arg) : (i1) -> ()
      gpu.terminator
    }
    return
  }
}

// CHECK:   func @meta(%arg0: i1) {
// CHECK-DAG:     %c1 = arith.constant 1 : index
// CHECK-DAG:     %c2 = arith.constant 2 : index
// CHECK-DAG:     %c0 = arith.constant 0 : index
// CHECK-NEXT:     scf.parallel (%arg1, %arg2, %arg3) = (%c0, %c0, %c0) to (%c2, %c1, %c1) step (%c1, %c1, %c1) {
// CHECK-NEXT:       scf.parallel (%arg4, %arg5, %arg6) = (%c0, %c0, %c0) to (%c2, %c1, %c1) step (%c1, %c1, %c1) {
// CHECK-NEXT:         memref.alloca_scope {
// CHECK-NEXT:         scf.execute_region {
// CHECK-NEXT:           call @somethingA() : () -> ()
// CHECK-NEXT:           scf.if %arg0 {
// CHECK-NEXT:             "polygeist.barrier"(%arg4, %arg5, %arg6) : (index, index, index) -> ()
// CHECK-NEXT:           }
// CHECK-NEXT:           call @somethingB() : () -> ()
// CHECK-NEXT:           scf.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.yield
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
