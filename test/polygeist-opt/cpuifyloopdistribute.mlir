// RUN: polygeist-opt --cpuify="method=distribute.mincut" --split-input-file %s | FileCheck %s

module {
  func.func private @use(%arg0: i32)
  func.func private @usememref(%arg0: memref<i32>)
  func.func @trivial(%arg0: i32, %c : i1) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c9 = arith.constant 9 : index
    scf.parallel (%arg4) = (%c0) to (%c9) step (%c1) {
      %0 = arith.index_cast %arg4 : index to i32
      "polygeist.barrier"(%arg4) : (index) -> ()
      func.call @use(%0) : (i32) -> ()
      scf.yield
    }
    return
  }
  func.func @add_if_barrier(%arg: i1, %amem: memref<i32>, %bmem : memref<i32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c9 = arith.constant 9 : index
    %alloc = memref.alloca() : memref<i32>
    scf.parallel (%arg4) = (%c0) to (%c9) step (%c1) {
      %a = memref.load %amem[] : memref<i32>
      %b = memref.load %bmem[] : memref<i32>
      %mul = arith.muli %a, %b : i32
      func.call @use(%mul) : (i32) -> ()
      "polygeist.barrier"(%arg4) : (index) -> ()
      scf.if %arg {
        func.call @use(%mul) : (i32) -> ()
        "polygeist.barrier"(%arg4) : (index) -> ()
        func.call @use(%mul) : (i32) -> ()
        scf.yield
      }
      scf.yield
    }
    return
  }
  func.func @add_if_barrier_(%arg: i1, %amem: memref<i32>, %bmem : memref<i32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c9 = arith.constant 9 : index
    %alloc = memref.alloca() : memref<i32>
    scf.parallel (%arg4) = (%c0) to (%c9) step (%c1) {
      %a = memref.load %amem[] : memref<i32>
      %b = memref.load %bmem[] : memref<i32>
      %mul = arith.muli %a, %b : i32
      func.call @usememref(%amem) : (memref<i32>) -> ()
      func.call @use(%mul) : (i32) -> ()
      "polygeist.barrier"(%arg4) : (index) -> ()
      scf.if %arg {
        func.call @use(%mul) : (i32) -> ()
        "polygeist.barrier"(%arg4) : (index) -> ()
        func.call @use(%mul) : (i32) -> ()
        scf.yield
      }
      scf.yield
    }
    return
  }
  func.func @mincut_for_barrier(%amem: memref<i32>, %arg: i1, %bound: index) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %i1 = arith.constant 1 : i32
    %i2 = arith.constant 2 : i32
    %i3 = arith.constant 3 : i32
    %c9 = arith.constant 9 : index
    %alloc = memref.alloca() : memref<i32>
    scf.parallel (%arg4) = (%c0) to (%c9) step (%c1) {
      %a = memref.load %amem[] : memref<i32>
      %a1 = arith.addi %a, %i1 : i32
      %a2 = arith.addi %a, %i2 : i32
      %a3 = arith.addi %a, %i3 : i32
      func.call @use(%a) : (i32) -> ()
      "polygeist.barrier"(%arg4) : (index) -> ()
      scf.for %forarg = %c0 to %bound step %c1 {
        func.call @use(%a1) : (i32) -> ()
        func.call @use(%a2) : (i32) -> ()
        func.call @use(%a3) : (i32) -> ()
        "polygeist.barrier"(%arg4) : (index) -> ()
        func.call @use(%a1) : (i32) -> ()
        func.call @use(%a2) : (i32) -> ()
        scf.yield
      }
      scf.yield
    }
    return
  }
  func.func @mincut_if_barrier(%amem: memref<i32>, %arg : i1) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %i1 = arith.constant 1 : i32
    %i2 = arith.constant 2 : i32
    %i3 = arith.constant 3 : i32
    %c9 = arith.constant 9 : index
    %alloc = memref.alloca() : memref<i32>
    scf.parallel (%arg4) = (%c0) to (%c9) step (%c1) {
      %a = memref.load %amem[] : memref<i32>
      %a1 = arith.addi %a, %i1 : i32
      %a2 = arith.addi %a, %i2 : i32
      %a3 = arith.addi %a, %i3 : i32
      func.call @use(%a) : (i32) -> ()
      "polygeist.barrier"(%arg4) : (index) -> ()
      scf.if %arg {
        func.call @use(%a1) : (i32) -> ()
        func.call @use(%a2) : (i32) -> ()
        func.call @use(%a3) : (i32) -> ()
        "polygeist.barrier"(%arg4) : (index) -> ()
        func.call @use(%a1) : (i32) -> ()
        func.call @use(%a2) : (i32) -> ()
        func.call @use(%a3) : (i32) -> ()
        scf.yield
      }
      scf.yield
    }
    return
  }
  func.func @mincut(%amem: memref<i32>, %bmem : memref<i32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %i1 = arith.constant 1 : i32
    %i2 = arith.constant 2 : i32
    %i3 = arith.constant 3 : i32
    %c9 = arith.constant 9 : index
    %alloc = memref.alloca() : memref<i32>
    scf.parallel (%arg4) = (%c0) to (%c9) step (%c1) {
      %a = memref.load %amem[] : memref<i32>
      %a1 = arith.addi %a, %i1 : i32
      %a2 = arith.addi %a, %i2 : i32
      %a3 = arith.addi %a, %i3 : i32
      func.call @use(%a) : (i32) -> ()
      "polygeist.barrier"(%arg4) : (index) -> ()
      func.call @use(%a1) : (i32) -> ()
      func.call @use(%a2) : (i32) -> ()
      func.call @use(%a3) : (i32) -> ()
      scf.yield
    }
    return
  }
  func.func @add(%amem: memref<i32>, %bmem : memref<i32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c9 = arith.constant 9 : index
    %alloc = memref.alloca() : memref<i32>
    scf.parallel (%arg4) = (%c0) to (%c9) step (%c1) {
      %a = memref.load %amem[] : memref<i32>
      %b = memref.load %bmem[] : memref<i32>
      %mul = arith.muli %a, %b : i32
      func.call @use(%mul) : (i32) -> ()
      "polygeist.barrier"(%arg4) : (index) -> ()
      func.call @use(%mul) : (i32) -> ()
      scf.yield
    }
    return
  }
  func.func @matmul(%arg0: memref<?x3xi32>, %arg1: memref<?x3xi32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: i32, %arg6: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %0 = arith.index_cast %arg5 : i32 to index
    %1 = arith.index_cast %arg6 : i32 to index
    %2 = affine.load %arg0[0, 0] : memref<?x3xi32>
    %3 = affine.load %arg0[0, 1] : memref<?x3xi32>
    %4 = affine.load %arg0[0, 2] : memref<?x3xi32>
    %5 = arith.index_cast %2 : i32 to index
    %6 = arith.index_cast %3 : i32 to index
    %7 = arith.index_cast %4 : i32 to index
    %8 = affine.load %arg1[0, 0] : memref<?x3xi32>
    %9 = affine.load %arg1[0, 1] : memref<?x3xi32>
    %10 = affine.load %arg1[0, 2] : memref<?x3xi32>
    %11 = arith.index_cast %8 : i32 to index
    %12 = arith.index_cast %9 : i32 to index
    %13 = arith.index_cast %10 : i32 to index
    %14 = arith.divsi %0, %c2 : index
    %15 = arith.muli %1, %c2 : index
    scf.parallel (%arg7, %arg8, %arg9) = (%c0, %c0, %c0) to (%5, %6, %7) step (%c1, %c1, %c1) {
      %16 = memref.alloca() : memref<2x2xf32>
      %17 = memref.alloca() : memref<2x2xf32>
      %18 = arith.muli %arg8, %c2 : index
      %19 = arith.muli %18, %0 : index
      %20 = "polygeist.subindex"(%arg2, %19) : (memref<?xf32>, index) -> memref<?xf32>
      %21 = arith.muli %arg7, %c2 : index
      %22 = "polygeist.subindex"(%arg3, %21) : (memref<?xf32>, index) -> memref<?xf32>
      %23 = arith.muli %18, %1 : index
      %24 = arith.addi %21, %23 : index
      scf.parallel (%arg10, %arg11, %arg12) = (%c0, %c0, %c0) to (%11, %12, %13) step (%c1, %c1, %c1) {
        %25 = arith.muli %0, %arg11 : index
        %26 = arith.addi %25, %arg10 : index
        %27 = arith.muli %1, %arg11 : index
        %28 = arith.addi %27, %arg10 : index
        %29:4 = scf.for %arg13 = %c0 to %14 step %c1 iter_args(%arg14 = %cst, %arg15 = %22, %arg16 = %20, %arg17 = %cst) -> (f32, memref<?xf32>, memref<?xf32>, f32) {
          %33 = memref.load %arg16[%26] : memref<?xf32>
          memref.store %33, %16[%arg11, %arg10] : memref<2x2xf32>
          %34 = memref.load %arg15[%28] : memref<?xf32>
          memref.store %34, %17[%arg11, %arg10] : memref<2x2xf32>
          "polygeist.barrier"(%arg10, %arg11, %arg12) : (index, index, index) -> ()
          %35:2 = scf.for %arg18 = %c0 to %c2 step %c1 iter_args(%arg19 = %arg14, %arg20 = %arg14) -> (f32, f32) {
            %38 = memref.load %16[%arg11, %arg18] : memref<2x2xf32>
            %39 = memref.load %17[%arg18, %arg10] : memref<2x2xf32>
            %40 = arith.mulf %38, %39 : f32
            %41 = arith.addf %arg19, %40 : f32
            scf.yield %41, %41 : f32, f32
          }
          "polygeist.barrier"(%arg10, %arg11, %arg12) : (index, index, index) -> ()
          %36 = "polygeist.subindex"(%arg16, %c2) : (memref<?xf32>, index) -> memref<?xf32>
          %37 = "polygeist.subindex"(%arg15, %15) : (memref<?xf32>, index) -> memref<?xf32>
          scf.yield %35#1, %37, %36, %35#1 : f32, memref<?xf32>, memref<?xf32>, f32
        }
        %30 = arith.muli %arg11, %1 : index
        %31 = arith.addi %arg10, %30 : index
        %32 = arith.addi %31, %24 : index
        memref.store %29#3, %arg4[%32] : memref<?xf32>
        scf.yield
      }
      scf.yield
    }
    return
  }
}

// CHECK:  func.func @trivial(%[[arg0:.+]]: i32, %[[arg1:.+]]: i1)
// CHECK-NEXT:    %[[c0:.+]] = arith.constant 0 : index
// CHECK-NEXT:    %[[c1:.+]] = arith.constant 1 : index
// CHECK-NEXT:    %[[c9:.+]] = arith.constant 9 : index
// CHECK-NEXT:    memref.alloca_scope  {
// CHECK-NEXT:      scf.parallel (%[[arg2:.+]]) = (%[[c0]]) to (%[[c9]]) step (%[[c1]]) {
// CHECK-NEXT:        %[[V0:.+]] = arith.index_cast %[[arg2]] : index to i32
// CHECK-NEXT:        func.call @use(%[[V0:.+]]) : (i32) -> ()
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK:  func.func @add_if_barrier(%[[arg0:.+]]: i1, %[[arg1:.+]]: memref<i32>, %[[arg2:.+]]: memref<i32>)
// CHECK-NEXT:    %[[c0:.+]] = arith.constant 0 : index
// CHECK-NEXT:    %[[c1:.+]] = arith.constant 1 : index
// CHECK-NEXT:    %[[c9:.+]] = arith.constant 9 : index
// CHECK-NEXT:    memref.alloca_scope  {
// CHECK-NEXT:      %[[V0:.+]] = memref.alloca(%[[c9]]) : memref<?xi32>
// CHECK-NEXT:      scf.parallel (%[[arg3:.+]]) = (%[[c0]]) to (%[[c9]]) step (%[[c1]]) {
// CHECK-NEXT:        %[[V1:.+]] = memref.load %[[arg1]][] : memref<i32>
// CHECK-NEXT:        %[[V2:.+]] = memref.load %[[arg2]][] : memref<i32>
// CHECK-NEXT:        %[[V3:.+]] = arith.muli %[[V1]], %[[V2]] : i32
// CHECK-NEXT:        memref.store %[[V3]], %[[V0]][%[[arg3]]] : memref<?xi32>
// CHECK-NEXT:        func.call @use(%[[V3:.+]]) : (i32) -> ()
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.if %[[arg0]] {
// CHECK-NEXT:        memref.alloca_scope  {
// CHECK-NEXT:          scf.parallel (%[[arg3:.+]]) = (%[[c0]]) to (%[[c9]]) step (%[[c1]]) {
// CHECK-NEXT:            %[[V1:.+]] = memref.load %[[V0]][%[[arg3]]] : memref<?xi32>
// CHECK-NEXT:            func.call @use(%[[V1:.+]]) : (i32) -> ()
// CHECK-NEXT:            scf.yield
// CHECK-NEXT:          }
// CHECK-NEXT:          scf.parallel (%[[arg3:.+]]) = (%[[c0]]) to (%[[c9]]) step (%[[c1]]) {
// CHECK-NEXT:            %[[V1:.+]] = memref.load %[[V0]][%[[arg3]]] : memref<?xi32>
// CHECK-NEXT:            func.call @use(%[[V1:.+]]) : (i32) -> ()
// CHECK-NEXT:            scf.yield
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      } else {
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK:  func.func @mincut_for_barrier(%[[arg0:.+]]: memref<i32>, %[[arg1:.+]]: i1, %[[arg2:.+]]: index)
// CHECK-NEXT:    %[[c0:.+]] = arith.constant 0 : index
// CHECK-NEXT:    %[[c1:.+]] = arith.constant 1 : index
// CHECK-NEXT:    %[[c1_i32:.+]] = arith.constant 1 : i32
// CHECK-NEXT:    %[[c2_i32:.+]] = arith.constant 2 : i32
// CHECK-NEXT:    %[[c3_i32:.+]] = arith.constant 3 : i32
// CHECK-NEXT:    %[[c9:.+]] = arith.constant 9 : index
// CHECK-NEXT:    memref.alloca_scope  {
// CHECK-NEXT:      %[[V0:.+]] = memref.alloca(%[[c9]]) : memref<?xi32>
// CHECK-NEXT:      scf.parallel (%[[arg3:.+]]) = (%[[c0]]) to (%[[c9]]) step (%[[c1]]) {
// CHECK-NEXT:        %[[V1:.+]] = memref.load %[[arg0]][] : memref<i32>
// CHECK-NEXT:        memref.store %[[V1]], %[[V0]][%[[arg3]]] : memref<?xi32>
// CHECK-NEXT:        func.call @use(%[[V1:.+]]) : (i32) -> ()
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.for %[[arg3:.+]] = %[[c0]] to %[[arg2]] step %[[c1]] {
// CHECK-NEXT:        memref.alloca_scope  {
// CHECK-NEXT:          scf.parallel (%[[arg4:.+]]) = (%[[c0]]) to (%[[c9]]) step (%[[c1]]) {
// CHECK-NEXT:            %[[V1:.+]] = memref.load %[[V0]][%[[arg4]]] : memref<?xi32>
// CHECK-NEXT:            %[[V2:.+]] = arith.addi %[[V1]], %[[c3_i32]] : i32
// CHECK-NEXT:            %[[V3:.+]] = arith.addi %[[V1]], %[[c2_i32]] : i32
// CHECK-NEXT:            %[[V4:.+]] = arith.addi %[[V1]], %[[c1_i32]] : i32
// CHECK-NEXT:            func.call @use(%[[V4:.+]]) : (i32) -> ()
// CHECK-NEXT:            func.call @use(%[[V3:.+]]) : (i32) -> ()
// CHECK-NEXT:            func.call @use(%[[V2:.+]]) : (i32) -> ()
// CHECK-NEXT:            scf.yield
// CHECK-NEXT:          }
// CHECK-NEXT:          scf.parallel (%[[arg4:.+]]) = (%[[c0]]) to (%[[c9]]) step (%[[c1]]) {
// CHECK-NEXT:            %[[V1:.+]] = memref.load %[[V0]][%[[arg4]]] : memref<?xi32>
// CHECK-NEXT:            %[[V2:.+]] = arith.addi %[[V1]], %[[c1_i32]] : i32
// CHECK-NEXT:            %[[V3:.+]] = arith.addi %[[V1]], %[[c2_i32]] : i32
// CHECK-NEXT:            func.call @use(%[[V2:.+]]) : (i32) -> ()
// CHECK-NEXT:            func.call @use(%[[V3:.+]]) : (i32) -> ()
// CHECK-NEXT:            scf.yield
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK:  func.func @mincut_if_barrier(%[[arg0:.+]]: memref<i32>, %[[arg1:.+]]: i1)
// CHECK-NEXT:    %[[c0:.+]] = arith.constant 0 : index
// CHECK-NEXT:    %[[c1:.+]] = arith.constant 1 : index
// CHECK-NEXT:    %[[c1_i32:.+]] = arith.constant 1 : i32
// CHECK-NEXT:    %[[c2_i32:.+]] = arith.constant 2 : i32
// CHECK-NEXT:    %[[c3_i32:.+]] = arith.constant 3 : i32
// CHECK-NEXT:    %[[c9:.+]] = arith.constant 9 : index
// CHECK-NEXT:    memref.alloca_scope  {
// CHECK-NEXT:      %[[V0:.+]] = memref.alloca(%[[c9]]) : memref<?xi32>
// CHECK-NEXT:      scf.parallel (%[[arg2:.+]]) = (%[[c0]]) to (%[[c9]]) step (%[[c1]]) {
// CHECK-NEXT:        %[[V1:.+]] = memref.load %[[arg0]][] : memref<i32>
// CHECK-NEXT:        memref.store %[[V1]], %[[V0]][%[[arg2]]] : memref<?xi32>
// CHECK-NEXT:        func.call @use(%[[V1:.+]]) : (i32) -> ()
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.if %[[arg1]] {
// CHECK-NEXT:        memref.alloca_scope  {
// CHECK-NEXT:          scf.parallel (%[[arg2:.+]]) = (%[[c0]]) to (%[[c9]]) step (%[[c1]]) {
// CHECK-NEXT:            %[[V1:.+]] = memref.load %[[V0]][%[[arg2]]] : memref<?xi32>
// CHECK-NEXT:            %[[V2:.+]] = arith.addi %[[V1]], %[[c3_i32]] : i32
// CHECK-NEXT:            %[[V3:.+]] = arith.addi %[[V1]], %[[c2_i32]] : i32
// CHECK-NEXT:            %[[V4:.+]] = arith.addi %[[V1]], %[[c1_i32]] : i32
// CHECK-NEXT:            func.call @use(%[[V4:.+]]) : (i32) -> ()
// CHECK-NEXT:            func.call @use(%[[V3:.+]]) : (i32) -> ()
// CHECK-NEXT:            func.call @use(%[[V2:.+]]) : (i32) -> ()
// CHECK-NEXT:            scf.yield
// CHECK-NEXT:          }
// CHECK-NEXT:          scf.parallel (%[[arg2:.+]]) = (%[[c0]]) to (%[[c9]]) step (%[[c1]]) {
// CHECK-NEXT:            %[[V1:.+]] = memref.load %[[V0]][%[[arg2]]] : memref<?xi32>
// CHECK-NEXT:            %[[V2:.+]] = arith.addi %[[V1]], %[[c1_i32]] : i32
// CHECK-NEXT:            %[[V3:.+]] = arith.addi %[[V1]], %[[c2_i32]] : i32
// CHECK-NEXT:            %[[V4:.+]] = arith.addi %[[V1]], %[[c3_i32]] : i32
// CHECK-NEXT:            func.call @use(%[[V2:.+]]) : (i32) -> ()
// CHECK-NEXT:            func.call @use(%[[V3:.+]]) : (i32) -> ()
// CHECK-NEXT:            func.call @use(%[[V4:.+]]) : (i32) -> ()
// CHECK-NEXT:            scf.yield
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      } else {
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK:  func.func @mincut(%[[arg0:.+]]: memref<i32>, %[[arg1:.+]]: memref<i32>)
// CHECK-NEXT:    %[[c0:.+]] = arith.constant 0 : index
// CHECK-NEXT:    %[[c1:.+]] = arith.constant 1 : index
// CHECK-NEXT:    %[[c1_i32:.+]] = arith.constant 1 : i32
// CHECK-NEXT:    %[[c2_i32:.+]] = arith.constant 2 : i32
// CHECK-NEXT:    %[[c3_i32:.+]] = arith.constant 3 : i32
// CHECK-NEXT:    %[[c9:.+]] = arith.constant 9 : index
// CHECK-NEXT:    memref.alloca_scope  {
// CHECK-NEXT:      %[[V0:.+]] = memref.alloca(%[[c9]]) : memref<?xi32>
// CHECK-NEXT:      scf.parallel (%[[arg2:.+]]) = (%[[c0]]) to (%[[c9]]) step (%[[c1]]) {
// CHECK-NEXT:        %[[V1:.+]] = memref.load %[[arg0]][] : memref<i32>
// CHECK-NEXT:        memref.store %[[V1]], %[[V0]][%[[arg2]]] : memref<?xi32>
// CHECK-NEXT:        func.call @use(%[[V1:.+]]) : (i32) -> ()
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.parallel (%[[arg2:.+]]) = (%[[c0]]) to (%[[c9]]) step (%[[c1]]) {
// CHECK-NEXT:        %[[V1:.+]] = memref.load %[[V0]][%[[arg2]]] : memref<?xi32>
// CHECK-NEXT:        %[[V2:.+]] = arith.addi %[[V1]], %[[c3_i32]] : i32
// CHECK-NEXT:        %[[V3:.+]] = arith.addi %[[V1]], %[[c2_i32]] : i32
// CHECK-NEXT:        %[[V4:.+]] = arith.addi %[[V1]], %[[c1_i32]] : i32
// CHECK-NEXT:        func.call @use(%[[V4:.+]]) : (i32) -> ()
// CHECK-NEXT:        func.call @use(%[[V3:.+]]) : (i32) -> ()
// CHECK-NEXT:        func.call @use(%[[V2:.+]]) : (i32) -> ()
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK:  func.func @add(%[[arg0:.+]]: memref<i32>, %[[arg1:.+]]: memref<i32>)
// CHECK-NEXT:    %[[c0:.+]] = arith.constant 0 : index
// CHECK-NEXT:    %[[c1:.+]] = arith.constant 1 : index
// CHECK-NEXT:    %[[c9:.+]] = arith.constant 9 : index
// CHECK-NEXT:    memref.alloca_scope  {
// CHECK-NEXT:      %[[V0:.+]] = memref.alloca(%[[c9]]) : memref<?xi32>
// CHECK-NEXT:      scf.parallel (%[[arg2:.+]]) = (%[[c0]]) to (%[[c9]]) step (%[[c1]]) {
// CHECK-NEXT:        %[[V1:.+]] = memref.load %[[arg0]][] : memref<i32>
// CHECK-NEXT:        %[[V2:.+]] = memref.load %[[arg1]][] : memref<i32>
// CHECK-NEXT:        %[[V3:.+]] = arith.muli %[[V1]], %[[V2]] : i32
// CHECK-NEXT:        memref.store %[[V3]], %[[V0]][%[[arg2]]] : memref<?xi32>
// CHECK-NEXT:        func.call @use(%[[V3:.+]]) : (i32) -> ()
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.parallel (%[[arg2:.+]]) = (%[[c0]]) to (%[[c9]]) step (%[[c1]]) {
// CHECK-NEXT:        %[[V1:.+]] = memref.load %[[V0]][%[[arg2]]] : memref<?xi32>
// CHECK-NEXT:        func.call @use(%[[V1:.+]]) : (i32) -> ()
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// CHECK:   func.func @matmul(%[[arg0:.+]]: memref<?x3xi32>, %[[arg1:.+]]: memref<?x3xi32>, %[[arg2:.+]]: memref<?xf32>, %[[arg3:.+]]: memref<?xf32>, %[[arg4:.+]]: memref<?xf32>, %[[arg5:.+]]: i32, %[[arg6:.+]]: i32)
// CHECK-NEXT:    %[[cst:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %[[c1:.+]] = arith.constant 1 : index
// CHECK-NEXT:    %[[c0:.+]] = arith.constant 0 : index
// CHECK-NEXT:    %[[c2:.+]] = arith.constant 2 : index
// CHECK-NEXT:    %[[V0:.+]] = arith.index_cast %[[arg5]] : i32 to index
// CHECK-NEXT:    %[[V1:.+]] = arith.index_cast %[[arg6]] : i32 to index
// CHECK-NEXT:    %[[V2:.+]] = affine.load %[[arg0]][0, 0] : memref<?x3xi32>
// CHECK-NEXT:    %[[V3:.+]] = affine.load %[[arg0]][0, 1] : memref<?x3xi32>
// CHECK-NEXT:    %[[V4:.+]] = affine.load %[[arg0]][0, 2] : memref<?x3xi32>
// CHECK-NEXT:    %[[V5:.+]] = arith.index_cast %[[V2]] : i32 to index
// CHECK-NEXT:    %[[V6:.+]] = arith.index_cast %[[V3]] : i32 to index
// CHECK-NEXT:    %[[V7:.+]] = arith.index_cast %[[V4]] : i32 to index
// CHECK-NEXT:    %[[V8:.+]] = affine.load %[[arg1]][0, 0] : memref<?x3xi32>
// CHECK-NEXT:    %[[V9:.+]] = affine.load %[[arg1]][0, 1] : memref<?x3xi32>
// CHECK-NEXT:    %[[V10:.+]] = affine.load %[[arg1]][0, 2] : memref<?x3xi32>
// CHECK-NEXT:    %[[V11:.+]] = arith.index_cast %[[V8]] : i32 to index
// CHECK-NEXT:    %[[V12:.+]] = arith.index_cast %[[V9]] : i32 to index
// CHECK-NEXT:    %[[V13:.+]] = arith.index_cast %[[V10]] : i32 to index
// CHECK-NEXT:    %[[V14:.+]] = arith.divsi %[[V0]], %[[c2]] : index
// CHECK-NEXT:    %[[V15:.+]] = arith.muli %[[V1]], %[[c2]] : index
// CHECK-NEXT:    scf.parallel (%[[arg7:.+]], %[[arg8:.+]], %[[arg9:.+]]) = (%[[c0]], %[[c0]], %[[c0]]) to (%[[V5]], %[[V6]], %[[V7]]) step (%[[c1]], %[[c1]], %[[c1]]) {
// CHECK-NEXT:      %[[V16:.+]] = memref.alloca() : memref<2x2xf32>
// CHECK-NEXT:      %[[V17:.+]] = memref.alloca() : memref<2x2xf32>
// CHECK-NEXT:      %[[V18:.+]] = arith.muli %[[arg8]], %[[c2]] : index
// CHECK-NEXT:      %[[V19:.+]] = arith.muli %[[V18]], %[[V0]] : index
// CHECK-NEXT:      %[[V20:.+]] = "polygeist.subindex"(%[[arg2]], %[[V19]]) : (memref<?xf32>, index) -> memref<?xf32>
// CHECK-NEXT:      %[[V21:.+]] = arith.muli %[[arg7]], %[[c2]] : index
// CHECK-NEXT:      %[[V22:.+]] = "polygeist.subindex"(%[[arg3]], %[[V21]]) : (memref<?xf32>, index) -> memref<?xf32>
// CHECK-NEXT:      %[[V23:.+]] = arith.muli %[[V18]], %[[V1]] : index
// CHECK-NEXT:      %[[V24:.+]] = arith.addi %[[V21]], %[[V23]] : index
// CHECK-NEXT:      memref.alloca_scope  {
// CHECK-NEXT:        %[[V25:.+]] = memref.alloca(%[[V11]], %[[V12]], %[[V13]]) : memref<?x?x?xf32>
// CHECK-NEXT:        %[[V26:.+]] = memref.alloca(%[[V11]], %[[V12]], %[[V13]]) : memref<?x?x?xmemref<?xf32>>
// CHECK-NEXT:        %[[V27:.+]] = memref.alloca(%[[V11]], %[[V12]], %[[V13]]) : memref<?x?x?xmemref<?xf32>>
// CHECK-NEXT:        %[[V28:.+]] = memref.alloca(%[[V11]], %[[V12]], %[[V13]]) : memref<?x?x?xf32>
// CHECK-NEXT:        scf.parallel (%[[arg10:.+]], %[[arg11:.+]], %[[arg12:.+]]) = (%[[c0]], %[[c0]], %[[c0]]) to (%[[V11]], %[[V12]], %[[V13]]) step (%[[c1]], %[[c1]], %[[c1]]) {
// CHECK-NEXT:          %[[V29:.+]] = "polygeist.subindex"(%[[V25]], %[[arg10]]) : (memref<?x?x?xf32>, index) -> memref<?x?xf32>
// CHECK-NEXT:          %[[V30:.+]] = "polygeist.subindex"(%[[V29]], %[[arg11]]) : (memref<?x?xf32>, index) -> memref<?xf32>
// CHECK-NEXT:          %[[V31:.+]] = "polygeist.subindex"(%[[V30]], %[[arg12]]) : (memref<?xf32>, index) -> memref<f32>
// CHECK-NEXT:          memref.store %[[cst]], %[[V31]][] : memref<f32>
// CHECK-NEXT:          %[[V32:.+]] = "polygeist.subindex"(%[[V26]], %[[arg10]]) : (memref<?x?x?xmemref<?xf32>>, index) -> memref<?x?xmemref<?xf32>>
// CHECK-NEXT:          %[[V33:.+]] = "polygeist.subindex"(%[[V32]], %[[arg11]]) : (memref<?x?xmemref<?xf32>>, index) -> memref<?xmemref<?xf32>>
// CHECK-NEXT:          %[[V34:.+]] = "polygeist.subindex"(%[[V33]], %[[arg12]]) : (memref<?xmemref<?xf32>>, index) -> memref<memref<?xf32>>
// CHECK-NEXT:          memref.store %[[V22]], %[[V34]][] : memref<memref<?xf32>>
// CHECK-NEXT:          %[[V35:.+]] = "polygeist.subindex"(%[[V27]], %[[arg10]]) : (memref<?x?x?xmemref<?xf32>>, index) -> memref<?x?xmemref<?xf32>>
// CHECK-NEXT:          %[[V36:.+]] = "polygeist.subindex"(%[[V35]], %[[arg11]]) : (memref<?x?xmemref<?xf32>>, index) -> memref<?xmemref<?xf32>>
// CHECK-NEXT:          %[[V37:.+]] = "polygeist.subindex"(%[[V36]], %[[arg12]]) : (memref<?xmemref<?xf32>>, index) -> memref<memref<?xf32>>
// CHECK-NEXT:          memref.store %[[V20]], %[[V37]][] : memref<memref<?xf32>>
// CHECK-NEXT:          %[[V38:.+]] = "polygeist.subindex"(%[[V28]], %[[arg10]]) : (memref<?x?x?xf32>, index) -> memref<?x?xf32>
// CHECK-NEXT:          %[[V39:.+]] = "polygeist.subindex"(%[[V38]], %[[arg11]]) : (memref<?x?xf32>, index) -> memref<?xf32>
// CHECK-NEXT:          %[[V40:.+]] = "polygeist.subindex"(%[[V39]], %[[arg12]]) : (memref<?xf32>, index) -> memref<f32>
// CHECK-NEXT:          memref.store %[[cst]], %[[V40]][] : memref<f32>
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }
// CHECK-NEXT:        memref.alloca_scope  {
// CHECK-NEXT:          scf.for %[[arg10:.+]] = %[[c0]] to %[[V14]] step %[[c1]] {
// CHECK-NEXT:            memref.alloca_scope  {
// CHECK-NEXT:              scf.parallel (%[[arg11:.+]], %[[arg12:.+]], %[[arg13:.+]]) = (%[[c0]], %[[c0]], %[[c0]]) to (%[[V11]], %[[V12]], %[[V13]]) step (%[[c1]], %[[c1]], %[[c1]]) {
// CHECK-NEXT:                %[[V29:.+]] = arith.muli %[[V1]], %[[arg12]] : index
// CHECK-NEXT:                %[[V30:.+]] = arith.addi %[[V29]], %[[arg11]] : index
// CHECK-NEXT:                %[[V31:.+]] = arith.muli %[[V0]], %[[arg12]] : index
// CHECK-NEXT:                %[[V32:.+]] = arith.addi %[[V31]], %[[arg11]] : index
// CHECK-NEXT:                %[[V33:.+]] = "polygeist.subindex"(%[[V26]], %[[arg11]]) : (memref<?x?x?xmemref<?xf32>>, index) -> memref<?x?xmemref<?xf32>>
// CHECK-NEXT:                %[[V34:.+]] = "polygeist.subindex"(%[[V33]], %[[arg12]]) : (memref<?x?xmemref<?xf32>>, index) -> memref<?xmemref<?xf32>>
// CHECK-NEXT:                %[[V35:.+]] = "polygeist.subindex"(%[[V34]], %[[arg13]]) : (memref<?xmemref<?xf32>>, index) -> memref<memref<?xf32>>
// CHECK-NEXT:                %[[V36:.+]] = memref.load %[[V35]][] : memref<memref<?xf32>>
// CHECK-NEXT:                %[[V37:.+]] = "polygeist.subindex"(%[[V27]], %[[arg11]]) : (memref<?x?x?xmemref<?xf32>>, index) -> memref<?x?xmemref<?xf32>>
// CHECK-NEXT:                %[[V38:.+]] = "polygeist.subindex"(%[[V37]], %[[arg12]]) : (memref<?x?xmemref<?xf32>>, index) -> memref<?xmemref<?xf32>>
// CHECK-NEXT:                %[[V39:.+]] = "polygeist.subindex"(%[[V38]], %[[arg13]]) : (memref<?xmemref<?xf32>>, index) -> memref<memref<?xf32>>
// CHECK-NEXT:                %[[V40:.+]] = memref.load %[[V39]][] : memref<memref<?xf32>>
// CHECK-NEXT:                %[[V41:.+]] = memref.load %[[V40]][%[[V32]]] : memref<?xf32>
// CHECK-NEXT:                memref.store %[[V41]], %[[V16]][%[[arg12]], %[[arg11]]] : memref<2x2xf32>
// CHECK-NEXT:                %[[V42:.+]] = memref.load %[[V36]][%[[V30]]] : memref<?xf32>
// CHECK-NEXT:                memref.store %[[V42]], %[[V17]][%[[arg12]], %[[arg11]]] : memref<2x2xf32>
// CHECK-NEXT:                scf.yield
// CHECK-NEXT:              }
// CHECK-NEXT:              scf.parallel (%[[arg11:.+]], %[[arg12:.+]], %[[arg13:.+]]) = (%[[c0]], %[[c0]], %[[c0]]) to (%[[V11]], %[[V12]], %[[V13]]) step (%[[c1]], %[[c1]], %[[c1]]) {
// CHECK-NEXT:                %[[V29:.+]] = "polygeist.subindex"(%[[V27]], %[[arg11]]) : (memref<?x?x?xmemref<?xf32>>, index) -> memref<?x?xmemref<?xf32>>
// CHECK-NEXT:                %[[V30:.+]] = "polygeist.subindex"(%[[V29]], %[[arg12]]) : (memref<?x?xmemref<?xf32>>, index) -> memref<?xmemref<?xf32>>
// CHECK-NEXT:                %[[V31:.+]] = "polygeist.subindex"(%[[V30]], %[[arg13]]) : (memref<?xmemref<?xf32>>, index) -> memref<memref<?xf32>>
// CHECK-NEXT:                %[[V32:.+]] = memref.load %[[V31]][] : memref<memref<?xf32>>
// CHECK-NEXT:                %[[V33:.+]] = "polygeist.subindex"(%[[V26]], %[[arg11]]) : (memref<?x?x?xmemref<?xf32>>, index) -> memref<?x?xmemref<?xf32>>
// CHECK-NEXT:                %[[V34:.+]] = "polygeist.subindex"(%[[V33]], %[[arg12]]) : (memref<?x?xmemref<?xf32>>, index) -> memref<?xmemref<?xf32>>
// CHECK-NEXT:                %[[V35:.+]] = "polygeist.subindex"(%[[V34]], %[[arg13]]) : (memref<?xmemref<?xf32>>, index) -> memref<memref<?xf32>>
// CHECK-NEXT:                %[[V36:.+]] = memref.load %[[V35]][] : memref<memref<?xf32>>
// CHECK-NEXT:                %[[V37:.+]] = "polygeist.subindex"(%[[V25]], %[[arg11]]) : (memref<?x?x?xf32>, index) -> memref<?x?xf32>
// CHECK-NEXT:                %[[V38:.+]] = "polygeist.subindex"(%[[V37]], %[[arg12]]) : (memref<?x?xf32>, index) -> memref<?xf32>
// CHECK-NEXT:                %[[V39:.+]] = "polygeist.subindex"(%[[V38]], %[[arg13]]) : (memref<?xf32>, index) -> memref<f32>
// CHECK-NEXT:                %[[V40:.+]] = memref.load %[[V39]][] : memref<f32>
// CHECK-NEXT:                %[[V41]]:2 = scf.for %[[arg14:.+]] = %[[c0:.+]] to %[[c2:.+]] step %[[c1:.+]] iter_args(%[[arg15:.+]] = %[[V40:.+]], %[[arg16:.+]] = %[[V40]]) -> (f32, f32) {
// CHECK-NEXT:                  %[[V56:.+]] = memref.load %[[V16]][%[[arg12]], %[[arg14]]] : memref<2x2xf32>
// CHECK-NEXT:                  %[[V57:.+]] = memref.load %[[V17]][%[[arg14]], %[[arg11]]] : memref<2x2xf32>
// CHECK-NEXT:                  %[[V58:.+]] = arith.mulf %[[V56]], %[[V57]] : f32
// CHECK-NEXT:                  %[[V59:.+]] = arith.addf %[[arg15]], %[[V58]] : f32
// CHECK-NEXT:                  scf.yield %[[V59]], %[[V59]] : f32, f32
// CHECK-NEXT:                }
// CHECK-NEXT:                %[[V42:.+]] = "polygeist.subindex"(%[[V32]], %[[c2]]) : (memref<?xf32>, index) -> memref<?xf32>
// CHECK-NEXT:                %[[V43:.+]] = "polygeist.subindex"(%[[V36]], %[[V15]]) : (memref<?xf32>, index) -> memref<?xf32>
// CHECK-NEXT:                %[[V44:.+]] = "polygeist.subindex"(%[[V25]], %[[arg11]]) : (memref<?x?x?xf32>, index) -> memref<?x?xf32>
// CHECK-NEXT:                %[[V45:.+]] = "polygeist.subindex"(%[[V44]], %[[arg12]]) : (memref<?x?xf32>, index) -> memref<?xf32>
// CHECK-NEXT:                %[[V46:.+]] = "polygeist.subindex"(%[[V45]], %[[arg13]]) : (memref<?xf32>, index) -> memref<f32>
// CHECK-NEXT:                memref.store %[[V41]]#1, %[[V46]][] : memref<f32>
// CHECK-NEXT:                %[[V47:.+]] = "polygeist.subindex"(%[[V26]], %[[arg11]]) : (memref<?x?x?xmemref<?xf32>>, index) -> memref<?x?xmemref<?xf32>>
// CHECK-NEXT:                %[[V48:.+]] = "polygeist.subindex"(%[[V47]], %[[arg12]]) : (memref<?x?xmemref<?xf32>>, index) -> memref<?xmemref<?xf32>>
// CHECK-NEXT:                %[[V49:.+]] = "polygeist.subindex"(%[[V48]], %[[arg13]]) : (memref<?xmemref<?xf32>>, index) -> memref<memref<?xf32>>
// CHECK-NEXT:                memref.store %[[V43]], %[[V49]][] : memref<memref<?xf32>>
// CHECK-NEXT:                %[[V50:.+]] = "polygeist.subindex"(%[[V27]], %[[arg11]]) : (memref<?x?x?xmemref<?xf32>>, index) -> memref<?x?xmemref<?xf32>>
// CHECK-NEXT:                %[[V51:.+]] = "polygeist.subindex"(%[[V50]], %[[arg12]]) : (memref<?x?xmemref<?xf32>>, index) -> memref<?xmemref<?xf32>>
// CHECK-NEXT:                %[[V52:.+]] = "polygeist.subindex"(%[[V51]], %[[arg13]]) : (memref<?xmemref<?xf32>>, index) -> memref<memref<?xf32>>
// CHECK-NEXT:                memref.store %[[V42]], %[[V52]][] : memref<memref<?xf32>>
// CHECK-NEXT:                %[[V53:.+]] = "polygeist.subindex"(%[[V28]], %[[arg11]]) : (memref<?x?x?xf32>, index) -> memref<?x?xf32>
// CHECK-NEXT:                %[[V54:.+]] = "polygeist.subindex"(%[[V53]], %[[arg12]]) : (memref<?x?xf32>, index) -> memref<?xf32>
// CHECK-NEXT:                %[[V55:.+]] = "polygeist.subindex"(%[[V54]], %[[arg13]]) : (memref<?xf32>, index) -> memref<f32>
// CHECK-NEXT:                memref.store %[[V41]]#1, %[[V55]][] : memref<f32>
// CHECK-NEXT:                scf.yield
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:          }
// CHECK-NEXT:          scf.parallel (%[[arg10:.+]], %[[arg11:.+]], %[[arg12:.+]]) = (%[[c0]], %[[c0]], %[[c0]]) to (%[[V11]], %[[V12]], %[[V13]]) step (%[[c1]], %[[c1]], %[[c1]]) {
// CHECK-NEXT:            %[[V29:.+]] = "polygeist.subindex"(%[[V28]], %[[arg10]]) : (memref<?x?x?xf32>, index) -> memref<?x?xf32>
// CHECK-NEXT:            %[[V30:.+]] = "polygeist.subindex"(%[[V29]], %[[arg11]]) : (memref<?x?xf32>, index) -> memref<?xf32>
// CHECK-NEXT:            %[[V31:.+]] = "polygeist.subindex"(%[[V30]], %[[arg12]]) : (memref<?xf32>, index) -> memref<f32>
// CHECK-NEXT:            %[[V32:.+]] = memref.load %[[V31]][] : memref<f32>
// CHECK-NEXT:            %[[V33:.+]] = arith.muli %[[arg11]], %[[V1]] : index
// CHECK-NEXT:            %[[V34:.+]] = arith.addi %[[arg10]], %[[V33]] : index
// CHECK-NEXT:            %[[V35:.+]] = arith.addi %[[V34]], %[[V24]] : index
// CHECK-NEXT:            memref.store %[[V32]], %[[arg4]][%[[V35]]] : memref<?xf32>
// CHECK-NEXT:            scf.yield
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield
// CHECK-NEXT:    }
// CHECK-NEXT:    return
