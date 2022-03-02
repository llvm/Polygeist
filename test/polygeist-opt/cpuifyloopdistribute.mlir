// RUN: polygeist-opt --cpuify="method=distribute" --split-input-file %s | FileCheck %s

module {
  func private @use(%arg0: i32)
  func @trivial(%arg0: i32, %c : i1) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c9 = arith.constant 9 : index
    scf.parallel (%arg4) = (%c0) to (%c9) step (%c1) {
      %0 = arith.index_cast %arg4 : index to i32
      "polygeist.barrier"(%arg4) : (index) -> ()
      call @use(%0) : (i32) -> ()
      scf.yield
    }
    return
  }

//  func @add_if_barrier(%arg: i1, %amem: memref<i32>, %bmem : memref<i32>) attributes {llvm.linkage = #llvm.linkage<external>} {
//    %c0 = arith.constant 0 : index
//    %c1 = arith.constant 1 : index
//    %c9 = arith.constant 9 : index
//    %alloc = memref.alloca() : memref<i32>
//    scf.parallel (%arg4) = (%c0) to (%c9) step (%c1) {
//      %a = memref.load %amem[] : memref<i32>
//      %b = memref.load %bmem[] : memref<i32>
//      %mul = arith.muli %a, %b : i32
//      call @use(%mul) : (i32) -> ()
//      "polygeist.barrier"(%arg4) : (index) -> ()
//      scf.if %arg {
//        call @use(%mul) : (i32) -> ()
//        "polygeist.barrier"(%arg4) : (index) -> ()
//        call @use(%mul) : (i32) -> ()
//        scf.yield
//      }
//      scf.yield
//    }
//    return
//  }
  func @mincut_for_barrier(%amem: memref<i32>, %arg: i1, %bound: index) attributes {llvm.linkage = #llvm.linkage<external>} {
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
      call @use(%a) : (i32) -> ()
      "polygeist.barrier"(%arg4) : (index) -> ()
      scf.for %forarg = %c0 to %bound step %c1 {
        call @use(%a1) : (i32) -> ()
        call @use(%a2) : (i32) -> ()
        call @use(%a3) : (i32) -> ()
        "polygeist.barrier"(%arg4) : (index) -> ()
        call @use(%a1) : (i32) -> ()
        call @use(%a2) : (i32) -> ()
        scf.yield
      }
      scf.yield
    }
    return
  }
//  func @mincut_if_barrier(%amem: memref<i32>, %arg : i1) attributes {llvm.linkage = #llvm.linkage<external>} {
//    %c0 = arith.constant 0 : index
//    %c1 = arith.constant 1 : index
//    %i1 = arith.constant 1 : i32
//    %i2 = arith.constant 2 : i32
//    %i3 = arith.constant 3 : i32
//    %c9 = arith.constant 9 : index
//    %alloc = memref.alloca() : memref<i32>
//    scf.parallel (%arg4) = (%c0) to (%c9) step (%c1) {
//      %a = memref.load %amem[] : memref<i32>
//      %a1 = arith.addi %a, %i1 : i32
//      %a2 = arith.addi %a, %i2 : i32
//      %a3 = arith.addi %a, %i3 : i32
//      call @use(%a) : (i32) -> ()
//      "polygeist.barrier"(%arg4) : (index) -> ()
//      scf.if %arg {
//        call @use(%a1) : (i32) -> ()
//        call @use(%a2) : (i32) -> ()
//        call @use(%a3) : (i32) -> ()
//        "polygeist.barrier"(%arg4) : (index) -> ()
//        call @use(%a1) : (i32) -> ()
//        call @use(%a2) : (i32) -> ()
//        call @use(%a3) : (i32) -> ()
//        scf.yield
//      }
//      scf.yield
//    }
//    return
//  }
  func @mincut(%amem: memref<i32>, %bmem : memref<i32>) attributes {llvm.linkage = #llvm.linkage<external>} {
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
      call @use(%a) : (i32) -> ()
      "polygeist.barrier"(%arg4) : (index) -> ()
      call @use(%a1) : (i32) -> ()
      call @use(%a2) : (i32) -> ()
      call @use(%a3) : (i32) -> ()
      scf.yield
    }
    return
  }
  func @add(%amem: memref<i32>, %bmem : memref<i32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c9 = arith.constant 9 : index
    %alloc = memref.alloca() : memref<i32>
    scf.parallel (%arg4) = (%c0) to (%c9) step (%c1) {
      %a = memref.load %amem[] : memref<i32>
      %b = memref.load %bmem[] : memref<i32>
      %mul = arith.muli %a, %b : i32
      call @use(%mul) : (i32) -> ()
      "polygeist.barrier"(%arg4) : (index) -> ()
      call @use(%mul) : (i32) -> ()
      scf.yield
    }
    return
  }
  func @matmul(%arg0: memref<?x3xi32>, %arg1: memref<?x3xi32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: i32, %arg6: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
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


// CHECK:  func @trivial(%arg0: i32, %arg1: i1) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c9 = arith.constant 9 : index
// CHECK-NEXT:    scf.parallel (%arg2) = (%c0) to (%c9) step (%c1) {
// CHECK-NEXT:      %0 = arith.index_cast %arg2 : index to i32
// CHECK-NEXT:      call @use(%0) : (i32) -> ()
// CHECK-NEXT:      scf.yield
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK:  func @mincut_for_barrier(%arg0: memref<i32>, %arg1: i1, %arg2: index) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:    %c2_i32 = arith.constant 2 : i32
// CHECK-NEXT:    %c3_i32 = arith.constant 3 : i32
// CHECK-NEXT:    %c9 = arith.constant 9 : index
// CHECK-NEXT:    %0 = memref.alloc(%c9) : memref<?xi32>
// CHECK-NEXT:    scf.parallel (%arg3) = (%c0) to (%c9) step (%c1) {
// CHECK-NEXT:      %1 = memref.load %arg0[] : memref<i32>
// CHECK-NEXT:      memref.store %1, %0[%arg3] : memref<?xi32>
// CHECK-NEXT:      call @use(%1) : (i32) -> ()
// CHECK-NEXT:      scf.yield
// CHECK-NEXT:    }
// CHECK-NEXT:    scf.for %arg3 = %c0 to %arg2 step %c1 {
// CHECK-NEXT:      scf.parallel (%arg4) = (%c0) to (%c9) step (%c1) {
// CHECK-NEXT:        %1 = memref.load %0[%arg4] : memref<?xi32>
// CHECK-NEXT:        %2 = arith.addi %1, %c3_i32 : i32
// CHECK-NEXT:        %3 = arith.addi %1, %c2_i32 : i32
// CHECK-NEXT:        %4 = arith.addi %1, %c1_i32 : i32
// CHECK-NEXT:        call @use(%4) : (i32) -> ()
// CHECK-NEXT:        call @use(%3) : (i32) -> ()
// CHECK-NEXT:        call @use(%2) : (i32) -> ()
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.parallel (%arg4) = (%c0) to (%c9) step (%c1) {
// CHECK-NEXT:        %1 = memref.load %0[%arg4] : memref<?xi32>
// CHECK-NEXT:        %2 = arith.addi %1, %c1_i32 : i32
// CHECK-NEXT:        %3 = arith.addi %1, %c2_i32 : i32
// CHECK-NEXT:        call @use(%2) : (i32) -> ()
// CHECK-NEXT:        call @use(%3) : (i32) -> ()
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    memref.dealloc %0 : memref<?xi32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK:  func @mincut(%arg0: memref<i32>, %arg1: memref<i32>) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:    %c2_i32 = arith.constant 2 : i32
// CHECK-NEXT:    %c3_i32 = arith.constant 3 : i32
// CHECK-NEXT:    %c9 = arith.constant 9 : index
// CHECK-NEXT:    %0 = memref.alloc(%c9) : memref<?xi32>
// CHECK-NEXT:    scf.parallel (%arg2) = (%c0) to (%c9) step (%c1) {
// CHECK-NEXT:      %1 = memref.load %arg0[] : memref<i32>
// CHECK-NEXT:      memref.store %1, %0[%arg2] : memref<?xi32>
// CHECK-NEXT:      call @use(%1) : (i32) -> ()
// CHECK-NEXT:      scf.yield
// CHECK-NEXT:    }
// CHECK-NEXT:    scf.parallel (%arg2) = (%c0) to (%c9) step (%c1) {
// CHECK-NEXT:      %1 = memref.load %0[%arg2] : memref<?xi32>
// CHECK-NEXT:      %2 = arith.addi %1, %c3_i32 : i32
// CHECK-NEXT:      %3 = arith.addi %1, %c2_i32 : i32
// CHECK-NEXT:      %4 = arith.addi %1, %c1_i32 : i32
// CHECK-NEXT:      call @use(%4) : (i32) -> ()
// CHECK-NEXT:      call @use(%3) : (i32) -> ()
// CHECK-NEXT:      call @use(%2) : (i32) -> ()
// CHECK-NEXT:      scf.yield
// CHECK-NEXT:    }
// CHECK-NEXT:    memref.dealloc %0 : memref<?xi32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK:  func @add(%arg0: memref<i32>, %arg1: memref<i32>) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c9 = arith.constant 9 : index
// CHECK-NEXT:    %0 = memref.alloc(%c9) : memref<?xi32>
// CHECK-NEXT:    scf.parallel (%arg2) = (%c0) to (%c9) step (%c1) {
// CHECK-NEXT:      %1 = memref.load %arg0[] : memref<i32>
// CHECK-NEXT:      %2 = memref.load %arg1[] : memref<i32>
// CHECK-NEXT:      %3 = arith.muli %1, %2 : i32
// CHECK-NEXT:      memref.store %3, %0[%arg2] : memref<?xi32>
// CHECK-NEXT:      call @use(%3) : (i32) -> ()
// CHECK-NEXT:      scf.yield
// CHECK-NEXT:    }
// CHECK-NEXT:    scf.parallel (%arg2) = (%c0) to (%c9) step (%c1) {
// CHECK-NEXT:      %1 = memref.load %0[%arg2] : memref<?xi32>
// CHECK-NEXT:      call @use(%1) : (i32) -> ()
// CHECK-NEXT:      scf.yield
// CHECK-NEXT:    }
// CHECK-NEXT:    memref.dealloc %0 : memref<?xi32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK:  func @matmul(%arg0: memref<?x3xi32>, %arg1: memref<?x3xi32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: i32, %arg6: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %c2 = arith.constant 2 : index
// CHECK-NEXT:    %0 = arith.index_cast %arg5 : i32 to index
// CHECK-NEXT:    %1 = arith.index_cast %arg6 : i32 to index
// CHECK-NEXT:    %2 = affine.load %arg0[0, 0] : memref<?x3xi32>
// CHECK-NEXT:    %3 = affine.load %arg0[0, 1] : memref<?x3xi32>
// CHECK-NEXT:    %4 = affine.load %arg0[0, 2] : memref<?x3xi32>
// CHECK-NEXT:    %5 = arith.index_cast %2 : i32 to index
// CHECK-NEXT:    %6 = arith.index_cast %3 : i32 to index
// CHECK-NEXT:    %7 = arith.index_cast %4 : i32 to index
// CHECK-NEXT:    %8 = affine.load %arg1[0, 0] : memref<?x3xi32>
// CHECK-NEXT:    %9 = affine.load %arg1[0, 1] : memref<?x3xi32>
// CHECK-NEXT:    %10 = affine.load %arg1[0, 2] : memref<?x3xi32>
// CHECK-NEXT:    %11 = arith.index_cast %8 : i32 to index
// CHECK-NEXT:    %12 = arith.index_cast %9 : i32 to index
// CHECK-NEXT:    %13 = arith.index_cast %10 : i32 to index
// CHECK-NEXT:    %14 = arith.divsi %0, %c2 : index
// CHECK-NEXT:    scf.parallel (%arg7, %arg8, %arg9) = (%c0, %c0, %c0) to (%5, %6, %7) step (%c1, %c1, %c1) {
// CHECK-NEXT:      %15 = memref.alloca() : memref<2x2xf32>
// CHECK-NEXT:      %16 = memref.alloca() : memref<2x2xf32>
// CHECK-NEXT:      %17 = arith.muli %arg8, %c2 : index
// CHECK-NEXT:      %18 = arith.muli %17, %0 : index
// CHECK-NEXT:      %19 = "polygeist.subindex"(%arg2, %18) : (memref<?xf32>, index) -> memref<?xf32>
// CHECK-NEXT:      %20 = arith.muli %arg7, %c2 : index
// CHECK-NEXT:      %21 = "polygeist.subindex"(%arg3, %20) : (memref<?xf32>, index) -> memref<?xf32>
// CHECK-NEXT:      %22 = arith.muli %17, %1 : index
// CHECK-NEXT:      %23 = arith.addi %20, %22 : index
// CHECK-NEXT:      %24 = memref.alloc(%11, %12, %13) : memref<?x?x?xf32>
// CHECK-NEXT:      %25 = memref.alloc(%11, %12, %13) : memref<?x?x?xmemref<?xf32>>
// CHECK-NEXT:      %26 = memref.alloc(%11, %12, %13) : memref<?x?x?xmemref<?xf32>>
// CHECK-NEXT:      %27 = memref.alloc(%11, %12, %13) : memref<?x?x?xf32>
// CHECK-NEXT:      scf.parallel (%arg10, %arg11, %arg12) = (%c0, %c0, %c0) to (%11, %12, %13) step (%c1, %c1, %c1) {
// CHECK-NEXT:        %28 = "polygeist.subindex"(%24, %arg10) : (memref<?x?x?xf32>, index) -> memref<?x?xf32>
// CHECK-NEXT:        %29 = "polygeist.subindex"(%28, %arg11) : (memref<?x?xf32>, index) -> memref<?xf32>
// CHECK-NEXT:        %30 = "polygeist.subindex"(%29, %arg12) : (memref<?xf32>, index) -> memref<f32>
// CHECK-NEXT:        memref.store %cst, %30[] : memref<f32>
// CHECK-NEXT:        %31 = "polygeist.subindex"(%25, %arg10) : (memref<?x?x?xmemref<?xf32>>, index) -> memref<?x?xmemref<?xf32>>
// CHECK-NEXT:        %32 = "polygeist.subindex"(%31, %arg11) : (memref<?x?xmemref<?xf32>>, index) -> memref<?xmemref<?xf32>>
// CHECK-NEXT:        %33 = "polygeist.subindex"(%32, %arg12) : (memref<?xmemref<?xf32>>, index) -> memref<memref<?xf32>>
// CHECK-NEXT:        memref.store %21, %33[] : memref<memref<?xf32>>
// CHECK-NEXT:        %34 = "polygeist.subindex"(%26, %arg10) : (memref<?x?x?xmemref<?xf32>>, index) -> memref<?x?xmemref<?xf32>>
// CHECK-NEXT:        %35 = "polygeist.subindex"(%34, %arg11) : (memref<?x?xmemref<?xf32>>, index) -> memref<?xmemref<?xf32>>
// CHECK-NEXT:        %36 = "polygeist.subindex"(%35, %arg12) : (memref<?xmemref<?xf32>>, index) -> memref<memref<?xf32>>
// CHECK-NEXT:        memref.store %19, %36[] : memref<memref<?xf32>>
// CHECK-NEXT:        %37 = "polygeist.subindex"(%27, %arg10) : (memref<?x?x?xf32>, index) -> memref<?x?xf32>
// CHECK-NEXT:        %38 = "polygeist.subindex"(%37, %arg11) : (memref<?x?xf32>, index) -> memref<?xf32>
// CHECK-NEXT:        %39 = "polygeist.subindex"(%38, %arg12) : (memref<?xf32>, index) -> memref<f32>
// CHECK-NEXT:        memref.store %cst, %39[] : memref<f32>
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.for %arg10 = %c0 to %14 step %c1 {
// CHECK-NEXT:        scf.parallel (%arg11, %arg12, %arg13) = (%c0, %c0, %c0) to (%11, %12, %13) step (%c1, %c1, %c1) {
// CHECK-NEXT:          %28 = arith.muli %1, %arg12 : index
// CHECK-NEXT:          %29 = arith.addi %28, %arg11 : index
// CHECK-NEXT:          %30 = arith.muli %0, %arg12 : index
// CHECK-NEXT:          %31 = arith.addi %30, %arg11 : index
// CHECK-NEXT:          %32 = "polygeist.subindex"(%25, %arg11) : (memref<?x?x?xmemref<?xf32>>, index) -> memref<?x?xmemref<?xf32>>
// CHECK-NEXT:          %33 = "polygeist.subindex"(%32, %arg12) : (memref<?x?xmemref<?xf32>>, index) -> memref<?xmemref<?xf32>>
// CHECK-NEXT:          %34 = "polygeist.subindex"(%33, %arg13) : (memref<?xmemref<?xf32>>, index) -> memref<memref<?xf32>>
// CHECK-NEXT:          %35 = memref.load %34[] : memref<memref<?xf32>>
// CHECK-NEXT:          %36 = "polygeist.subindex"(%26, %arg11) : (memref<?x?x?xmemref<?xf32>>, index) -> memref<?x?xmemref<?xf32>>
// CHECK-NEXT:          %37 = "polygeist.subindex"(%36, %arg12) : (memref<?x?xmemref<?xf32>>, index) -> memref<?xmemref<?xf32>>
// CHECK-NEXT:          %38 = "polygeist.subindex"(%37, %arg13) : (memref<?xmemref<?xf32>>, index) -> memref<memref<?xf32>>
// CHECK-NEXT:          %39 = memref.load %38[] : memref<memref<?xf32>>
// CHECK-NEXT:          %40 = memref.load %39[%31] : memref<?xf32>
// CHECK-NEXT:          memref.store %40, %15[%arg12, %arg11] : memref<2x2xf32>
// CHECK-NEXT:          %41 = memref.load %35[%29] : memref<?xf32>
// CHECK-NEXT:          memref.store %41, %16[%arg12, %arg11] : memref<2x2xf32>
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.parallel (%arg10, %arg11, %arg12) = (%c0, %c0, %c0) to (%11, %12, %13) step (%c1, %c1, %c1) {
// CHECK-NEXT:        %28 = "polygeist.subindex"(%27, %arg10) : (memref<?x?x?xf32>, index) -> memref<?x?xf32>
// CHECK-NEXT:        %29 = "polygeist.subindex"(%28, %arg11) : (memref<?x?xf32>, index) -> memref<?xf32>
// CHECK-NEXT:        %30 = "polygeist.subindex"(%29, %arg12) : (memref<?xf32>, index) -> memref<f32>
// CHECK-NEXT:        %31 = memref.load %30[] : memref<f32>
// CHECK-NEXT:        %32 = arith.muli %arg11, %1 : index
// CHECK-NEXT:        %33 = arith.addi %arg10, %32 : index
// CHECK-NEXT:        %34 = arith.addi %33, %23 : index
// CHECK-NEXT:        memref.store %31, %arg4[%34] : memref<?xf32>
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }
// CHECK-NEXT:      memref.dealloc %24 : memref<?x?x?xf32>
// CHECK-NEXT:      memref.dealloc %25 : memref<?x?x?xmemref<?xf32>>
// CHECK-NEXT:      memref.dealloc %26 : memref<?x?x?xmemref<?xf32>>
// CHECK-NEXT:      memref.dealloc %27 : memref<?x?x?xf32>
// CHECK-NEXT:      scf.yield
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK-NEXT:}

