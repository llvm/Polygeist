// RUN: polygeist-opt --cpuify="method=distribute.mincut" --split-input-file %s | FileCheck %s

module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @_Z11bpnnwrapperiPfiS_iS_S_(%arg0: i32, %arg1: memref<?xf32>, %arg2: i32, %arg3: memref<?xf32>, %arg4: i32, %arg5: memref<?xf32>, %arg6: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c16 = arith.constant 16 : index
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 3.000000e-01 : f64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg2 : i32 to index
    %1 = arith.index_cast %arg0 : i32 to index
    %2 = arith.addi %0, %c1 : index
    %3 = arith.muli %2, %c16 : index
    scf.parallel (%arg7, %arg8, %arg9) = (%c0, %c0, %c0) to (%1, %c16, %c16) step (%c1, %c1, %c1) {
      %4 = arith.index_cast %arg7 : index to i32
      %5 = arith.index_cast %arg9 : index to i32
      %6 = arith.muli %3, %arg7 : index
      %7 = arith.muli %2, %arg9 : index
      %8 = arith.addi %6, %7 : index
      %9 = arith.addi %8, %arg8 : index
      %10 = arith.addi %9, %c1 : index
      %11 = arith.muli %arg7, %c16 : index
      %12 = arith.addi %11, %arg9 : index
      %13 = arith.addi %10, %2 : index
      %14 = arith.addi %arg8, %c1 : index
      %15 = memref.load %arg1[%14] : memref<?xf32>
      %16 = arith.extf %15 : f32 to f64
      %17 = arith.mulf %cst, %16 : f64
      %18 = arith.addi %12, %c1 : index
      %19 = memref.load %arg3[%18] : memref<?xf32>
      %20 = arith.extf %19 : f32 to f64
      %21 = arith.mulf %17, %20 : f64
      %22 = memref.load %arg6[%13] : memref<?xf32>
      %23 = arith.extf %22 : f32 to f64
      %24 = arith.mulf %cst, %23 : f64
      %25 = arith.addf %21, %24 : f64
      %26 = memref.load %arg5[%13] : memref<?xf32>
      %27 = arith.truncf %25 : f64 to f32
      %28 = arith.addf %26, %27 : f32
      memref.store %28, %arg5[%13] : memref<?xf32>
      %29 = memref.load %arg1[%14] : memref<?xf32>
      %30 = arith.extf %29 : f32 to f64
      %31 = arith.mulf %cst, %30 : f64
      %32 = memref.load %arg3[%18] : memref<?xf32>
      %33 = arith.extf %32 : f32 to f64
      %34 = arith.mulf %31, %33 : f64
      %35 = memref.load %arg6[%13] : memref<?xf32>
      %36 = arith.extf %35 : f32 to f64
      %37 = arith.mulf %cst, %36 : f64
      %38 = arith.addf %34, %37 : f64
      %39 = arith.truncf %38 : f64 to f32
      memref.store %39, %arg6[%13] : memref<?xf32>
      "polygeist.barrier"(%arg8, %arg9, %c0) : (index, index, index) -> ()
      %40 = arith.cmpi eq, %5, %c0_i32 : i32
      scf.if %40 {
        %41 = arith.cmpi eq, %4, %c0_i32 : i32
        scf.if %41 {
          %42 = memref.load %arg1[%14] : memref<?xf32>
          %43 = arith.extf %42 : f32 to f64
          %44 = arith.mulf %cst, %43 : f64
          %45 = memref.load %arg6[%14] : memref<?xf32>
          %46 = arith.extf %45 : f32 to f64
          %47 = arith.mulf %cst, %46 : f64
          %48 = arith.addf %44, %47 : f64
          %49 = memref.load %arg5[%14] : memref<?xf32>
          %50 = arith.truncf %48 : f64 to f32
          %51 = arith.addf %49, %50 : f32
          memref.store %51, %arg5[%14] : memref<?xf32>
          %52 = memref.load %arg1[%14] : memref<?xf32>
          %53 = arith.extf %52 : f32 to f64
          %54 = arith.mulf %cst, %53 : f64
          %55 = memref.load %arg6[%14] : memref<?xf32>
          %56 = arith.extf %55 : f32 to f64
          %57 = arith.mulf %cst, %56 : f64
          %58 = arith.addf %54, %57 : f64
          %59 = arith.truncf %58 : f64 to f32
          memref.store %59, %arg6[%14] : memref<?xf32>
        }
      }
      scf.yield
    }
    return
  }
  func.func @_Z30bpnn_layerforward_CUDA_wrapperiPfS_S_S_ii(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: i32, %arg6: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c16 = arith.constant 16 : index
    %cst = arith.constant 2.000000e+00 : f32
    %cst_0 = arith.constant 4.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %0 = arith.index_cast %arg6 : i32 to index
    %1 = arith.index_cast %arg0 : i32 to index
    %2 = arith.addi %0, %c1 : index
    %3 = arith.muli %2, %c16 : index
    scf.parallel (%arg7) = (%c0) to (%1) step (%c1) {
      %4 = memref.alloca() : memref<16xf32>
      %5 = memref.alloca() : memref<16x16xf32>
      %6 = arith.muli %3, %arg7 : index
      %7 = arith.muli %arg7, %c16 : index
      scf.parallel (%arg8, %arg9) = (%c0, %c0) to (%c16, %c16) step (%c1, %c1) {
        %8 = arith.index_cast %arg8 : index to i32
        %9 = arith.index_cast %arg9 : index to i32
        %10 = arith.muli %2, %arg9 : index
        %11 = arith.addi %6, %10 : index
        %12 = arith.addi %11, %arg8 : index
        %13 = arith.addi %12, %c1 : index
        %14 = arith.addi %7, %arg9 : index
        %15 = arith.cmpi eq, %8, %c0_i32 : i32
        scf.if %15 {
          %23 = arith.addi %14, %c1 : index
          %24 = memref.load %arg1[%23] : memref<?xf32>
          memref.store %24, %4[%arg9] : memref<16xf32>
        }
        "polygeist.barrier"(%arg8, %arg9, %c0) : (index, index, index) -> ()
        %16 = arith.addi %13, %2 : index
        %17 = memref.load %arg3[%16] : memref<?xf32>
        memref.store %17, %5[%arg9, %arg8] : memref<16x16xf32>
        "polygeist.barrier"(%arg8, %arg9, %c0) : (index, index, index) -> ()
        %18 = memref.load %5[%arg9, %arg8] : memref<16x16xf32>
        %19 = memref.load %4[%arg9] : memref<16xf32>
        %20 = arith.mulf %18, %19 : f32
        memref.store %20, %5[%arg9, %arg8] : memref<16x16xf32>
        "polygeist.barrier"(%arg8, %arg9, %c0) : (index, index, index) -> ()
        %21 = scf.while (%arg10 = %c1_i32) : (i32) -> i32 {
          %23 = arith.sitofp %arg10 : i32 to f32
          %24 = arith.cmpf ule, %23, %cst_0 : f32
          scf.condition(%24) %arg10 : i32
        } do {
        ^bb0(%arg10: i32):  // no predecessors
          %23 = arith.sitofp %arg10 : i32 to f32
          %24 = math.powf %cst, %23 : f32
          %25 = arith.fptosi %24 : f32 to i32
          %26 = arith.index_cast %25 : i32 to index
          %27 = arith.remsi %9, %25 : i32
          %28 = arith.cmpi eq, %27, %c0_i32 : i32
          scf.if %28 {
            %30 = memref.load %5[%arg9, %arg8] : memref<16x16xf32>
            %31 = arith.divsi %26, %c2 : index
            %32 = arith.addi %arg9, %31 : index
            %33 = memref.load %5[%32, %arg8] : memref<16x16xf32>
            %34 = arith.addf %30, %33 : f32
            memref.store %34, %5[%arg9, %arg8] : memref<16x16xf32>
          }
          "polygeist.barrier"(%arg8, %arg9, %c0) : (index, index, index) -> ()
          %29 = arith.addi %arg10, %c1_i32 : i32
          scf.yield %29 : i32
        }
        %22 = memref.load %5[%arg9, %arg8] : memref<16x16xf32>
        memref.store %22, %arg3[%16] : memref<?xf32>
        "polygeist.barrier"(%arg8, %arg9, %c0) : (index, index, index) -> ()
        scf.if %15 {
          %23 = memref.load %5[%arg8, %arg9] : memref<16x16xf32>
          %24 = arith.muli %arg7, %0 : index
          %25 = arith.addi %arg9, %24 : index
          memref.store %23, %arg4[%25] : memref<?xf32>
        }
        scf.yield
      }
      scf.yield
    }
    return
  }
}

// CHECK:  func.func @_Z11bpnnwrapperiPfiS_iS_S_(%[[arg0:.+]]: i32, %[[arg1:.+]]: memref<?xf32>, %[[arg2:.+]]: i32, %[[arg3:.+]]: memref<?xf32>, %[[arg4:.+]]: i32, %[[arg5:.+]]: memref<?xf32>, %[[arg6:.+]]: memref<?xf32>)
// CHECK-NEXT:    %[[c16:.+]] = arith.constant 16 : index
// CHECK-NEXT:    %[[c0_i32:.+]] = arith.constant 0 : i32
// CHECK-NEXT:    %[[cst:.+]] = arith.constant 3.000000e-01 : f64
// CHECK-NEXT:    %[[c1:.+]] = arith.constant 1 : index
// CHECK-NEXT:    %[[c0:.+]] = arith.constant 0 : index
// CHECK-NEXT:    %[[V0:.+]] = arith.index_cast %[[arg2]] : i32 to index
// CHECK-NEXT:    %[[V1:.+]] = arith.index_cast %[[arg0]] : i32 to index
// CHECK-NEXT:    %[[V2:.+]] = arith.addi %[[V0]], %[[c1]] : index
// CHECK-NEXT:    %[[V3:.+]] = arith.muli %[[V2]], %[[c16]] : index
// CHECK-NEXT:    scf.parallel (%[[arg7:.+]]) = (%[[c0]]) to (%[[V1]]) step (%[[c1]]) {
// CHECK-NEXT:      scf.parallel (%[[arg8:.+]], %[[arg9:.+]]) = (%[[c0]], %[[c0]]) to (%[[c16]], %[[c16]]) step (%[[c1]], %[[c1]]) {
// CHECK-NEXT:        %[[V4:.+]] = arith.muli %[[V3]], %[[arg7]] : index
// CHECK-NEXT:        %[[V5:.+]] = arith.muli %[[V2]], %[[arg9]] : index
// CHECK-NEXT:        %[[V6:.+]] = arith.addi %[[V4]], %[[V5]] : index
// CHECK-NEXT:        %[[V7:.+]] = arith.addi %[[V6]], %[[arg8]] : index
// CHECK-NEXT:        %[[V8:.+]] = arith.addi %[[V7]], %[[c1]] : index
// CHECK-NEXT:        %[[V9:.+]] = arith.muli %[[arg7]], %[[c16]] : index
// CHECK-NEXT:        %[[V10:.+]] = arith.addi %[[V9]], %[[arg9]] : index
// CHECK-NEXT:        %[[V11:.+]] = arith.addi %[[V8]], %[[V2]] : index
// CHECK-NEXT:        %[[V12:.+]] = arith.addi %[[arg8]], %[[c1]] : index
// CHECK-NEXT:        %[[V13:.+]] = memref.load %[[arg1]][%[[V12]]] : memref<?xf32>
// CHECK-NEXT:        %[[V14:.+]] = arith.extf %[[V13]] : f32 to f64
// CHECK-NEXT:        %[[V15:.+]] = arith.mulf %[[V14]], %[[cst]] : f64
// CHECK-NEXT:        %[[V16:.+]] = arith.addi %[[V10]], %[[c1]] : index
// CHECK-NEXT:        %[[V17:.+]] = memref.load %[[arg3]][%[[V16]]] : memref<?xf32>
// CHECK-NEXT:        %[[V18:.+]] = arith.extf %[[V17]] : f32 to f64
// CHECK-NEXT:        %[[V19:.+]] = arith.mulf %[[V15]], %[[V18]] : f64
// CHECK-NEXT:        %[[V20:.+]] = memref.load %[[arg6]][%[[V11]]] : memref<?xf32>
// CHECK-NEXT:        %[[V21:.+]] = arith.extf %[[V20]] : f32 to f64
// CHECK-NEXT:        %[[V22:.+]] = arith.mulf %[[V21]], %[[cst]] : f64
// CHECK-NEXT:        %[[V23:.+]] = arith.addf %[[V19]], %[[V22]] : f64
// CHECK-NEXT:        %[[V24:.+]] = memref.load %[[arg5]][%[[V11]]] : memref<?xf32>
// CHECK-NEXT:        %[[V25:.+]] = arith.truncf %[[V23]] : f64 to f32
// CHECK-NEXT:        %[[V26:.+]] = arith.addf %[[V24]], %[[V25]] : f32
// CHECK-NEXT:        memref.store %[[V26]], %[[arg5]][%[[V11]]] : memref<?xf32>
// CHECK-NEXT:        %[[V27:.+]] = memref.load %[[arg1]][%[[V12]]] : memref<?xf32>
// CHECK-NEXT:        %[[V28:.+]] = arith.extf %[[V27]] : f32 to f64
// CHECK-NEXT:        %[[V29:.+]] = arith.mulf %[[V28]], %[[cst]] : f64
// CHECK-NEXT:        %[[V30:.+]] = memref.load %[[arg3]][%[[V16]]] : memref<?xf32>
// CHECK-NEXT:        %[[V31:.+]] = arith.extf %[[V30]] : f32 to f64
// CHECK-NEXT:        %[[V32:.+]] = arith.mulf %[[V29]], %[[V31]] : f64
// CHECK-NEXT:        %[[V33:.+]] = memref.load %[[arg6]][%[[V11]]] : memref<?xf32>
// CHECK-NEXT:        %[[V34:.+]] = arith.extf %[[V33]] : f32 to f64
// CHECK-NEXT:        %[[V35:.+]] = arith.mulf %[[V34]], %[[cst]] : f64
// CHECK-NEXT:        %[[V36:.+]] = arith.addf %[[V32]], %[[V35]] : f64
// CHECK-NEXT:        %[[V37:.+]] = arith.truncf %[[V36]] : f64 to f32
// CHECK-NEXT:        memref.store %[[V37]], %[[arg6]][%[[V11]]] : memref<?xf32>
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.parallel (%[[arg8:.+]], %[[arg9:.+]]) = (%[[c0]], %[[c0]]) to (%[[c16]], %[[c16]]) step (%[[c1]], %[[c1]]) {
// CHECK-NEXT:        %[[V4:.+]] = arith.addi %[[arg8]], %[[c1]] : index
// CHECK-NEXT:        %[[V5:.+]] = arith.index_cast %[[arg9]] : index to i32
// CHECK-NEXT:        %[[V6:.+]] = arith.index_cast %[[arg7]] : index to i32
// CHECK-NEXT:        %[[V7:.+]] = arith.cmpi eq, %[[V5]], %[[c0_i32]] : i32
// CHECK-NEXT:        scf.if %[[V7]] {
// CHECK-NEXT:          %[[V8:.+]] = arith.cmpi eq, %[[V6]], %[[c0_i32]] : i32
// CHECK-NEXT:          scf.if %[[V8]] {
// CHECK-NEXT:            %[[V9:.+]] = memref.load %[[arg1]][%[[V4]]] : memref<?xf32>
// CHECK-NEXT:            %[[V10:.+]] = arith.extf %[[V9]] : f32 to f64
// CHECK-NEXT:            %[[V11:.+]] = arith.mulf %[[V10]], %[[cst]] : f64
// CHECK-NEXT:            %[[V12:.+]] = memref.load %[[arg6]][%[[V4]]] : memref<?xf32>
// CHECK-NEXT:            %[[V13:.+]] = arith.extf %[[V12]] : f32 to f64
// CHECK-NEXT:            %[[V14:.+]] = arith.mulf %[[V13]], %[[cst]] : f64
// CHECK-NEXT:            %[[V15:.+]] = arith.addf %[[V11]], %[[V14]] : f64
// CHECK-NEXT:            %[[V16:.+]] = memref.load %[[arg5]][%[[V4]]] : memref<?xf32>
// CHECK-NEXT:            %[[V17:.+]] = arith.truncf %[[V15]] : f64 to f32
// CHECK-NEXT:            %[[V18:.+]] = arith.addf %[[V16]], %[[V17]] : f32
// CHECK-NEXT:            memref.store %[[V18]], %[[arg5]][%[[V4]]] : memref<?xf32>
// CHECK-NEXT:            %[[V19:.+]] = memref.load %[[arg1]][%[[V4]]] : memref<?xf32>
// CHECK-NEXT:            %[[V20:.+]] = arith.extf %[[V19]] : f32 to f64
// CHECK-NEXT:            %[[V21:.+]] = arith.mulf %[[V20]], %[[cst]] : f64
// CHECK-NEXT:            %[[V22:.+]] = memref.load %[[arg6]][%[[V4]]] : memref<?xf32>
// CHECK-NEXT:            %[[V23:.+]] = arith.extf %[[V22]] : f32 to f64
// CHECK-NEXT:            %[[V24:.+]] = arith.mulf %[[V23]], %[[cst]] : f64
// CHECK-NEXT:            %[[V25:.+]] = arith.addf %[[V21]], %[[V24]] : f64
// CHECK-NEXT:            %[[V26:.+]] = arith.truncf %[[V25]] : f64 to f32
// CHECK-NEXT:            memref.store %[[V26]], %[[arg6]][%[[V4]]] : memref<?xf32>
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }


// CHECK:   func.func @_Z30bpnn_layerforward_CUDA_wrapperiPfS_S_S_ii(%[[arg0:.+]]: i32, %[[arg1:.+]]: memref<?xf32>, %[[arg2:.+]]: memref<?xf32>, %[[arg3:.+]]: memref<?xf32>, %[[arg4:.+]]: memref<?xf32>, %[[arg5:.+]]: i32, %[[arg6:.+]]: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:    %[[c16:.+]] = arith.constant 16 : index
// CHECK-NEXT:    %[[cst:.+]] = arith.constant 2.000000e+00 : f32
// CHECK-NEXT:    %[[cst_0:.+]] = arith.constant 4.000000e+00 : f32
// CHECK-NEXT:    %[[c0_i32:.+]] = arith.constant 0 : i32
// CHECK-NEXT:    %[[c1_i32:.+]] = arith.constant 1 : i32
// CHECK-NEXT:    %[[c1:.+]] = arith.constant 1 : index
// CHECK-NEXT:    %[[c0:.+]] = arith.constant 0 : index
// CHECK-NEXT:    %[[c2:.+]] = arith.constant 2 : index
// CHECK-NEXT:    %[[V0:.+]] = arith.index_cast %[[arg6]] : i32 to index
// CHECK-NEXT:    %[[V1:.+]] = arith.index_cast %[[arg0]] : i32 to index
// CHECK-NEXT:    %[[V2:.+]] = arith.addi %[[V0]], %[[c1]] : index
// CHECK-NEXT:    %[[V3:.+]] = arith.muli %[[V2]], %[[c16]] : index
// CHECK-NEXT:    scf.parallel (%[[arg7:.+]]) = (%[[c0]]) to (%[[V1]]) step (%[[c1]]) {
// CHECK-NEXT:      %[[V4:.+]] = memref.alloca() : memref<16xf32>
// CHECK-NEXT:      %[[V5:.+]] = memref.alloca() : memref<16x16xf32>
// CHECK-NEXT:      %[[V6:.+]] = arith.muli %[[V3]], %[[arg7]] : index
// CHECK-NEXT:      %[[V7:.+]] = arith.muli %[[arg7]], %[[c16]] : index
// CHECK-NEXT:      memref.alloca_scope  {
// CHECK-NEXT:        scf.parallel (%[[arg8:.+]], %[[arg9:.+]]) = (%[[c0]], %[[c0]]) to (%[[c16]], %[[c16]]) step (%[[c1]], %[[c1]]) {
// CHECK-NEXT:          %[[V8:.+]] = arith.index_cast %[[arg8]] : index to i32
// CHECK-NEXT:          %[[V9:.+]] = arith.addi %[[V7]], %[[arg9]] : index
// CHECK-NEXT:          %[[V10:.+]] = arith.cmpi eq, %[[V8]], %[[c0_i32]] : i32
// CHECK-NEXT:          scf.if %[[V10]] {
// CHECK-NEXT:            %[[V11:.+]] = arith.addi %[[V9]], %[[c1]] : index
// CHECK-NEXT:            %[[V12:.+]] = memref.load %[[arg1]][%[[V11]]] : memref<?xf32>
// CHECK-NEXT:            memref.store %[[V12]], %[[V4]][%[[arg9]]] : memref<16xf32>
// CHECK-NEXT:          }
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }
// CHECK-NEXT:        memref.alloca_scope  {
// CHECK-NEXT:          %[[V8:.+]] = memref.alloca(%[[c16]], %[[c16]]) : memref<?x?xi32>
// CHECK-NEXT:          %[[V9:.+]] = memref.alloca(%[[c16]], %[[c16]]) : memref<?x?xi32>
// CHECK-NEXT:          memref.alloca_scope  {
// CHECK-NEXT:            scf.parallel (%[[arg8:.+]], %[[arg9:.+]]) = (%[[c0]], %[[c0]]) to (%[[c16]], %[[c16]]) step (%[[c1]], %[[c1]]) {
// CHECK-NEXT:              %[[V10:.+]] = arith.muli %[[V2]], %[[arg9]] : index
// CHECK-NEXT:              %[[V11:.+]] = arith.addi %[[V6]], %[[V10]] : index
// CHECK-NEXT:              %[[V12:.+]] = arith.addi %[[V11]], %[[arg8]] : index
// CHECK-NEXT:              %[[V13:.+]] = arith.addi %[[V12]], %[[c1]] : index
// CHECK-NEXT:              %[[V14:.+]] = arith.addi %[[V13]], %[[V2]] : index
// CHECK-NEXT:              %[[V15:.+]] = memref.load %[[arg3]][%[[V14]]] : memref<?xf32>
// CHECK-NEXT:              memref.store %[[V15]], %[[V5]][%[[arg9]], %[[arg8]]] : memref<16x16xf32>
// CHECK-NEXT:              scf.yield
// CHECK-NEXT:            }
// CHECK-NEXT:            scf.parallel (%[[arg8:.+]], %[[arg9:.+]]) = (%[[c0]], %[[c0]]) to (%[[c16]], %[[c16]]) step (%[[c1]], %[[c1]]) {
// CHECK-NEXT:              %[[V10:.+]] = memref.load %[[V5]][%[[arg9]], %[[arg8]]] : memref<16x16xf32>
// CHECK-NEXT:              %[[V11:.+]] = memref.load %[[V4]][%[[arg9]]] : memref<16xf32>
// CHECK-NEXT:              %[[V12:.+]] = arith.mulf %[[V10]], %[[V11]] : f32
// CHECK-NEXT:              memref.store %[[V12]], %[[V5]][%[[arg9]], %[[arg8]]] : memref<16x16xf32>
// CHECK-NEXT:              %[[V13:.+]] = "polygeist.subindex"(%[[V8]], %[[arg8]]) : (memref<?x?xi32>, index) -> memref<?xi32>
// CHECK-NEXT:              %[[V14:.+]] = "polygeist.subindex"(%[[V13]], %[[arg9]]) : (memref<?xi32>, index) -> memref<i32>
// CHECK-NEXT:              memref.store %[[c1_i32]], %[[V14]][] : memref<i32>
// CHECK-NEXT:              scf.yield
// CHECK-NEXT:            }
// CHECK-NEXT:          }
// CHECK-NEXT:          memref.alloca_scope  {
// CHECK-NEXT:            scf.while : () -> () {
// CHECK-NEXT:              %[[V10:.+]] = memref.alloca() : memref<i1>
// CHECK-NEXT:              scf.parallel (%[[arg8:.+]], %[[arg9:.+]]) = (%[[c0]], %[[c0]]) to (%[[c16]], %[[c16]]) step (%[[c1]], %[[c1]]) {
// CHECK-NEXT:                %[[V12:.+]] = "polygeist.subindex"(%[[V8]], %[[arg8]]) : (memref<?x?xi32>, index) -> memref<?xi32>
// CHECK-NEXT:                %[[V13:.+]] = "polygeist.subindex"(%[[V12]], %[[arg9]]) : (memref<?xi32>, index) -> memref<i32>
// CHECK-NEXT:                %[[V14:.+]] = memref.load %[[V13]][] : memref<i32>
// CHECK-NEXT:                %[[V15:.+]] = arith.sitofp %[[V14]] : i32 to f32
// CHECK-NEXT:                %[[V16:.+]] = arith.cmpf ule, %[[V15]], %[[cst_0]] : f32
// CHECK-NEXT:                %[[V17:.+]] = arith.cmpi eq, %[[arg8]], %[[c0]] : index
// CHECK-NEXT:                %[[V18:.+]] = arith.cmpi eq, %[[arg9]], %[[c0]] : index
// CHECK-NEXT:                %[[V19:.+]] = arith.andi %[[V18]], %[[V17]] : i1
// CHECK-NEXT:                scf.if %[[V19]] {
// CHECK-NEXT:                  memref.store %[[V16]], %[[V10]][] : memref<i1>
// CHECK-NEXT:                }
// CHECK-NEXT:                %[[V20:.+]] = "polygeist.subindex"(%[[V9]], %[[arg8]]) : (memref<?x?xi32>, index) -> memref<?xi32>
// CHECK-NEXT:                %[[V21:.+]] = "polygeist.subindex"(%[[V20]], %[[arg9]]) : (memref<?xi32>, index) -> memref<i32>
// CHECK-NEXT:                memref.store %[[V14]], %[[V21]][] : memref<i32>
// CHECK-NEXT:                scf.yield
// CHECK-NEXT:              }
// CHECK-NEXT:              %[[V11:.+]] = memref.load %[[V10]][] : memref<i1>
// CHECK-NEXT:              scf.condition(%[[V11]])
// CHECK-NEXT:            } do {
// CHECK-NEXT:              memref.alloca_scope  {
// CHECK-NEXT:                scf.parallel (%[[arg8:.+]], %[[arg9:.+]]) = (%[[c0]], %[[c0]]) to (%[[c16]], %[[c16]]) step (%[[c1]], %[[c1]]) {
// CHECK-NEXT:                  %[[V10:.+]] = arith.index_cast %[[arg9]] : index to i32
// CHECK-NEXT:                  %[[V11:.+]] = "polygeist.subindex"(%[[V9]], %[[arg8]]) : (memref<?x?xi32>, index) -> memref<?xi32>
// CHECK-NEXT:                  %[[V12:.+]] = "polygeist.subindex"(%[[V11]], %[[arg9]]) : (memref<?xi32>, index) -> memref<i32>
// CHECK-NEXT:                  %[[V13:.+]] = memref.load %[[V12]][] : memref<i32>
// CHECK-NEXT:                  %[[V14:.+]] = arith.sitofp %[[V13]] : i32 to f32
// CHECK-NEXT:                  %[[V15:.+]] = math.powf %[[cst]], %[[V14]] : f32
// CHECK-NEXT:                  %[[V16:.+]] = arith.fptosi %[[V15]] : f32 to i32
// CHECK-NEXT:                  %[[V17:.+]] = arith.index_cast %[[V16]] : i32 to index
// CHECK-NEXT:                  %[[V18:.+]] = arith.remsi %[[V10]], %[[V16]] : i32
// CHECK-NEXT:                  %[[V19:.+]] = arith.cmpi eq, %[[V18]], %[[c0_i32]] : i32
// CHECK-NEXT:                  scf.if %[[V19]] {
// CHECK-NEXT:                    %[[V20:.+]] = memref.load %[[V5]][%[[arg9]], %[[arg8]]] : memref<16x16xf32>
// CHECK-NEXT:                    %[[V21:.+]] = arith.divsi %[[V17]], %[[c2]] : index
// CHECK-NEXT:                    %[[V22:.+]] = arith.addi %[[arg9]], %[[V21]] : index
// CHECK-NEXT:                    %[[V23:.+]] = memref.load %[[V5]][%[[V22]], %[[arg8]]] : memref<16x16xf32>
// CHECK-NEXT:                    %[[V24:.+]] = arith.addf %[[V20]], %[[V23]] : f32
// CHECK-NEXT:                    memref.store %[[V24]], %[[V5]][%[[arg9]], %[[arg8]]] : memref<16x16xf32>
// CHECK-NEXT:                  }
// CHECK-NEXT:                  scf.yield
// CHECK-NEXT:                }
// CHECK-NEXT:                scf.parallel (%[[arg8:.+]], %[[arg9:.+]]) = (%[[c0]], %[[c0]]) to (%[[c16]], %[[c16]]) step (%[[c1]], %[[c1]]) {
// CHECK-NEXT:                  %[[V10:.+]] = "polygeist.subindex"(%[[V9]], %[[arg8]]) : (memref<?x?xi32>, index) -> memref<?xi32>
// CHECK-NEXT:                  %[[V11:.+]] = "polygeist.subindex"(%[[V10]], %[[arg9]]) : (memref<?xi32>, index) -> memref<i32>
// CHECK-NEXT:                  %[[V12:.+]] = memref.load %[[V11]][] : memref<i32>
// CHECK-NEXT:                  %[[V13:.+]] = arith.addi %[[V12]], %[[c1_i32]] : i32
// CHECK-NEXT:                  %[[V14:.+]] = "polygeist.subindex"(%[[V8]], %[[arg8]]) : (memref<?x?xi32>, index) -> memref<?xi32>
// CHECK-NEXT:                  %[[V15:.+]] = "polygeist.subindex"(%[[V14]], %[[arg9]]) : (memref<?xi32>, index) -> memref<i32>
// CHECK-NEXT:                  memref.store %[[V13]], %[[V15]][] : memref<i32>
// CHECK-NEXT:                  scf.yield
// CHECK-NEXT:                }
// CHECK-NEXT:              }
// CHECK-NEXT:              scf.yield
// CHECK-NEXT:            }
// CHECK-NEXT:            memref.alloca_scope  {
// CHECK-NEXT:              scf.parallel (%[[arg8:.+]], %[[arg9:.+]]) = (%[[c0]], %[[c0]]) to (%[[c16]], %[[c16]]) step (%[[c1]], %[[c1]]) {
// CHECK-NEXT:                %[[V10:.+]] = arith.muli %[[V2]], %[[arg9]] : index
// CHECK-NEXT:                %[[V11:.+]] = arith.addi %[[V6]], %[[V10]] : index
// CHECK-NEXT:                %[[V12:.+]] = arith.addi %[[V11]], %[[arg8]] : index
// CHECK-NEXT:                %[[V13:.+]] = arith.addi %[[V12]], %[[c1]] : index
// CHECK-NEXT:                %[[V14:.+]] = arith.addi %[[V13]], %[[V2]] : index
// CHECK-NEXT:                %[[V15:.+]] = memref.load %[[V5]][%[[arg9]], %[[arg8]]] : memref<16x16xf32>
// CHECK-NEXT:                memref.store %[[V15]], %[[arg3]][%[[V14]]] : memref<?xf32>
// CHECK-NEXT:                scf.yield
// CHECK-NEXT:              }
// CHECK-NEXT:              scf.parallel (%[[arg8:.+]], %[[arg9:.+]]) = (%[[c0]], %[[c0]]) to (%[[c16]], %[[c16]]) step (%[[c1]], %[[c1]]) {
// CHECK-NEXT:                %[[V10:.+]] = arith.index_cast %[[arg8]] : index to i32
// CHECK-NEXT:                %[[V11:.+]] = arith.cmpi eq, %[[V10]], %[[c0_i32]] : i32
// CHECK-NEXT:                scf.if %[[V11]] {
// CHECK-NEXT:                  %[[V12:.+]] = memref.load %[[V5]][%[[arg8]], %[[arg9]]] : memref<16x16xf32>
// CHECK-NEXT:                  %[[V13:.+]] = arith.muli %[[arg7]], %[[V0]] : index
// CHECK-NEXT:                  %[[V14:.+]] = arith.addi %[[arg9]], %[[V13]] : index
// CHECK-NEXT:                  memref.store %[[V12]], %[[arg4]][%[[V14]]] : memref<?xf32>
// CHECK-NEXT:                }
// CHECK-NEXT:                scf.yield
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
