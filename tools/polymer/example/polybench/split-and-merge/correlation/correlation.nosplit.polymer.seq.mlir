#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 * 32)>
#map2 = affine_map<(d0, d1) -> (2599, d0 * 32 + 32, d1 * 32 + 31)>
#map3 = affine_map<(d0, d1) -> (d0 * 32, d1 + 1)>
#map4 = affine_map<(d0) -> (2600, d0 * 32 + 32)>
#map5 = affine_map<(d0) -> (2599, d0 * 32 + 32)>
#map6 = affine_map<(d0) -> (3000, d0 * 32 + 32)>
#set = affine_set<(d0) : (d0 - 81 >= 0)>
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  llvm.mlir.global internal constant @str7("==END   DUMP_ARRAYS==\0A\00")
  llvm.mlir.global internal constant @str6("\0Aend   dump: %s\0A\00")
  llvm.mlir.global internal constant @str5("%0.2lf \00")
  llvm.mlir.global internal constant @str4("\0A\00")
  llvm.mlir.global internal constant @str3("corr\00")
  llvm.mlir.global internal constant @str2("begin dump: %s\00")
  llvm.mlir.global internal constant @str1("==BEGIN DUMP_ARRAYS==\0A\00")
  llvm.mlir.global external @stderr() : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>
  llvm.func @fprintf(!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, ...) -> i32
  llvm.mlir.global internal constant @str0("\00")
  llvm.func @strcmp(!llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
  func @main(%arg0: i32, %arg1: !llvm.ptr<ptr<i8>>) -> i32 {
    %c42_i32 = constant 42 : i32
    %c0_i64 = constant 0 : i64
    %true = constant true
    %false = constant false
    %c3000_i32 = constant 3000 : i32
    %c0_i32 = constant 0 : i32
    %c2600_i32 = constant 2600 : i32
    %c1_i32 = constant 1 : i32
    %cst = constant 0.000000e+00 : f64
    %cst_0 = constant 1.000000e-01 : f64
    %cst_1 = constant 1.000000e+00 : f64
    %c2599 = constant 2599 : index
    %0 = memref.alloca() : memref<1xf64>
    %1 = memref.alloc() : memref<3000x2600xf64>
    %2 = memref.alloc() : memref<2600x2600xf64>
    %3 = memref.alloc() : memref<2600xf64>
    %4 = memref.alloc() : memref<2600xf64>
    %5 = sitofp %c3000_i32 : i32 to f64
    affine.store %5, %0[0] : memref<1xf64>
    %6 = scf.while (%arg2 = %c0_i32) : (i32) -> i32 {
      %11 = cmpi slt, %arg2, %c3000_i32 : i32
      scf.condition(%11) %arg2 : i32
    } do {
    ^bb0(%arg2: i32):  // no predecessors
      %11 = index_cast %arg2 : i32 to index
      %12 = scf.while (%arg3 = %c0_i32) : (i32) -> i32 {
        %14 = cmpi slt, %arg3, %c2600_i32 : i32
        scf.condition(%14) %arg3 : i32
      } do {
      ^bb0(%arg3: i32):  // no predecessors
        %14 = index_cast %arg3 : i32 to index
        %15 = muli %arg2, %arg3 : i32
        %16 = sitofp %15 : i32 to f64
        %17 = sitofp %c2600_i32 : i32 to f64
        %18 = divf %16, %17 : f64
        %19 = sitofp %arg2 : i32 to f64
        %20 = addf %18, %19 : f64
        memref.store %20, %1[%11, %14] : memref<3000x2600xf64>
        %21 = addi %arg3, %c1_i32 : i32
        scf.yield %21 : i32
      }
      %13 = addi %arg2, %c1_i32 : i32
      scf.yield %13 : i32
    }
    call @polybench_timer_start() : () -> ()
    %7 = affine.load %0[0] : memref<1xf64>
    %8 = memref.cast %2 : memref<2600x2600xf64> to memref<?x2600xf64>
    memref.store %cst_1, %2[%c2599, %c2599] : memref<2600x2600xf64>
    affine.for %arg2 = 0 to 82 {
      affine.for %arg3 = #map0(%arg2) to 82 {
        affine.for %arg4 = #map1(%arg2) to min #map2(%arg2, %arg3) {
          affine.for %arg5 = max #map3(%arg3, %arg4) to min #map4(%arg3) {
            affine.store %cst, %2[%arg4, %arg5] : memref<2600x2600xf64>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 82 {
      affine.for %arg3 = #map1(%arg2) to min #map5(%arg2) {
        affine.store %cst, %3[%arg3] : memref<2600xf64>
        affine.store %cst, %4[%arg3] : memref<2600xf64>
        affine.store %cst_1, %2[%arg3, %arg3] : memref<2600x2600xf64>
      }
      affine.if #set(%arg2) {
        affine.store %cst, %3[2599] : memref<2600xf64>
        affine.store %cst, %4[2599] : memref<2600xf64>
      }
    }
    affine.for %arg2 = 0 to 82 {
      affine.for %arg3 = 0 to 94 {
        affine.for %arg4 = #map1(%arg3) to min #map6(%arg3) {
          affine.for %arg5 = #map1(%arg2) to min #map4(%arg2) {
            %11 = affine.load %3[%arg5] : memref<2600xf64>
            %12 = affine.load %1[%arg4, %arg5] : memref<3000x2600xf64>
            %13 = addf %11, %12 : f64
            affine.store %13, %3[%arg5] : memref<2600xf64>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 82 {
      affine.for %arg3 = #map1(%arg2) to min #map4(%arg2) {
        %11 = affine.load %3[%arg3] : memref<2600xf64>
        %12 = divf %11, %7 : f64
        affine.store %12, %3[%arg3] : memref<2600xf64>
      }
    }
    affine.for %arg2 = 0 to 82 {
      affine.for %arg3 = 0 to 94 {
        affine.for %arg4 = #map1(%arg3) to min #map6(%arg3) {
          affine.for %arg5 = #map1(%arg2) to min #map4(%arg2) {
            %11 = affine.load %4[%arg5] : memref<2600xf64>
            %12 = affine.load %1[%arg4, %arg5] : memref<3000x2600xf64>
            %13 = affine.load %3[%arg5] : memref<2600xf64>
            %14 = subf %12, %13 : f64
            %15 = mulf %14, %14 : f64
            %16 = addf %11, %15 : f64
            affine.store %16, %4[%arg5] : memref<2600xf64>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 82 {
      affine.for %arg3 = #map1(%arg2) to min #map4(%arg2) {
        %11 = affine.load %4[%arg3] : memref<2600xf64>
        %12 = divf %11, %7 : f64
        %13 = math.sqrt %12 : f64
        %14 = cmpf ule, %13, %cst_0 : f64
        %15 = select %14, %cst_1, %13 : f64
        affine.store %15, %4[%arg3] : memref<2600xf64>
      }
    }
    affine.for %arg2 = 0 to 94 {
      affine.for %arg3 = 0 to 82 {
        affine.for %arg4 = #map1(%arg2) to min #map6(%arg2) {
          affine.for %arg5 = #map1(%arg3) to min #map4(%arg3) {
            %11 = affine.load %1[%arg4, %arg5] : memref<3000x2600xf64>
            %12 = affine.load %3[%arg5] : memref<2600xf64>
            %13 = subf %11, %12 : f64
            %14 = math.sqrt %7 : f64
            %15 = affine.load %4[%arg5] : memref<2600xf64>
            %16 = mulf %14, %15 : f64
            %17 = divf %13, %16 : f64
            affine.store %17, %1[%arg4, %arg5] : memref<3000x2600xf64>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 82 {
      affine.for %arg3 = #map0(%arg2) to 82 {
        affine.for %arg4 = 0 to 94 {
          affine.for %arg5 = #map1(%arg2) to min #map2(%arg2, %arg3) {
            affine.for %arg6 = #map1(%arg4) to min #map6(%arg4) {
              affine.for %arg7 = max #map3(%arg3, %arg5) to min #map4(%arg3) {
                %11 = affine.load %2[%arg5, %arg7] : memref<2600x2600xf64>
                %12 = affine.load %1[%arg6, %arg5] : memref<3000x2600xf64>
                %13 = affine.load %1[%arg6, %arg7] : memref<3000x2600xf64>
                %14 = mulf %12, %13 : f64
                %15 = addf %11, %14 : f64
                affine.store %15, %2[%arg5, %arg7] : memref<2600x2600xf64>
              }
            }
          }
        }
      }
    }
    affine.for %arg2 = 0 to 82 {
      affine.for %arg3 = #map0(%arg2) to 82 {
        affine.for %arg4 = #map1(%arg2) to min #map2(%arg2, %arg3) {
          affine.for %arg5 = max #map3(%arg3, %arg4) to min #map4(%arg3) {
            %11 = affine.load %2[%arg4, %arg5] : memref<2600x2600xf64>
            affine.store %11, %2[%arg5, %arg4] : memref<2600x2600xf64>
          }
        }
      }
    }
    call @polybench_timer_stop() : () -> ()
    call @polybench_timer_print() : () -> ()
    %9 = cmpi sgt, %arg0, %c42_i32 : i32
    %10 = scf.if %9 -> (i1) {
      %11 = llvm.getelementptr %arg1[%c0_i64] : (!llvm.ptr<ptr<i8>>, i64) -> !llvm.ptr<ptr<i8>>
      %12 = llvm.load %11 : !llvm.ptr<ptr<i8>>
      %13 = llvm.mlir.addressof @str0 : !llvm.ptr<array<1 x i8>>
      %14 = llvm.getelementptr %13[%c0_i64, %c0_i64] : (!llvm.ptr<array<1 x i8>>, i64, i64) -> !llvm.ptr<i8>
      %15 = llvm.call @strcmp(%12, %14) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
      %16 = trunci %15 : i32 to i1
      %17 = xor %16, %true : i1
      scf.yield %17 : i1
    } else {
      scf.yield %false : i1
    }
    scf.if %10 {
      call @print_array(%c2600_i32, %8) : (i32, memref<?x2600xf64>) -> ()
    }
    memref.dealloc %1 : memref<3000x2600xf64>
    memref.dealloc %2 : memref<2600x2600xf64>
    memref.dealloc %3 : memref<2600xf64>
    memref.dealloc %4 : memref<2600xf64>
    return %c0_i32 : i32
  }
  func private @polybench_timer_start()
  func private @polybench_timer_stop()
  func private @polybench_timer_print()
  func private @S0(%arg0: memref<?xf64>, %arg1: index) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f64
    affine.store %cst, %arg0[symbol(%arg1)] : memref<?xf64>
    return
  }
  func private @S1(%arg0: memref<?xf64>, %arg1: index, %arg2: memref<?x2600xf64>, %arg3: index) attributes {scop.stmt} {
    %0 = affine.load %arg0[symbol(%arg1)] : memref<?xf64>
    %1 = affine.load %arg2[symbol(%arg3), symbol(%arg1)] : memref<?x2600xf64>
    %2 = addf %0, %1 : f64
    affine.store %2, %arg0[symbol(%arg1)] : memref<?xf64>
    return
  }
  func private @S2(%arg0: memref<?xf64>, %arg1: index, %arg2: f64) attributes {scop.stmt} {
    %0 = affine.load %arg0[symbol(%arg1)] : memref<?xf64>
    %1 = divf %0, %arg2 : f64
    affine.store %1, %arg0[symbol(%arg1)] : memref<?xf64>
    return
  }
  func private @S3(%arg0: memref<?xf64>, %arg1: index) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f64
    affine.store %cst, %arg0[symbol(%arg1)] : memref<?xf64>
    return
  }
  func private @S4(%arg0: memref<?xf64>, %arg1: index, %arg2: memref<?xf64>, %arg3: memref<?x2600xf64>, %arg4: index) attributes {scop.stmt} {
    %0 = affine.load %arg0[symbol(%arg1)] : memref<?xf64>
    %1 = affine.load %arg3[symbol(%arg4), symbol(%arg1)] : memref<?x2600xf64>
    %2 = affine.load %arg2[symbol(%arg1)] : memref<?xf64>
    %3 = subf %1, %2 : f64
    %4 = mulf %3, %3 : f64
    %5 = addf %0, %4 : f64
    affine.store %5, %arg0[symbol(%arg1)] : memref<?xf64>
    return
  }
  func private @S5(%arg0: memref<?xf64>, %arg1: index, %arg2: f64) attributes {scop.stmt} {
    %cst = constant 1.000000e-01 : f64
    %cst_0 = constant 1.000000e+00 : f64
    %0 = affine.load %arg0[symbol(%arg1)] : memref<?xf64>
    %1 = divf %0, %arg2 : f64
    %2 = math.sqrt %1 : f64
    %3 = cmpf ule, %2, %cst : f64
    %4 = select %3, %cst_0, %2 : f64
    affine.store %4, %arg0[symbol(%arg1)] : memref<?xf64>
    return
  }
  func private @S6(%arg0: memref<?x2600xf64>, %arg1: index, %arg2: index, %arg3: memref<?xf64>, %arg4: f64, %arg5: memref<?xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x2600xf64>
    %1 = affine.load %arg5[symbol(%arg2)] : memref<?xf64>
    %2 = subf %0, %1 : f64
    %3 = math.sqrt %arg4 : f64
    %4 = affine.load %arg3[symbol(%arg2)] : memref<?xf64>
    %5 = mulf %3, %4 : f64
    %6 = divf %2, %5 : f64
    affine.store %6, %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x2600xf64>
    return
  }
  func private @S7(%arg0: memref<?x2600xf64>, %arg1: index) attributes {scop.stmt} {
    %cst = constant 1.000000e+00 : f64
    affine.store %cst, %arg0[symbol(%arg1), symbol(%arg1)] : memref<?x2600xf64>
    return
  }
  func private @S8(%arg0: memref<?x2600xf64>, %arg1: index, %arg2: index) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f64
    affine.store %cst, %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x2600xf64>
    return
  }
  func private @S9(%arg0: memref<?x2600xf64>, %arg1: index, %arg2: index, %arg3: memref<?x2600xf64>, %arg4: index) attributes {scop.stmt} {
    %0 = affine.load %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x2600xf64>
    %1 = affine.load %arg3[symbol(%arg4), symbol(%arg1)] : memref<?x2600xf64>
    %2 = affine.load %arg3[symbol(%arg4), symbol(%arg2)] : memref<?x2600xf64>
    %3 = mulf %1, %2 : f64
    %4 = addf %0, %3 : f64
    affine.store %4, %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x2600xf64>
    return
  }
  func private @S10(%arg0: memref<?x2600xf64>, %arg1: index, %arg2: index) attributes {scop.stmt} {
    %0 = affine.load %arg0[symbol(%arg2), symbol(%arg1)] : memref<?x2600xf64>
    affine.store %0, %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x2600xf64>
    return
  }
  func private @S11(%arg0: memref<?x2600xf64>, %arg1: index) attributes {scop.stmt} {
    %c1 = constant 1 : index
    %cst = constant 1.000000e+00 : f64
    %0 = subi %arg1, %c1 : index
    memref.store %cst, %arg0[%0, %0] : memref<?x2600xf64>
    return
  }
  func private @print_array(%arg0: i32, %arg1: memref<?x2600xf64>) {
    %c0_i64 = constant 0 : i64
    %c0_i32 = constant 0 : i32
    %c20_i32 = constant 20 : i32
    %c1_i32 = constant 1 : i32
    %0 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %1 = llvm.load %0 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %2 = llvm.mlir.addressof @str1 : !llvm.ptr<array<23 x i8>>
    %3 = llvm.getelementptr %2[%c0_i64, %c0_i64] : (!llvm.ptr<array<23 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %4 = llvm.call @fprintf(%1, %3) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> i32
    %5 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %6 = llvm.load %5 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %7 = llvm.mlir.addressof @str2 : !llvm.ptr<array<15 x i8>>
    %8 = llvm.getelementptr %7[%c0_i64, %c0_i64] : (!llvm.ptr<array<15 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %9 = llvm.mlir.addressof @str3 : !llvm.ptr<array<5 x i8>>
    %10 = llvm.getelementptr %9[%c0_i64, %c0_i64] : (!llvm.ptr<array<5 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %11 = llvm.call @fprintf(%6, %8, %10) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
    %12 = scf.while (%arg2 = %c0_i32) : (i32) -> i32 {
      %25 = cmpi slt, %arg2, %arg0 : i32
      scf.condition(%25) %arg2 : i32
    } do {
    ^bb0(%arg2: i32):  // no predecessors
      %25 = index_cast %arg2 : i32 to index
      %26 = scf.while (%arg3 = %c0_i32) : (i32) -> i32 {
        %28 = cmpi slt, %arg3, %arg0 : i32
        scf.condition(%28) %arg3 : i32
      } do {
      ^bb0(%arg3: i32):  // no predecessors
        %28 = index_cast %arg3 : i32 to index
        %29 = muli %arg2, %arg0 : i32
        %30 = addi %29, %arg3 : i32
        %31 = remi_signed %30, %c20_i32 : i32
        %32 = cmpi eq, %31, %c0_i32 : i32
        scf.if %32 {
          %40 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
          %41 = llvm.load %40 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
          %42 = llvm.mlir.addressof @str4 : !llvm.ptr<array<2 x i8>>
          %43 = llvm.getelementptr %42[%c0_i64, %c0_i64] : (!llvm.ptr<array<2 x i8>>, i64, i64) -> !llvm.ptr<i8>
          %44 = llvm.call @fprintf(%41, %43) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> i32
        }
        %33 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
        %34 = llvm.load %33 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
        %35 = llvm.mlir.addressof @str5 : !llvm.ptr<array<8 x i8>>
        %36 = llvm.getelementptr %35[%c0_i64, %c0_i64] : (!llvm.ptr<array<8 x i8>>, i64, i64) -> !llvm.ptr<i8>
        %37 = memref.load %arg1[%25, %28] : memref<?x2600xf64>
        %38 = llvm.call @fprintf(%34, %36, %37) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, f64) -> i32
        %39 = addi %arg3, %c1_i32 : i32
        scf.yield %39 : i32
      }
      %27 = addi %arg2, %c1_i32 : i32
      scf.yield %27 : i32
    }
    %13 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %14 = llvm.load %13 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %15 = llvm.mlir.addressof @str6 : !llvm.ptr<array<17 x i8>>
    %16 = llvm.getelementptr %15[%c0_i64, %c0_i64] : (!llvm.ptr<array<17 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %17 = llvm.mlir.addressof @str3 : !llvm.ptr<array<5 x i8>>
    %18 = llvm.getelementptr %17[%c0_i64, %c0_i64] : (!llvm.ptr<array<5 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %19 = llvm.call @fprintf(%14, %16, %18) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
    %20 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %21 = llvm.load %20 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %22 = llvm.mlir.addressof @str7 : !llvm.ptr<array<23 x i8>>
    %23 = llvm.getelementptr %22[%c0_i64, %c0_i64] : (!llvm.ptr<array<23 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %24 = llvm.call @fprintf(%21, %23) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> i32
    return
  }
}

