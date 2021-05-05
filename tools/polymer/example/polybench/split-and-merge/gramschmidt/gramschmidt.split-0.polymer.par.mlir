#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 * 32)>
#map2 = affine_map<(d0, d1) -> (2599, d0 * 32 + 32, d1 * 32 + 31)>
#map3 = affine_map<(d0, d1) -> (d0 * 32, d1 + 1)>
#map4 = affine_map<(d0) -> (2600, d0 * 32 + 32)>
#map5 = affine_map<(d0) -> (2000, d0 * 32 + 32)>
#map6 = affine_map<(d0) -> ((d0 - 30) ceildiv 32)>
#set = affine_set<(d0) : (-d0 + 2598 >= 0)>
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  llvm.mlir.global internal constant @str8("==END   DUMP_ARRAYS==\0A\00")
  llvm.mlir.global internal constant @str7("Q\00")
  llvm.mlir.global internal constant @str6("\0Aend   dump: %s\0A\00")
  llvm.mlir.global internal constant @str5("%0.2lf \00")
  llvm.mlir.global internal constant @str4("\0A\00")
  llvm.mlir.global internal constant @str3("R\00")
  llvm.mlir.global internal constant @str2("begin dump: %s\00")
  llvm.mlir.global internal constant @str1("==BEGIN DUMP_ARRAYS==\0A\00")
  llvm.mlir.global external @stderr() : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>
  llvm.func @fprintf(!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, ...) -> i32
  llvm.mlir.global internal constant @str0("\00")
  llvm.func @strcmp(!llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
  func @main(%arg0: i32, %arg1: !llvm.ptr<ptr<i8>>) -> i32 {
    %c2000_i32 = constant 2000 : i32
    %c2600_i32 = constant 2600 : i32
    %c42_i32 = constant 42 : i32
    %c0_i64 = constant 0 : i64
    %true = constant true
    %false = constant false
    %c100_i32 = constant 100 : i32
    %c10_i32 = constant 10 : i32
    %c1_i32 = constant 1 : i32
    %c0_i32 = constant 0 : i32
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c2600 = constant 2600 : index
    %cst = constant 0.000000e+00 : f64
    %0 = memref.alloca() : memref<2000xf64>
    %1 = memref.alloc() : memref<2000x2600xf64>
    %2 = memref.alloc() : memref<2600x2600xf64>
    %3 = memref.alloc() : memref<2000x2600xf64>
    %4 = memref.cast %1 : memref<2000x2600xf64> to memref<?x2600xf64>
    %5 = memref.cast %2 : memref<2600x2600xf64> to memref<?x2600xf64>
    %6 = memref.cast %3 : memref<2000x2600xf64> to memref<?x2600xf64>
    %7:2 = scf.while (%arg2 = %c0_i32) : (i32) -> (i32, i32) {
      %14 = cmpi slt, %arg2, %c2000_i32 : i32
      scf.condition(%14) %c0_i32, %arg2 : i32, i32
    } do {
    ^bb0(%arg2: i32, %arg3: i32):  // no predecessors
      %14 = index_cast %arg3 : i32 to index
      %15 = scf.while (%arg4 = %c0_i32) : (i32) -> i32 {
        %17 = cmpi slt, %arg4, %c2600_i32 : i32
        scf.condition(%17) %arg4 : i32
      } do {
      ^bb0(%arg4: i32):  // no predecessors
        %17 = index_cast %arg4 : i32 to index
        %18 = muli %arg3, %arg4 : i32
        %19 = remi_signed %18, %c2000_i32 : i32
        %20 = sitofp %19 : i32 to f64
        %21 = sitofp %c2000_i32 : i32 to f64
        %22 = divf %20, %21 : f64
        %23 = sitofp %c100_i32 : i32 to f64
        %24 = mulf %22, %23 : f64
        %25 = sitofp %c10_i32 : i32 to f64
        %26 = addf %24, %25 : f64
        memref.store %26, %1[%14, %17] : memref<2000x2600xf64>
        memref.store %cst, %3[%14, %17] : memref<2000x2600xf64>
        %27 = addi %arg4, %c1_i32 : i32
        scf.yield %27 : i32
      }
      %16 = addi %arg3, %c1_i32 : i32
      scf.yield %16 : i32
    }
    %8 = index_cast %7#0 : i32 to index
    %9 = scf.for %arg2 = %8 to %c2600 step %c1 iter_args(%arg3 = %7#0) -> (i32) {
      %14 = index_cast %arg3 : i32 to index
      scf.for %arg4 = %c0 to %c2600 step %c1 {
        memref.store %cst, %2[%14, %arg4] : memref<2600x2600xf64>
      }
      %15 = addi %arg3, %c1_i32 : i32
      scf.yield %15 : i32
    }
    call @polybench_timer_start() : () -> ()
    %10 = memref.alloca() {scop.scratchpad} : memref<1xf64>
    %11 = memref.alloca() : memref<1xf64>
    affine.for %arg2 = 0 to 82 {
      affine.for %arg3 = #map0(%arg2) to 82 {
        affine.for %arg4 = #map1(%arg2) to min #map2(%arg2, %arg3) {
          affine.for %arg5 = max #map3(%arg3, %arg4) to min #map4(%arg3) {
            affine.store %cst, %2[%arg4, %arg5] : memref<2600x2600xf64>
          }
        }
      }
    } {scop.parallelizable}
    affine.for %arg2 = 0 to 2600 {
      affine.for %arg3 = 0 to 63 {
        affine.for %arg4 = #map1(%arg3) to min #map5(%arg3) {
          %17 = affine.load %1[%arg4, %arg2] : memref<2000x2600xf64>
          %18 = mulf %17, %17 {scop.splittable = 0 : index} : f64
          affine.store %18, %0[%arg4] : memref<2000xf64>
        }
      } {scop.parallelizable}
      affine.store %cst, %11[0] : memref<1xf64>
      affine.for %arg3 = 0 to 2000 {
        %17 = affine.load %11[0] : memref<1xf64>
        %18 = affine.load %0[%arg3] : memref<2000xf64>
        %19 = addf %17, %18 : f64
        affine.store %19, %11[0] : memref<1xf64>
      }
      %14 = affine.load %11[0] : memref<1xf64>
      %15 = math.sqrt %14 : f64
      affine.store %15, %10[0] : memref<1xf64>
      affine.for %arg3 = 0 to 63 {
        affine.for %arg4 = #map1(%arg3) to min #map5(%arg3) {
          %17 = affine.load %1[%arg4, %arg2] : memref<2000x2600xf64>
          %18 = affine.load %10[0] : memref<1xf64>
          %19 = divf %17, %18 : f64
          affine.store %19, %3[%arg4, %arg2] : memref<2000x2600xf64>
        }
      } {scop.parallelizable}
      affine.if #set(%arg2) {
        affine.for %arg3 = #map6(%arg2) to 82 {
          affine.for %arg4 = 0 to 63 {
            affine.for %arg5 = #map1(%arg4) to min #map5(%arg4) {
              affine.for %arg6 = max #map3(%arg3, %arg2) to min #map4(%arg3) {
                %17 = affine.load %2[%arg2, %arg6] : memref<2600x2600xf64>
                %18 = affine.load %3[%arg5, %arg2] : memref<2000x2600xf64>
                %19 = affine.load %1[%arg5, %arg6] : memref<2000x2600xf64>
                %20 = mulf %18, %19 {scop.splittable = 1 : index} : f64
                %21 = addf %17, %20 : f64
                affine.store %21, %2[%arg2, %arg6] : memref<2600x2600xf64>
              }
            }
          }
          affine.for %arg4 = 0 to 63 {
            affine.for %arg5 = #map1(%arg4) to min #map5(%arg4) {
              affine.for %arg6 = max #map3(%arg3, %arg2) to min #map4(%arg3) {
                %17 = affine.load %1[%arg5, %arg6] : memref<2000x2600xf64>
                %18 = affine.load %3[%arg5, %arg2] : memref<2000x2600xf64>
                %19 = affine.load %2[%arg2, %arg6] : memref<2600x2600xf64>
                %20 = mulf %18, %19 {scop.splittable = 2 : index} : f64
                %21 = subf %17, %20 : f64
                affine.store %21, %1[%arg5, %arg6] : memref<2000x2600xf64>
              }
            }
          }
        } {scop.parallelizable}
      }
      %16 = affine.load %10[0] : memref<1xf64>
      affine.store %16, %2[%arg2, %arg2] : memref<2600x2600xf64>
    }
    call @polybench_timer_stop() : () -> ()
    call @polybench_timer_print() : () -> ()
    %12 = cmpi sgt, %arg0, %c42_i32 : i32
    %13 = scf.if %12 -> (i1) {
      %14 = llvm.getelementptr %arg1[%c0_i64] : (!llvm.ptr<ptr<i8>>, i64) -> !llvm.ptr<ptr<i8>>
      %15 = llvm.load %14 : !llvm.ptr<ptr<i8>>
      %16 = llvm.mlir.addressof @str0 : !llvm.ptr<array<1 x i8>>
      %17 = llvm.getelementptr %16[%c0_i64, %c0_i64] : (!llvm.ptr<array<1 x i8>>, i64, i64) -> !llvm.ptr<i8>
      %18 = llvm.call @strcmp(%15, %17) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
      %19 = trunci %18 : i32 to i1
      %20 = xor %19, %true : i1
      scf.yield %20 : i1
    } else {
      scf.yield %false : i1
    }
    scf.if %13 {
      call @print_array(%c2000_i32, %c2600_i32, %4, %5, %6) : (i32, i32, memref<?x2600xf64>, memref<?x2600xf64>, memref<?x2600xf64>) -> ()
    }
    memref.dealloc %1 : memref<2000x2600xf64>
    memref.dealloc %2 : memref<2600x2600xf64>
    memref.dealloc %3 : memref<2000x2600xf64>
    return %c0_i32 : i32
  }
  func private @polybench_timer_start()
  func private @polybench_timer_stop()
  func private @polybench_timer_print()
  func private @S0(%arg0: memref<1xf64>) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f64
    affine.store %cst, %arg0[0] : memref<1xf64>
    return
  }
  func private @S1(%arg0: memref<?xf64>, %arg1: index, %arg2: memref<?x2600xf64>, %arg3: index) attributes {scop.stmt} {
    %0 = affine.load %arg2[symbol(%arg1), symbol(%arg3)] : memref<?x2600xf64>
    %1 = mulf %0, %0 {scop.splittable = 0 : index} : f64
    affine.store %1, %arg0[symbol(%arg1)] : memref<?xf64>
    return
  }
  func private @S2(%arg0: memref<1xf64>, %arg1: memref<?xf64>, %arg2: index) attributes {scop.stmt} {
    %0 = affine.load %arg0[0] : memref<1xf64>
    %1 = affine.load %arg1[symbol(%arg2)] : memref<?xf64>
    %2 = addf %0, %1 : f64
    affine.store %2, %arg0[0] : memref<1xf64>
    return
  }
  func private @S3(%arg0: memref<1xf64>, %arg1: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg1[0] : memref<1xf64>
    %1 = math.sqrt %0 : f64
    affine.store %1, %arg0[0] : memref<1xf64>
    return
  }
  func private @S4(%arg0: memref<?x2600xf64>, %arg1: index, %arg2: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg2[0] : memref<1xf64>
    affine.store %0, %arg0[symbol(%arg1), symbol(%arg1)] : memref<?x2600xf64>
    return
  }
  func private @S5(%arg0: memref<?x2600xf64>, %arg1: index, %arg2: index, %arg3: memref<1xf64>, %arg4: memref<?x2600xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg4[symbol(%arg1), symbol(%arg2)] : memref<?x2600xf64>
    %1 = affine.load %arg3[0] : memref<1xf64>
    %2 = divf %0, %1 : f64
    affine.store %2, %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x2600xf64>
    return
  }
  func private @S6(%arg0: memref<?x2600xf64>, %arg1: index, %arg2: index) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f64
    affine.store %cst, %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x2600xf64>
    return
  }
  func private @S7(%arg0: memref<?x2600xf64>, %arg1: index, %arg2: index, %arg3: memref<?x2600xf64>, %arg4: index, %arg5: memref<?x2600xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x2600xf64>
    %1 = affine.load %arg5[symbol(%arg4), symbol(%arg1)] : memref<?x2600xf64>
    %2 = affine.load %arg3[symbol(%arg4), symbol(%arg2)] : memref<?x2600xf64>
    %3 = mulf %1, %2 {scop.splittable = 1 : index} : f64
    %4 = addf %0, %3 : f64
    affine.store %4, %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x2600xf64>
    return
  }
  func private @S8(%arg0: memref<?x2600xf64>, %arg1: index, %arg2: index, %arg3: memref<?x2600xf64>, %arg4: index, %arg5: memref<?x2600xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x2600xf64>
    %1 = affine.load %arg5[symbol(%arg1), symbol(%arg4)] : memref<?x2600xf64>
    %2 = affine.load %arg3[symbol(%arg4), symbol(%arg2)] : memref<?x2600xf64>
    %3 = mulf %1, %2 {scop.splittable = 2 : index} : f64
    %4 = subf %0, %3 : f64
    affine.store %4, %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x2600xf64>
    return
  }
  func private @print_array(%arg0: i32, %arg1: i32, %arg2: memref<?x2600xf64>, %arg3: memref<?x2600xf64>, %arg4: memref<?x2600xf64>) {
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
    %9 = llvm.mlir.addressof @str3 : !llvm.ptr<array<2 x i8>>
    %10 = llvm.getelementptr %9[%c0_i64, %c0_i64] : (!llvm.ptr<array<2 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %11 = llvm.call @fprintf(%6, %8, %10) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
    %12 = scf.while (%arg5 = %c0_i32) : (i32) -> i32 {
      %40 = cmpi slt, %arg5, %arg1 : i32
      scf.condition(%40) %arg5 : i32
    } do {
    ^bb0(%arg5: i32):  // no predecessors
      %40 = index_cast %arg5 : i32 to index
      %41 = scf.while (%arg6 = %c0_i32) : (i32) -> i32 {
        %43 = cmpi slt, %arg6, %arg1 : i32
        scf.condition(%43) %arg6 : i32
      } do {
      ^bb0(%arg6: i32):  // no predecessors
        %43 = index_cast %arg6 : i32 to index
        %44 = muli %arg5, %arg1 : i32
        %45 = addi %44, %arg6 : i32
        %46 = remi_signed %45, %c20_i32 : i32
        %47 = cmpi eq, %46, %c0_i32 : i32
        scf.if %47 {
          %55 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
          %56 = llvm.load %55 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
          %57 = llvm.mlir.addressof @str4 : !llvm.ptr<array<2 x i8>>
          %58 = llvm.getelementptr %57[%c0_i64, %c0_i64] : (!llvm.ptr<array<2 x i8>>, i64, i64) -> !llvm.ptr<i8>
          %59 = llvm.call @fprintf(%56, %58) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> i32
        }
        %48 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
        %49 = llvm.load %48 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
        %50 = llvm.mlir.addressof @str5 : !llvm.ptr<array<8 x i8>>
        %51 = llvm.getelementptr %50[%c0_i64, %c0_i64] : (!llvm.ptr<array<8 x i8>>, i64, i64) -> !llvm.ptr<i8>
        %52 = memref.load %arg3[%40, %43] : memref<?x2600xf64>
        %53 = llvm.call @fprintf(%49, %51, %52) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, f64) -> i32
        %54 = addi %arg6, %c1_i32 : i32
        scf.yield %54 : i32
      }
      %42 = addi %arg5, %c1_i32 : i32
      scf.yield %42 : i32
    }
    %13 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %14 = llvm.load %13 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %15 = llvm.mlir.addressof @str6 : !llvm.ptr<array<17 x i8>>
    %16 = llvm.getelementptr %15[%c0_i64, %c0_i64] : (!llvm.ptr<array<17 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %17 = llvm.mlir.addressof @str3 : !llvm.ptr<array<2 x i8>>
    %18 = llvm.getelementptr %17[%c0_i64, %c0_i64] : (!llvm.ptr<array<2 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %19 = llvm.call @fprintf(%14, %16, %18) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
    %20 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %21 = llvm.load %20 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %22 = llvm.mlir.addressof @str2 : !llvm.ptr<array<15 x i8>>
    %23 = llvm.getelementptr %22[%c0_i64, %c0_i64] : (!llvm.ptr<array<15 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %24 = llvm.mlir.addressof @str7 : !llvm.ptr<array<2 x i8>>
    %25 = llvm.getelementptr %24[%c0_i64, %c0_i64] : (!llvm.ptr<array<2 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %26 = llvm.call @fprintf(%21, %23, %25) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
    %27 = scf.while (%arg5 = %c0_i32) : (i32) -> i32 {
      %40 = cmpi slt, %arg5, %arg0 : i32
      scf.condition(%40) %arg5 : i32
    } do {
    ^bb0(%arg5: i32):  // no predecessors
      %40 = index_cast %arg5 : i32 to index
      %41 = scf.while (%arg6 = %c0_i32) : (i32) -> i32 {
        %43 = cmpi slt, %arg6, %arg1 : i32
        scf.condition(%43) %arg6 : i32
      } do {
      ^bb0(%arg6: i32):  // no predecessors
        %43 = index_cast %arg6 : i32 to index
        %44 = muli %arg5, %arg1 : i32
        %45 = addi %44, %arg6 : i32
        %46 = remi_signed %45, %c20_i32 : i32
        %47 = cmpi eq, %46, %c0_i32 : i32
        scf.if %47 {
          %55 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
          %56 = llvm.load %55 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
          %57 = llvm.mlir.addressof @str4 : !llvm.ptr<array<2 x i8>>
          %58 = llvm.getelementptr %57[%c0_i64, %c0_i64] : (!llvm.ptr<array<2 x i8>>, i64, i64) -> !llvm.ptr<i8>
          %59 = llvm.call @fprintf(%56, %58) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> i32
        }
        %48 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
        %49 = llvm.load %48 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
        %50 = llvm.mlir.addressof @str5 : !llvm.ptr<array<8 x i8>>
        %51 = llvm.getelementptr %50[%c0_i64, %c0_i64] : (!llvm.ptr<array<8 x i8>>, i64, i64) -> !llvm.ptr<i8>
        %52 = memref.load %arg4[%40, %43] : memref<?x2600xf64>
        %53 = llvm.call @fprintf(%49, %51, %52) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, f64) -> i32
        %54 = addi %arg6, %c1_i32 : i32
        scf.yield %54 : i32
      }
      %42 = addi %arg5, %c1_i32 : i32
      scf.yield %42 : i32
    }
    %28 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %29 = llvm.load %28 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %30 = llvm.mlir.addressof @str6 : !llvm.ptr<array<17 x i8>>
    %31 = llvm.getelementptr %30[%c0_i64, %c0_i64] : (!llvm.ptr<array<17 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %32 = llvm.mlir.addressof @str7 : !llvm.ptr<array<2 x i8>>
    %33 = llvm.getelementptr %32[%c0_i64, %c0_i64] : (!llvm.ptr<array<2 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %34 = llvm.call @fprintf(%29, %31, %33) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
    %35 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %36 = llvm.load %35 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %37 = llvm.mlir.addressof @str8 : !llvm.ptr<array<23 x i8>>
    %38 = llvm.getelementptr %37[%c0_i64, %c0_i64] : (!llvm.ptr<array<23 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %39 = llvm.call @fprintf(%36, %38) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> i32
    return
  }
}

