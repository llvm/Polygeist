#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<()[s0] -> ((s0 - 1) floordiv 32 + 1)>
#map2 = affine_map<(d0) -> (d0 * 32)>
#map3 = affine_map<(d0)[s0] -> (s0, d0 * 32 + 32)>
#map4 = affine_map<(d0, d1) -> (d0 * 32, d1)>
#set = affine_set<()[s0, s1] : (s0 - 1 >= 0, s1 - 1 >= 0)>
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  llvm.mlir.global internal constant @str7("==END   DUMP_ARRAYS==\0A\00")
  llvm.mlir.global internal constant @str6("\0Aend   dump: %s\0A\00")
  llvm.mlir.global internal constant @str5("%0.2lf \00")
  llvm.mlir.global internal constant @str4("\0A\00")
  llvm.mlir.global internal constant @str3("cov\00")
  llvm.mlir.global internal constant @str2("begin dump: %s\00")
  llvm.mlir.global internal constant @str1("==BEGIN DUMP_ARRAYS==\0A\00")
  llvm.mlir.global external @stderr() : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>
  llvm.func @fprintf(!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, ...) -> i32
  llvm.mlir.global internal constant @str0("\00")
  llvm.func @strcmp(!llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
  func @main(%arg0: i32, %arg1: !llvm.ptr<ptr<i8>>) -> i32 {
    %c3000_i32 = constant 3000 : i32
    %c2600_i32 = constant 2600 : i32
    %c42_i32 = constant 42 : i32
    %c0_i64 = constant 0 : i64
    %true = constant true
    %false = constant false
    %c0_i32 = constant 0 : i32
    %0 = memref.alloca() : memref<1xf64>
    %1 = memref.alloc() : memref<3000x2600xf64>
    %2 = memref.alloc() : memref<2600x2600xf64>
    %3 = memref.alloc() : memref<2600xf64>
    %4 = memref.cast %0 : memref<1xf64> to memref<?xf64>
    %5 = memref.cast %1 : memref<3000x2600xf64> to memref<?x2600xf64>
    call @init_array(%c2600_i32, %c3000_i32, %4, %5) : (i32, i32, memref<?xf64>, memref<?x2600xf64>) -> ()
    call @polybench_timer_start() : () -> ()
    %6 = affine.load %0[0] : memref<1xf64>
    %7 = memref.cast %2 : memref<2600x2600xf64> to memref<?x2600xf64>
    %8 = memref.cast %3 : memref<2600xf64> to memref<?xf64>
    call @kernel_covariance_opt(%c2600_i32, %c3000_i32, %6, %5, %7, %8) : (i32, i32, f64, memref<?x2600xf64>, memref<?x2600xf64>, memref<?xf64>) -> ()
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
      call @print_array(%c2600_i32, %7) : (i32, memref<?x2600xf64>) -> ()
    }
    memref.dealloc %1 : memref<3000x2600xf64>
    memref.dealloc %2 : memref<2600x2600xf64>
    memref.dealloc %3 : memref<2600xf64>
    return %c0_i32 : i32
  }
  func private @init_array(%arg0: i32, %arg1: i32, %arg2: memref<?xf64>, %arg3: memref<?x2600xf64>) {
    %c0_i32 = constant 0 : i32
    %c2600_i32 = constant 2600 : i32
    %c1_i32 = constant 1 : i32
    %c3000 = constant 3000 : index
    %c2600 = constant 2600 : index
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %0 = sitofp %arg1 : i32 to f64
    affine.store %0, %arg2[0] : memref<?xf64>
    %1 = scf.for %arg4 = %c0 to %c3000 step %c1 iter_args(%arg5 = %c0_i32) -> (i32) {
      %2 = index_cast %arg5 : i32 to index
      %3 = scf.for %arg6 = %c0 to %c2600 step %c1 iter_args(%arg7 = %c0_i32) -> (i32) {
        %5 = index_cast %arg7 : i32 to index
        %6 = sitofp %arg5 : i32 to f64
        %7 = sitofp %arg7 : i32 to f64
        %8 = mulf %6, %7 : f64
        %9 = sitofp %c2600_i32 : i32 to f64
        %10 = divf %8, %9 : f64
        memref.store %10, %arg3[%2, %5] : memref<?x2600xf64>
        %11 = addi %arg7, %c1_i32 : i32
        scf.yield %11 : i32
      }
      %4 = addi %arg5, %c1_i32 : i32
      scf.yield %4 : i32
    }
    return
  }
  func private @polybench_timer_start()
  func private @kernel_covariance(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?x2600xf64>, %arg4: memref<?x2600xf64>, %arg5: memref<?xf64>) {
    %0 = index_cast %arg1 : i32 to index
    %1 = index_cast %arg0 : i32 to index
    affine.for %arg6 = 0 to %1 {
      call @S0(%arg5, %arg6) : (memref<?xf64>, index) -> ()
      affine.for %arg7 = 0 to %0 {
        call @S1(%arg5, %arg6, %arg3, %arg7) : (memref<?xf64>, index, memref<?x2600xf64>, index) -> ()
      }
      call @S2(%arg5, %arg6, %arg2) : (memref<?xf64>, index, f64) -> ()
    }
    affine.for %arg6 = 0 to %0 {
      affine.for %arg7 = 0 to %1 {
        call @S3(%arg3, %arg6, %arg7, %arg5) : (memref<?x2600xf64>, index, index, memref<?xf64>) -> ()
      }
    }
    affine.for %arg6 = 0 to %1 {
      affine.for %arg7 = #map0(%arg6) to %1 {
        call @S4(%arg4, %arg6, %arg7) : (memref<?x2600xf64>, index, index) -> ()
        affine.for %arg8 = 0 to %0 {
          call @S5(%arg4, %arg6, %arg7, %arg3, %arg8) : (memref<?x2600xf64>, index, index, memref<?x2600xf64>, index) -> ()
        }
        call @S6(%arg4, %arg6, %arg7, %arg2) : (memref<?x2600xf64>, index, index, f64) -> ()
        call @S7(%arg4, %arg7, %arg6) : (memref<?x2600xf64>, index, index) -> ()
      }
    }
    return
  }
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
  func private @S3(%arg0: memref<?x2600xf64>, %arg1: index, %arg2: index, %arg3: memref<?xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x2600xf64>
    %1 = affine.load %arg3[symbol(%arg2)] : memref<?xf64>
    %2 = subf %0, %1 : f64
    affine.store %2, %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x2600xf64>
    return
  }
  func private @S4(%arg0: memref<?x2600xf64>, %arg1: index, %arg2: index) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f64
    affine.store %cst, %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x2600xf64>
    return
  }
  func private @S5(%arg0: memref<?x2600xf64>, %arg1: index, %arg2: index, %arg3: memref<?x2600xf64>, %arg4: index) attributes {scop.stmt} {
    %0 = affine.load %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x2600xf64>
    %1 = affine.load %arg3[symbol(%arg4), symbol(%arg1)] : memref<?x2600xf64>
    %2 = affine.load %arg3[symbol(%arg4), symbol(%arg2)] : memref<?x2600xf64>
    %3 = mulf %1, %2 {scop.splittable = 0 : index} : f64
    %4 = addf %0, %3 : f64
    affine.store %4, %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x2600xf64>
    return
  }
  func private @S6(%arg0: memref<?x2600xf64>, %arg1: index, %arg2: index, %arg3: f64) attributes {scop.stmt} {
    %cst = constant 1.000000e+00 : f64
    %0 = affine.load %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x2600xf64>
    %1 = subf %arg3, %cst : f64
    %2 = divf %0, %1 : f64
    affine.store %2, %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x2600xf64>
    return
  }
  func private @S7(%arg0: memref<?x2600xf64>, %arg1: index, %arg2: index) attributes {scop.stmt} {
    %0 = affine.load %arg0[symbol(%arg2), symbol(%arg1)] : memref<?x2600xf64>
    affine.store %0, %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x2600xf64>
    return
  }
  func private @kernel_covariance_opt(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?x2600xf64>, %arg4: memref<?x2600xf64>, %arg5: memref<?xf64>) {
    %0 = index_cast %arg0 : i32 to index
    %1 = index_cast %arg1 : i32 to index
    affine.if #set()[%0, %1] {
      affine.for %arg6 = 0 to #map1()[%0] {
        affine.for %arg7 = #map0(%arg6) to #map1()[%0] {
          affine.for %arg8 = #map2(%arg6) to min #map3(%arg6)[%0] {
            affine.for %arg9 = max #map4(%arg7, %arg8) to min #map3(%arg7)[%0] {
              call @S4(%arg4, %arg8, %arg9) : (memref<?x2600xf64>, index, index) -> ()
            }
          }
        }
      }
      affine.for %arg6 = 0 to #map1()[%0] {
        affine.for %arg7 = #map2(%arg6) to min #map3(%arg6)[%0] {
          call @S0(%arg5, %arg7) : (memref<?xf64>, index) -> ()
        }
      }
      affine.for %arg6 = 0 to #map1()[%0] {
        affine.for %arg7 = 0 to #map1()[%1] {
          affine.for %arg8 = #map2(%arg7) to min #map3(%arg7)[%1] {
            affine.for %arg9 = #map2(%arg6) to min #map3(%arg6)[%0] {
              call @S1(%arg5, %arg9, %arg3, %arg8) : (memref<?xf64>, index, memref<?x2600xf64>, index) -> ()
            }
          }
        }
      }
      affine.for %arg6 = 0 to #map1()[%0] {
        affine.for %arg7 = #map2(%arg6) to min #map3(%arg6)[%0] {
          call @S2(%arg5, %arg7, %arg2) : (memref<?xf64>, index, f64) -> ()
        }
      }
      affine.for %arg6 = 0 to #map1()[%1] {
        affine.for %arg7 = 0 to #map1()[%0] {
          affine.for %arg8 = #map2(%arg6) to min #map3(%arg6)[%1] {
            affine.for %arg9 = #map2(%arg7) to min #map3(%arg7)[%0] {
              call @S3(%arg3, %arg8, %arg9, %arg5) : (memref<?x2600xf64>, index, index, memref<?xf64>) -> ()
            }
          }
        }
      }
      affine.for %arg6 = 0 to #map1()[%0] {
        affine.for %arg7 = #map0(%arg6) to #map1()[%0] {
          affine.for %arg8 = 0 to #map1()[%1] {
            affine.for %arg9 = #map2(%arg6) to min #map3(%arg6)[%0] {
              affine.for %arg10 = #map2(%arg8) to min #map3(%arg8)[%1] {
                affine.for %arg11 = max #map4(%arg7, %arg9) to min #map3(%arg7)[%0] {
                  call @S5(%arg4, %arg9, %arg11, %arg3, %arg10) : (memref<?x2600xf64>, index, index, memref<?x2600xf64>, index) -> ()
                }
              }
            }
          }
        }
      }
      affine.for %arg6 = 0 to #map1()[%0] {
        affine.for %arg7 = #map0(%arg6) to #map1()[%0] {
          affine.for %arg8 = #map2(%arg6) to min #map3(%arg6)[%0] {
            affine.for %arg9 = max #map4(%arg7, %arg8) to min #map3(%arg7)[%0] {
              call @S6(%arg4, %arg8, %arg9, %arg2) : (memref<?x2600xf64>, index, index, f64) -> ()
              call @S7(%arg4, %arg9, %arg8) : (memref<?x2600xf64>, index, index) -> ()
            }
          }
        }
      }
    }
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
    %9 = llvm.mlir.addressof @str3 : !llvm.ptr<array<4 x i8>>
    %10 = llvm.getelementptr %9[%c0_i64, %c0_i64] : (!llvm.ptr<array<4 x i8>>, i64, i64) -> !llvm.ptr<i8>
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
    %17 = llvm.mlir.addressof @str3 : !llvm.ptr<array<4 x i8>>
    %18 = llvm.getelementptr %17[%c0_i64, %c0_i64] : (!llvm.ptr<array<4 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %19 = llvm.call @fprintf(%14, %16, %18) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
    %20 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %21 = llvm.load %20 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %22 = llvm.mlir.addressof @str7 : !llvm.ptr<array<23 x i8>>
    %23 = llvm.getelementptr %22[%c0_i64, %c0_i64] : (!llvm.ptr<array<23 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %24 = llvm.call @fprintf(%21, %23) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> i32
    return
  }
}

