#map0 = affine_map<()[s0] -> (s0 - 1)>
#map1 = affine_map<(d0) -> (d0 + 1)>
#map2 = affine_map<()[s0] -> ((s0 - 2) floordiv 32 + 1)>
#map3 = affine_map<(d0) -> (d0)>
#map4 = affine_map<()[s0] -> ((s0 - 1) floordiv 32 + 1)>
#map5 = affine_map<(d0) -> (d0 * 32)>
#map6 = affine_map<(d0, d1)[s0] -> (s0 - 1, d0 * 32 + 32, d1 * 32 + 31)>
#map7 = affine_map<(d0, d1) -> (d0 * 32, d1 + 1)>
#map8 = affine_map<(d0)[s0] -> (s0, d0 * 32 + 32)>
#map9 = affine_map<(d0)[s0] -> (s0 - 1, d0 * 32 + 32)>
#set0 = affine_set<()[s0, s1] : (s0 - 2 >= 0, s1 - 1 >= 0)>
#set1 = affine_set<(d0)[s0] : (d0 - (s0 - 32) ceildiv 32 >= 0)>
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
    %4 = memref.alloc() : memref<2600xf64>
    %5 = memref.cast %0 : memref<1xf64> to memref<?xf64>
    %6 = memref.cast %1 : memref<3000x2600xf64> to memref<?x2600xf64>
    call @init_array(%c2600_i32, %c3000_i32, %5, %6) : (i32, i32, memref<?xf64>, memref<?x2600xf64>) -> ()
    call @polybench_timer_start() : () -> ()
    %7 = affine.load %0[0] : memref<1xf64>
    %8 = memref.cast %2 : memref<2600x2600xf64> to memref<?x2600xf64>
    %9 = memref.cast %3 : memref<2600xf64> to memref<?xf64>
    %10 = memref.cast %4 : memref<2600xf64> to memref<?xf64>
    call @kernel_correlation_opt(%c2600_i32, %c3000_i32, %7, %6, %8, %9, %10) : (i32, i32, f64, memref<?x2600xf64>, memref<?x2600xf64>, memref<?xf64>, memref<?xf64>) -> ()
    call @polybench_timer_stop() : () -> ()
    call @polybench_timer_print() : () -> ()
    %11 = cmpi sgt, %arg0, %c42_i32 : i32
    %12 = scf.if %11 -> (i1) {
      %13 = llvm.getelementptr %arg1[%c0_i64] : (!llvm.ptr<ptr<i8>>, i64) -> !llvm.ptr<ptr<i8>>
      %14 = llvm.load %13 : !llvm.ptr<ptr<i8>>
      %15 = llvm.mlir.addressof @str0 : !llvm.ptr<array<1 x i8>>
      %16 = llvm.getelementptr %15[%c0_i64, %c0_i64] : (!llvm.ptr<array<1 x i8>>, i64, i64) -> !llvm.ptr<i8>
      %17 = llvm.call @strcmp(%14, %16) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
      %18 = trunci %17 : i32 to i1
      %19 = xor %18, %true : i1
      scf.yield %19 : i1
    } else {
      scf.yield %false : i1
    }
    scf.if %12 {
      call @print_array(%c2600_i32, %8) : (i32, memref<?x2600xf64>) -> ()
    }
    memref.dealloc %1 : memref<3000x2600xf64>
    memref.dealloc %2 : memref<2600x2600xf64>
    memref.dealloc %3 : memref<2600xf64>
    memref.dealloc %4 : memref<2600xf64>
    return %c0_i32 : i32
  }
  func private @init_array(%arg0: i32, %arg1: i32, %arg2: memref<?xf64>, %arg3: memref<?x2600xf64>) {
    %c3000_i32 = constant 3000 : i32
    %c0_i32 = constant 0 : i32
    %c2600_i32 = constant 2600 : i32
    %c1_i32 = constant 1 : i32
    %0 = sitofp %c3000_i32 : i32 to f64
    affine.store %0, %arg2[0] : memref<?xf64>
    %1 = scf.while (%arg4 = %c0_i32) : (i32) -> i32 {
      %2 = cmpi slt, %arg4, %c3000_i32 : i32
      scf.condition(%2) %arg4 : i32
    } do {
    ^bb0(%arg4: i32):  // no predecessors
      %2 = index_cast %arg4 : i32 to index
      %3 = scf.while (%arg5 = %c0_i32) : (i32) -> i32 {
        %5 = cmpi slt, %arg5, %c2600_i32 : i32
        scf.condition(%5) %arg5 : i32
      } do {
      ^bb0(%arg5: i32):  // no predecessors
        %5 = index_cast %arg5 : i32 to index
        %6 = muli %arg4, %arg5 : i32
        %7 = sitofp %6 : i32 to f64
        %8 = sitofp %c2600_i32 : i32 to f64
        %9 = divf %7, %8 : f64
        %10 = sitofp %arg4 : i32 to f64
        %11 = addf %9, %10 : f64
        memref.store %11, %arg3[%2, %5] : memref<?x2600xf64>
        %12 = addi %arg5, %c1_i32 : i32
        scf.yield %12 : i32
      }
      %4 = addi %arg4, %c1_i32 : i32
      scf.yield %4 : i32
    }
    return
  }
  func private @polybench_timer_start()
  func private @kernel_correlation(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?x2600xf64>, %arg4: memref<?x2600xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>) {
    %0 = index_cast %arg1 : i32 to index
    %1 = index_cast %arg0 : i32 to index
    affine.for %arg7 = 0 to %1 {
      call @S0(%arg5, %arg7) : (memref<?xf64>, index) -> ()
      affine.for %arg8 = 0 to %0 {
        call @S1(%arg5, %arg7, %arg3, %arg8) : (memref<?xf64>, index, memref<?x2600xf64>, index) -> ()
      }
      call @S2(%arg5, %arg7, %arg2) : (memref<?xf64>, index, f64) -> ()
    }
    affine.for %arg7 = 0 to %1 {
      call @S3(%arg6, %arg7) : (memref<?xf64>, index) -> ()
      affine.for %arg8 = 0 to %0 {
        call @S4(%arg6, %arg7, %arg5, %arg3, %arg8) : (memref<?xf64>, index, memref<?xf64>, memref<?x2600xf64>, index) -> ()
      }
      call @S5(%arg6, %arg7, %arg2) : (memref<?xf64>, index, f64) -> ()
    }
    affine.for %arg7 = 0 to %0 {
      affine.for %arg8 = 0 to %1 {
        call @S6(%arg3, %arg7, %arg8, %arg6, %arg2, %arg5) : (memref<?x2600xf64>, index, index, memref<?xf64>, f64, memref<?xf64>) -> ()
      }
    }
    affine.for %arg7 = 0 to #map0()[%1] {
      call @S7(%arg4, %arg7) : (memref<?x2600xf64>, index) -> ()
      affine.for %arg8 = #map1(%arg7) to %1 {
        call @S8(%arg4, %arg7, %arg8) : (memref<?x2600xf64>, index, index) -> ()
        affine.for %arg9 = 0 to %0 {
          call @S9(%arg4, %arg7, %arg8, %arg3, %arg9) : (memref<?x2600xf64>, index, index, memref<?x2600xf64>, index) -> ()
        }
        call @S10(%arg4, %arg8, %arg7) : (memref<?x2600xf64>, index, index) -> ()
      }
    }
    call @S11(%arg4, %1) : (memref<?x2600xf64>, index) -> ()
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
  func private @S3(%arg0: memref<?xf64>, %arg1: index) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f64
    affine.store %cst, %arg0[symbol(%arg1)] : memref<?xf64>
    return
  }
  func private @S4(%arg0: memref<?xf64>, %arg1: index, %arg2: memref<?xf64>, %arg3: memref<?x2600xf64>, %arg4: index) attributes {scop.stmt} {
    %0 = affine.load %arg0[symbol(%arg1)] : memref<?xf64>
    %1 = affine.load %arg3[symbol(%arg4), symbol(%arg1)] : memref<?x2600xf64>
    %2 = affine.load %arg2[symbol(%arg1)] : memref<?xf64>
    %3 = subf %1, %2 {scop.splittable = 1 : index} : f64
    %4 = mulf %3, %3 {scop.splittable = 0 : index} : f64
    %5 = addf %0, %4 : f64
    affine.store %5, %arg0[symbol(%arg1)] : memref<?xf64>
    return
  }
  func private @S5(%arg0: memref<?xf64>, %arg1: index, %arg2: f64) attributes {scop.stmt} {
    %cst = constant 1.000000e-01 : f64
    %cst_0 = constant 1.000000e+00 : f64
    %0 = affine.load %arg0[symbol(%arg1)] : memref<?xf64>
    %1 = divf %0, %arg2 {scop.splittable = 4 : index} : f64
    %2 = math.sqrt %1 {scop.splittable = 3 : index} : f64
    %3 = cmpf ule, %2, %cst {scop.splittable = 2 : index} : f64
    %4 = select %3, %cst_0, %2 : f64
    affine.store %4, %arg0[symbol(%arg1)] : memref<?xf64>
    return
  }
  func private @S6(%arg0: memref<?x2600xf64>, %arg1: index, %arg2: index, %arg3: memref<?xf64>, %arg4: f64, %arg5: memref<?xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x2600xf64>
    %1 = affine.load %arg5[symbol(%arg2)] : memref<?xf64>
    %2 = subf %0, %1 {scop.splittable = 5 : index} : f64
    %3 = math.sqrt %arg4 {scop.splittable = 7 : index} : f64
    %4 = affine.load %arg3[symbol(%arg2)] : memref<?xf64>
    %5 = mulf %3, %4 {scop.splittable = 6 : index} : f64
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
    %3 = mulf %1, %2 {scop.splittable = 8 : index} : f64
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
  func private @kernel_correlation_opt(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?x2600xf64>, %arg4: memref<?x2600xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>) {
    %0 = index_cast %arg0 : i32 to index
    %1 = index_cast %arg1 : i32 to index
    affine.if #set0()[%0, %1] {
      call @S11(%arg4, %0) : (memref<?x2600xf64>, index) -> ()
      affine.for %arg7 = 0 to #map2()[%0] {
        affine.for %arg8 = #map3(%arg7) to #map4()[%0] {
          affine.for %arg9 = #map5(%arg7) to min #map6(%arg7, %arg8)[%0] {
            affine.for %arg10 = max #map7(%arg8, %arg9) to min #map8(%arg8)[%0] {
              call @S8(%arg4, %arg9, %arg10) : (memref<?x2600xf64>, index, index) -> ()
            }
          }
        }
      }
      affine.for %arg7 = 0 to #map4()[%0] {
        affine.for %arg8 = #map5(%arg7) to min #map9(%arg7)[%0] {
          call @S0(%arg5, %arg8) : (memref<?xf64>, index) -> ()
          call @S3(%arg6, %arg8) : (memref<?xf64>, index) -> ()
          call @S7(%arg4, %arg8) : (memref<?x2600xf64>, index) -> ()
        }
        affine.if #set1(%arg7)[%0] {
          %2 = affine.apply #map0()[%0]
          call @S0(%arg5, %2) : (memref<?xf64>, index) -> ()
          %3 = affine.apply #map0()[%0]
          call @S3(%arg6, %3) : (memref<?xf64>, index) -> ()
        }
      }
      affine.for %arg7 = 0 to #map4()[%0] {
        affine.for %arg8 = 0 to #map4()[%1] {
          affine.for %arg9 = #map5(%arg8) to min #map8(%arg8)[%1] {
            affine.for %arg10 = #map5(%arg7) to min #map8(%arg7)[%0] {
              call @S1(%arg5, %arg10, %arg3, %arg9) : (memref<?xf64>, index, memref<?x2600xf64>, index) -> ()
            }
          }
        }
      }
      affine.for %arg7 = 0 to #map4()[%0] {
        affine.for %arg8 = #map5(%arg7) to min #map8(%arg7)[%0] {
          call @S2(%arg5, %arg8, %arg2) : (memref<?xf64>, index, f64) -> ()
        }
      }
      affine.for %arg7 = 0 to #map4()[%0] {
        affine.for %arg8 = 0 to #map4()[%1] {
          affine.for %arg9 = #map5(%arg8) to min #map8(%arg8)[%1] {
            affine.for %arg10 = #map5(%arg7) to min #map8(%arg7)[%0] {
              call @S4(%arg6, %arg10, %arg5, %arg3, %arg9) : (memref<?xf64>, index, memref<?xf64>, memref<?x2600xf64>, index) -> ()
            }
          }
        }
      }
      affine.for %arg7 = 0 to #map4()[%0] {
        affine.for %arg8 = #map5(%arg7) to min #map8(%arg7)[%0] {
          call @S5(%arg6, %arg8, %arg2) : (memref<?xf64>, index, f64) -> ()
        }
      }
      affine.for %arg7 = 0 to #map4()[%1] {
        affine.for %arg8 = 0 to #map4()[%0] {
          affine.for %arg9 = #map5(%arg7) to min #map8(%arg7)[%1] {
            affine.for %arg10 = #map5(%arg8) to min #map8(%arg8)[%0] {
              call @S6(%arg3, %arg9, %arg10, %arg6, %arg2, %arg5) : (memref<?x2600xf64>, index, index, memref<?xf64>, f64, memref<?xf64>) -> ()
            }
          }
        }
      }
      affine.for %arg7 = 0 to #map2()[%0] {
        affine.for %arg8 = #map3(%arg7) to #map4()[%0] {
          affine.for %arg9 = 0 to #map4()[%1] {
            affine.for %arg10 = #map5(%arg7) to min #map6(%arg7, %arg8)[%0] {
              affine.for %arg11 = #map5(%arg9) to min #map8(%arg9)[%1] {
                affine.for %arg12 = max #map7(%arg8, %arg10) to min #map8(%arg8)[%0] {
                  call @S9(%arg4, %arg10, %arg12, %arg3, %arg11) : (memref<?x2600xf64>, index, index, memref<?x2600xf64>, index) -> ()
                }
              }
            }
          }
        }
      }
      affine.for %arg7 = 0 to #map2()[%0] {
        affine.for %arg8 = #map3(%arg7) to #map4()[%0] {
          affine.for %arg9 = #map5(%arg7) to min #map6(%arg7, %arg8)[%0] {
            affine.for %arg10 = max #map7(%arg8, %arg9) to min #map8(%arg8)[%0] {
              call @S10(%arg4, %arg10, %arg9) : (memref<?x2600xf64>, index, index) -> ()
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

