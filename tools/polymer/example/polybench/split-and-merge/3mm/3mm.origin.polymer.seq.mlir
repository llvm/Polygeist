#map0 = affine_map<()[s0, s1] -> ((s0 + s1 - 2) floordiv 32 + 1)>
#map1 = affine_map<()[s0] -> ((s0 - 1) floordiv 32 + 1)>
#map2 = affine_map<(d0) -> (d0 * 32)>
#map3 = affine_map<(d0)[s0, s1] -> (s0, s1, d0 * 32 + 32)>
#map4 = affine_map<(d0)[s0] -> (s0, d0 * 32 + 32)>
#map5 = affine_map<(d0)[s0] -> (s0, d0 * 32)>
#set = affine_set<()[s0, s1, s2, s3, s4] : (s0 - 1 >= 0, s1 - 1 >= 0, s2 - 1 >= 0, s3 - 1 >= 0, s4 - 1 >= 0)>
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  llvm.mlir.global internal constant @str7("==END   DUMP_ARRAYS==\0A\00")
  llvm.mlir.global internal constant @str6("\0Aend   dump: %s\0A\00")
  llvm.mlir.global internal constant @str5("%0.2lf \00")
  llvm.mlir.global internal constant @str4("\0A\00")
  llvm.mlir.global internal constant @str3("G\00")
  llvm.mlir.global internal constant @str2("begin dump: %s\00")
  llvm.mlir.global internal constant @str1("==BEGIN DUMP_ARRAYS==\0A\00")
  llvm.mlir.global external @stderr() : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>
  llvm.func @fprintf(!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, ...) -> i32
  llvm.mlir.global internal constant @str0("\00")
  llvm.func @strcmp(!llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
  func @main(%arg0: i32, %arg1: !llvm.ptr<ptr<i8>>) -> i32 {
    %c1600_i32 = constant 1600 : i32
    %c1800_i32 = constant 1800 : i32
    %c2000_i32 = constant 2000 : i32
    %c2200_i32 = constant 2200 : i32
    %c2400_i32 = constant 2400 : i32
    %c42_i32 = constant 42 : i32
    %c0_i64 = constant 0 : i64
    %true = constant true
    %false = constant false
    %c0_i32 = constant 0 : i32
    %0 = memref.alloc() : memref<1600x1800xf64>
    %1 = memref.alloc() : memref<1600x2000xf64>
    %2 = memref.alloc() : memref<2000x1800xf64>
    %3 = memref.alloc() : memref<1800x2200xf64>
    %4 = memref.alloc() : memref<1800x2400xf64>
    %5 = memref.alloc() : memref<2400x2200xf64>
    %6 = memref.alloc() : memref<1600x2200xf64>
    %7 = memref.cast %1 : memref<1600x2000xf64> to memref<?x2000xf64>
    %8 = memref.cast %2 : memref<2000x1800xf64> to memref<?x1800xf64>
    %9 = memref.cast %4 : memref<1800x2400xf64> to memref<?x2400xf64>
    %10 = memref.cast %5 : memref<2400x2200xf64> to memref<?x2200xf64>
    call @init_array(%c1600_i32, %c1800_i32, %c2000_i32, %c2200_i32, %c2400_i32, %7, %8, %9, %10) : (i32, i32, i32, i32, i32, memref<?x2000xf64>, memref<?x1800xf64>, memref<?x2400xf64>, memref<?x2200xf64>) -> ()
    call @polybench_timer_start() : () -> ()
    %11 = memref.cast %0 : memref<1600x1800xf64> to memref<?x1800xf64>
    %12 = memref.cast %3 : memref<1800x2200xf64> to memref<?x2200xf64>
    %13 = memref.cast %6 : memref<1600x2200xf64> to memref<?x2200xf64>
    call @kernel_3mm_opt(%c1600_i32, %c1800_i32, %c2000_i32, %c2200_i32, %c2400_i32, %11, %7, %8, %12, %9, %10, %13) : (i32, i32, i32, i32, i32, memref<?x1800xf64>, memref<?x2000xf64>, memref<?x1800xf64>, memref<?x2200xf64>, memref<?x2400xf64>, memref<?x2200xf64>, memref<?x2200xf64>) -> ()
    call @polybench_timer_stop() : () -> ()
    call @polybench_timer_print() : () -> ()
    %14 = cmpi sgt, %arg0, %c42_i32 : i32
    %15 = scf.if %14 -> (i1) {
      %16 = llvm.getelementptr %arg1[%c0_i64] : (!llvm.ptr<ptr<i8>>, i64) -> !llvm.ptr<ptr<i8>>
      %17 = llvm.load %16 : !llvm.ptr<ptr<i8>>
      %18 = llvm.mlir.addressof @str0 : !llvm.ptr<array<1 x i8>>
      %19 = llvm.getelementptr %18[%c0_i64, %c0_i64] : (!llvm.ptr<array<1 x i8>>, i64, i64) -> !llvm.ptr<i8>
      %20 = llvm.call @strcmp(%17, %19) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
      %21 = trunci %20 : i32 to i1
      %22 = xor %21, %true : i1
      scf.yield %22 : i1
    } else {
      scf.yield %false : i1
    }
    scf.if %15 {
      call @print_array(%c1600_i32, %c2200_i32, %13) : (i32, i32, memref<?x2200xf64>) -> ()
    }
    memref.dealloc %0 : memref<1600x1800xf64>
    memref.dealloc %1 : memref<1600x2000xf64>
    memref.dealloc %2 : memref<2000x1800xf64>
    memref.dealloc %3 : memref<1800x2200xf64>
    memref.dealloc %4 : memref<1800x2400xf64>
    memref.dealloc %5 : memref<2400x2200xf64>
    memref.dealloc %6 : memref<1600x2200xf64>
    return %c0_i32 : i32
  }
  func private @init_array(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: memref<?x2000xf64>, %arg6: memref<?x1800xf64>, %arg7: memref<?x2400xf64>, %arg8: memref<?x2200xf64>) {
    %c3_i32 = constant 3 : i32
    %c2_i32 = constant 2 : i32
    %c5_i32 = constant 5 : i32
    %c1_i32 = constant 1 : i32
    %c0_i32 = constant 0 : i32
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %0:2 = scf.while (%arg9 = %c0_i32) : (i32) -> (i32, i32) {
      %4 = cmpi slt, %arg9, %arg0 : i32
      scf.condition(%4) %c0_i32, %arg9 : i32, i32
    } do {
    ^bb0(%arg9: i32, %arg10: i32):  // no predecessors
      %4 = index_cast %arg10 : i32 to index
      %5 = scf.while (%arg11 = %c0_i32) : (i32) -> i32 {
        %7 = cmpi slt, %arg11, %arg2 : i32
        scf.condition(%7) %arg11 : i32
      } do {
      ^bb0(%arg11: i32):  // no predecessors
        %7 = index_cast %arg11 : i32 to index
        %8 = muli %arg10, %arg11 : i32
        %9 = addi %8, %c1_i32 : i32
        %10 = remi_signed %9, %arg0 : i32
        %11 = sitofp %10 : i32 to f64
        %12 = muli %arg0, %c5_i32 : i32
        %13 = sitofp %12 : i32 to f64
        %14 = divf %11, %13 : f64
        memref.store %14, %arg5[%4, %7] : memref<?x2000xf64>
        %15 = addi %arg11, %c1_i32 : i32
        scf.yield %15 : i32
      }
      %6 = addi %arg10, %c1_i32 : i32
      scf.yield %6 : i32
    }
    %1:2 = scf.while (%arg9 = %0#0) : (i32) -> (i32, i32) {
      %4 = cmpi slt, %arg9, %arg2 : i32
      scf.condition(%4) %c0_i32, %arg9 : i32, i32
    } do {
    ^bb0(%arg9: i32, %arg10: i32):  // no predecessors
      %4 = index_cast %arg10 : i32 to index
      %5 = index_cast %arg1 : i32 to index
      %6 = scf.for %arg11 = %c0 to %5 step %c1 iter_args(%arg12 = %c0_i32) -> (i32) {
        %8 = index_cast %arg12 : i32 to index
        %9 = addi %arg12, %c1_i32 : i32
        %10 = muli %arg10, %9 : i32
        %11 = addi %10, %c2_i32 : i32
        %12 = remi_signed %11, %arg1 : i32
        %13 = sitofp %12 : i32 to f64
        %14 = muli %arg1, %c5_i32 : i32
        %15 = sitofp %14 : i32 to f64
        %16 = divf %13, %15 : f64
        memref.store %16, %arg6[%4, %8] : memref<?x1800xf64>
        scf.yield %9 : i32
      }
      %7 = addi %arg10, %c1_i32 : i32
      scf.yield %7 : i32
    }
    %2:2 = scf.while (%arg9 = %1#0) : (i32) -> (i32, i32) {
      %4 = cmpi slt, %arg9, %arg1 : i32
      scf.condition(%4) %c0_i32, %arg9 : i32, i32
    } do {
    ^bb0(%arg9: i32, %arg10: i32):  // no predecessors
      %4 = index_cast %arg10 : i32 to index
      %5 = scf.while (%arg11 = %c0_i32) : (i32) -> i32 {
        %7 = cmpi slt, %arg11, %arg4 : i32
        scf.condition(%7) %arg11 : i32
      } do {
      ^bb0(%arg11: i32):  // no predecessors
        %7 = index_cast %arg11 : i32 to index
        %8 = addi %arg11, %c3_i32 : i32
        %9 = muli %arg10, %8 : i32
        %10 = remi_signed %9, %arg3 : i32
        %11 = sitofp %10 : i32 to f64
        %12 = muli %arg3, %c5_i32 : i32
        %13 = sitofp %12 : i32 to f64
        %14 = divf %11, %13 : f64
        memref.store %14, %arg7[%4, %7] : memref<?x2400xf64>
        %15 = addi %arg11, %c1_i32 : i32
        scf.yield %15 : i32
      }
      %6 = addi %arg10, %c1_i32 : i32
      scf.yield %6 : i32
    }
    %3 = scf.while (%arg9 = %2#0) : (i32) -> i32 {
      %4 = cmpi slt, %arg9, %arg4 : i32
      scf.condition(%4) %arg9 : i32
    } do {
    ^bb0(%arg9: i32):  // no predecessors
      %4 = index_cast %arg9 : i32 to index
      %5 = scf.while (%arg10 = %c0_i32) : (i32) -> i32 {
        %7 = cmpi slt, %arg10, %arg3 : i32
        scf.condition(%7) %arg10 : i32
      } do {
      ^bb0(%arg10: i32):  // no predecessors
        %7 = index_cast %arg10 : i32 to index
        %8 = addi %arg10, %c2_i32 : i32
        %9 = muli %arg9, %8 : i32
        %10 = addi %9, %c2_i32 : i32
        %11 = remi_signed %10, %arg2 : i32
        %12 = sitofp %11 : i32 to f64
        %13 = muli %arg2, %c5_i32 : i32
        %14 = sitofp %13 : i32 to f64
        %15 = divf %12, %14 : f64
        memref.store %15, %arg8[%4, %7] : memref<?x2200xf64>
        %16 = addi %arg10, %c1_i32 : i32
        scf.yield %16 : i32
      }
      %6 = addi %arg9, %c1_i32 : i32
      scf.yield %6 : i32
    }
    return
  }
  func private @polybench_timer_start()
  func private @kernel_3mm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: memref<?x1800xf64>, %arg6: memref<?x2000xf64>, %arg7: memref<?x1800xf64>, %arg8: memref<?x2200xf64>, %arg9: memref<?x2400xf64>, %arg10: memref<?x2200xf64>, %arg11: memref<?x2200xf64>) {
    %0 = index_cast %arg1 : i32 to index
    %1 = index_cast %arg2 : i32 to index
    %2 = index_cast %arg3 : i32 to index
    %3 = index_cast %arg4 : i32 to index
    %4 = index_cast %arg0 : i32 to index
    affine.for %arg12 = 0 to %4 {
      affine.for %arg13 = 0 to %0 {
        call @S0(%arg5, %arg12, %arg13) : (memref<?x1800xf64>, index, index) -> ()
        affine.for %arg14 = 0 to %1 {
          call @S1(%arg5, %arg12, %arg13, %arg7, %arg14, %arg6) : (memref<?x1800xf64>, index, index, memref<?x1800xf64>, index, memref<?x2000xf64>) -> ()
        }
      }
    }
    affine.for %arg12 = 0 to %0 {
      affine.for %arg13 = 0 to %2 {
        call @S2(%arg8, %arg12, %arg13) : (memref<?x2200xf64>, index, index) -> ()
        affine.for %arg14 = 0 to %3 {
          call @S3(%arg8, %arg12, %arg13, %arg10, %arg14, %arg9) : (memref<?x2200xf64>, index, index, memref<?x2200xf64>, index, memref<?x2400xf64>) -> ()
        }
      }
    }
    affine.for %arg12 = 0 to %4 {
      affine.for %arg13 = 0 to %2 {
        call @S4(%arg11, %arg12, %arg13) : (memref<?x2200xf64>, index, index) -> ()
        affine.for %arg14 = 0 to %0 {
          call @S5(%arg11, %arg12, %arg13, %arg8, %arg14, %arg5) : (memref<?x2200xf64>, index, index, memref<?x2200xf64>, index, memref<?x1800xf64>) -> ()
        }
      }
    }
    return
  }
  func private @polybench_timer_stop()
  func private @polybench_timer_print()
  func private @S0(%arg0: memref<?x1800xf64>, %arg1: index, %arg2: index) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f64
    affine.store %cst, %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x1800xf64>
    return
  }
  func private @S1(%arg0: memref<?x1800xf64>, %arg1: index, %arg2: index, %arg3: memref<?x1800xf64>, %arg4: index, %arg5: memref<?x2000xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x1800xf64>
    %1 = affine.load %arg5[symbol(%arg1), symbol(%arg4)] : memref<?x2000xf64>
    %2 = affine.load %arg3[symbol(%arg4), symbol(%arg2)] : memref<?x1800xf64>
    %3 = mulf %1, %2 {scop.splittable = 0 : index} : f64
    %4 = addf %0, %3 : f64
    affine.store %4, %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x1800xf64>
    return
  }
  func private @S2(%arg0: memref<?x2200xf64>, %arg1: index, %arg2: index) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f64
    affine.store %cst, %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x2200xf64>
    return
  }
  func private @S3(%arg0: memref<?x2200xf64>, %arg1: index, %arg2: index, %arg3: memref<?x2200xf64>, %arg4: index, %arg5: memref<?x2400xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x2200xf64>
    %1 = affine.load %arg5[symbol(%arg1), symbol(%arg4)] : memref<?x2400xf64>
    %2 = affine.load %arg3[symbol(%arg4), symbol(%arg2)] : memref<?x2200xf64>
    %3 = mulf %1, %2 {scop.splittable = 1 : index} : f64
    %4 = addf %0, %3 : f64
    affine.store %4, %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x2200xf64>
    return
  }
  func private @S4(%arg0: memref<?x2200xf64>, %arg1: index, %arg2: index) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f64
    affine.store %cst, %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x2200xf64>
    return
  }
  func private @S5(%arg0: memref<?x2200xf64>, %arg1: index, %arg2: index, %arg3: memref<?x2200xf64>, %arg4: index, %arg5: memref<?x1800xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x2200xf64>
    %1 = affine.load %arg5[symbol(%arg1), symbol(%arg4)] : memref<?x1800xf64>
    %2 = affine.load %arg3[symbol(%arg4), symbol(%arg2)] : memref<?x2200xf64>
    %3 = mulf %1, %2 {scop.splittable = 2 : index} : f64
    %4 = addf %0, %3 : f64
    affine.store %4, %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x2200xf64>
    return
  }
  func private @kernel_3mm_opt(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: memref<?x1800xf64>, %arg6: memref<?x2000xf64>, %arg7: memref<?x1800xf64>, %arg8: memref<?x2200xf64>, %arg9: memref<?x2400xf64>, %arg10: memref<?x2200xf64>, %arg11: memref<?x2200xf64>) {
    %0 = index_cast %arg0 : i32 to index
    %1 = index_cast %arg1 : i32 to index
    %2 = index_cast %arg2 : i32 to index
    %3 = index_cast %arg3 : i32 to index
    %4 = index_cast %arg4 : i32 to index
    affine.if #set()[%0, %1, %2, %3, %4] {
      affine.for %arg12 = 0 to #map0()[%0, %1] {
        affine.for %arg13 = 0 to #map1()[%3] {
          affine.for %arg14 = #map2(%arg12) to min #map3(%arg12)[%0, %1] {
            affine.for %arg15 = #map2(%arg13) to min #map4(%arg13)[%3] {
              call @S2(%arg8, %arg14, %arg15) : (memref<?x2200xf64>, index, index) -> ()
              call @S4(%arg11, %arg14, %arg15) : (memref<?x2200xf64>, index, index) -> ()
            }
          }
          affine.for %arg14 = max #map5(%arg12)[%0] to min #map4(%arg12)[%1] {
            affine.for %arg15 = #map2(%arg13) to min #map4(%arg13)[%3] {
              call @S2(%arg8, %arg14, %arg15) : (memref<?x2200xf64>, index, index) -> ()
            }
          }
          affine.for %arg14 = max #map5(%arg12)[%1] to min #map4(%arg12)[%0] {
            affine.for %arg15 = #map2(%arg13) to min #map4(%arg13)[%3] {
              call @S4(%arg11, %arg14, %arg15) : (memref<?x2200xf64>, index, index) -> ()
            }
          }
        }
      }
      affine.for %arg12 = 0 to #map1()[%1] {
        affine.for %arg13 = 0 to #map1()[%3] {
          affine.for %arg14 = 0 to #map1()[%4] {
            affine.for %arg15 = #map2(%arg12) to min #map4(%arg12)[%1] {
              affine.for %arg16 = #map2(%arg14) to min #map4(%arg14)[%4] {
                affine.for %arg17 = #map2(%arg13) to min #map4(%arg13)[%3] {
                  call @S3(%arg8, %arg15, %arg17, %arg10, %arg16, %arg9) : (memref<?x2200xf64>, index, index, memref<?x2200xf64>, index, memref<?x2400xf64>) -> ()
                }
              }
            }
          }
        }
      }
      affine.for %arg12 = 0 to #map1()[%0] {
        affine.for %arg13 = 0 to #map1()[%1] {
          affine.for %arg14 = #map2(%arg12) to min #map4(%arg12)[%0] {
            affine.for %arg15 = #map2(%arg13) to min #map4(%arg13)[%1] {
              call @S0(%arg5, %arg14, %arg15) : (memref<?x1800xf64>, index, index) -> ()
            }
          }
        }
      }
      affine.for %arg12 = 0 to #map1()[%0] {
        affine.for %arg13 = 0 to #map1()[%1] {
          affine.for %arg14 = 0 to #map1()[%2] {
            affine.for %arg15 = #map2(%arg12) to min #map4(%arg12)[%0] {
              affine.for %arg16 = #map2(%arg14) to min #map4(%arg14)[%2] {
                affine.for %arg17 = #map2(%arg13) to min #map4(%arg13)[%1] {
                  call @S1(%arg5, %arg15, %arg17, %arg7, %arg16, %arg6) : (memref<?x1800xf64>, index, index, memref<?x1800xf64>, index, memref<?x2000xf64>) -> ()
                }
              }
            }
          }
          affine.for %arg14 = 0 to #map1()[%3] {
            affine.for %arg15 = #map2(%arg12) to min #map4(%arg12)[%0] {
              affine.for %arg16 = #map2(%arg13) to min #map4(%arg13)[%1] {
                affine.for %arg17 = #map2(%arg14) to min #map4(%arg14)[%3] {
                  call @S5(%arg11, %arg15, %arg17, %arg8, %arg16, %arg5) : (memref<?x2200xf64>, index, index, memref<?x2200xf64>, index, memref<?x1800xf64>) -> ()
                }
              }
            }
          }
        }
      }
    }
    return
  }
  func private @print_array(%arg0: i32, %arg1: i32, %arg2: memref<?x2200xf64>) {
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
    %12 = scf.while (%arg3 = %c0_i32) : (i32) -> i32 {
      %25 = cmpi slt, %arg3, %arg0 : i32
      scf.condition(%25) %arg3 : i32
    } do {
    ^bb0(%arg3: i32):  // no predecessors
      %25 = index_cast %arg3 : i32 to index
      %26 = scf.while (%arg4 = %c0_i32) : (i32) -> i32 {
        %28 = cmpi slt, %arg4, %arg1 : i32
        scf.condition(%28) %arg4 : i32
      } do {
      ^bb0(%arg4: i32):  // no predecessors
        %28 = index_cast %arg4 : i32 to index
        %29 = muli %arg3, %arg0 : i32
        %30 = addi %29, %arg4 : i32
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
        %37 = memref.load %arg2[%25, %28] : memref<?x2200xf64>
        %38 = llvm.call @fprintf(%34, %36, %37) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, f64) -> i32
        %39 = addi %arg4, %c1_i32 : i32
        scf.yield %39 : i32
      }
      %27 = addi %arg3, %c1_i32 : i32
      scf.yield %27 : i32
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
    %22 = llvm.mlir.addressof @str7 : !llvm.ptr<array<23 x i8>>
    %23 = llvm.getelementptr %22[%c0_i64, %c0_i64] : (!llvm.ptr<array<23 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %24 = llvm.call @fprintf(%21, %23) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> i32
    return
  }
}

