#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 * 32)>
#map2 = affine_map<(d0, d1) -> (2599, d0 * 32 + 32, d1 * 32 + 31)>
#map3 = affine_map<(d0, d1) -> (d0 * 32, d1 + 1)>
#map4 = affine_map<(d0) -> (2600, d0 * 32 + 32)>
#map5 = affine_map<(d0) -> (2599, d0 * 32 + 32)>
#map6 = affine_map<(d0) -> (3000, d0 * 32 + 32)>
#map7 = affine_map<(d0) -> (d0 + 1)>
#map8 = affine_map<()[s0] -> ((s0 - 2) floordiv 32 + 1)>
#map9 = affine_map<()[s0] -> ((s0 - 1) floordiv 32 + 1)>
#map10 = affine_map<(d0, d1)[s0] -> (s0 - 1, d0 * 32 + 32, d1 * 32 + 31)>
#map11 = affine_map<(d0)[s0] -> (s0, d0 * 32 + 32)>
#map12 = affine_map<(d0)[s0] -> (s0 - 1, d0 * 32 + 32)>
#set0 = affine_set<(d0) : (d0 - 81 >= 0)>
#set1 = affine_set<(d0)[s0] : (d0 - (s0 - 32) ceildiv 32 >= 0)>
#set2 = affine_set<()[s0] : (s0 - 1 >= 0)>
#set3 = affine_set<()[s0] : (s0 - 2 >= 0)>
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
    %0 = memref.alloca() : memref<3000xf64>
    %1 = memref.alloca() : memref<1xf64>
    %2 = memref.alloc() : memref<3000x2600xf64>
    %3 = memref.alloc() : memref<2600x2600xf64>
    %4 = memref.alloc() : memref<2600xf64>
    %5 = memref.alloc() : memref<2600xf64>
    %6 = sitofp %c3000_i32 : i32 to f64
    affine.store %6, %1[0] : memref<1xf64>
    %7 = scf.while (%arg2 = %c0_i32) : (i32) -> i32 {
      %12 = cmpi slt, %arg2, %c3000_i32 : i32
      scf.condition(%12) %arg2 : i32
    } do {
    ^bb0(%arg2: i32):  // no predecessors
      %12 = index_cast %arg2 : i32 to index
      %13 = scf.while (%arg3 = %c0_i32) : (i32) -> i32 {
        %15 = cmpi slt, %arg3, %c2600_i32 : i32
        scf.condition(%15) %arg3 : i32
      } do {
      ^bb0(%arg3: i32):  // no predecessors
        %15 = index_cast %arg3 : i32 to index
        %16 = muli %arg2, %arg3 : i32
        %17 = sitofp %16 : i32 to f64
        %18 = sitofp %c2600_i32 : i32 to f64
        %19 = divf %17, %18 : f64
        %20 = sitofp %arg2 : i32 to f64
        %21 = addf %19, %20 : f64
        memref.store %21, %2[%12, %15] : memref<3000x2600xf64>
        %22 = addi %arg3, %c1_i32 : i32
        scf.yield %22 : i32
      }
      %14 = addi %arg2, %c1_i32 : i32
      scf.yield %14 : i32
    }
    call @polybench_timer_start() : () -> ()
    %8 = affine.load %1[0] : memref<1xf64>
    %9 = memref.cast %3 : memref<2600x2600xf64> to memref<?x2600xf64>
    memref.store %cst_1, %3[%c2599, %c2599] : memref<2600x2600xf64>
    affine.for %arg2 = 0 to 82 {
      affine.for %arg3 = #map0(%arg2) to 82 {
        affine.for %arg4 = #map1(%arg2) to min #map2(%arg2, %arg3) {
          affine.for %arg5 = max #map3(%arg3, %arg4) to min #map4(%arg3) {
            affine.store %cst, %3[%arg4, %arg5] : memref<2600x2600xf64>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 82 {
      affine.for %arg3 = #map1(%arg2) to min #map5(%arg2) {
        affine.store %cst_1, %3[%arg3, %arg3] : memref<2600x2600xf64>
        affine.store %cst, %5[%arg3] : memref<2600xf64>
        affine.store %cst, %4[%arg3] : memref<2600xf64>
      }
      affine.if #set0(%arg2) {
        affine.store %cst, %5[2599] : memref<2600xf64>
        affine.store %cst, %4[2599] : memref<2600xf64>
      }
    }
    affine.for %arg2 = 0 to 82 {
      affine.for %arg3 = 0 to 94 {
        affine.for %arg4 = #map1(%arg3) to min #map6(%arg3) {
          affine.for %arg5 = #map1(%arg2) to min #map4(%arg2) {
            %12 = affine.load %4[%arg5] : memref<2600xf64>
            %13 = affine.load %2[%arg4, %arg5] : memref<3000x2600xf64>
            %14 = addf %12, %13 : f64
            affine.store %14, %4[%arg5] : memref<2600xf64>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 82 {
      affine.for %arg3 = #map1(%arg2) to min #map4(%arg2) {
        %12 = affine.load %4[%arg3] : memref<2600xf64>
        %13 = divf %12, %8 : f64
        affine.store %13, %4[%arg3] : memref<2600xf64>
      }
    }
    affine.for %arg2 = 0 to 82 {
      affine.for %arg3 = 0 to 94 {
        affine.for %arg4 = #map1(%arg3) to min #map6(%arg3) {
          affine.for %arg5 = #map1(%arg2) to min #map4(%arg2) {
            %12 = affine.load %5[%arg5] : memref<2600xf64>
            %13 = affine.load %2[%arg4, %arg5] : memref<3000x2600xf64>
            %14 = affine.load %4[%arg5] : memref<2600xf64>
            %15 = subf %13, %14 : f64
            %16 = mulf %15, %15 : f64
            %17 = addf %12, %16 : f64
            affine.store %17, %5[%arg5] : memref<2600xf64>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 82 {
      affine.for %arg3 = #map1(%arg2) to min #map4(%arg2) {
        %12 = affine.load %5[%arg3] : memref<2600xf64>
        %13 = divf %12, %8 : f64
        %14 = math.sqrt %13 : f64
        %15 = cmpf ule, %14, %cst_0 : f64
        %16 = select %15, %cst_1, %14 : f64
        affine.store %16, %5[%arg3] : memref<2600xf64>
      }
    }
    affine.for %arg2 = 0 to 94 {
      affine.for %arg3 = 0 to 82 {
        affine.for %arg4 = #map1(%arg2) to min #map6(%arg2) {
          affine.for %arg5 = #map1(%arg3) to min #map4(%arg3) {
            %12 = affine.load %2[%arg4, %arg5] : memref<3000x2600xf64>
            %13 = affine.load %4[%arg5] : memref<2600xf64>
            %14 = subf %12, %13 : f64
            %15 = math.sqrt %8 : f64
            %16 = affine.load %5[%arg5] : memref<2600xf64>
            %17 = mulf %15, %16 : f64
            %18 = divf %14, %17 : f64
            affine.store %18, %2[%arg4, %arg5] : memref<3000x2600xf64>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 94 {
      affine.for %arg3 = 0 to 82 {
        affine.for %arg4 = #map1(%arg2) to min #map6(%arg2) {
          affine.for %arg5 = #map1(%arg3) to min #map5(%arg3) {
            affine.for %arg6 = #map7(%arg5) to 2600 {
              %12 = affine.load %2[%arg4, %arg5] : memref<3000x2600xf64>
              %13 = affine.load %2[%arg4, %arg6] : memref<3000x2600xf64>
              %14 = mulf %12, %13 {scop.splittable = 0 : index} : f64
              affine.store %14, %0[%arg4] : memref<3000xf64>
              %15 = affine.load %3[%arg5, %arg6] : memref<2600x2600xf64>
              %16 = affine.load %0[%arg4] : memref<3000xf64>
              %17 = addf %15, %16 : f64
              affine.store %17, %3[%arg5, %arg6] : memref<2600x2600xf64>
            }
          }
        }
      }
    }
    affine.for %arg2 = 0 to 82 {
      affine.for %arg3 = #map0(%arg2) to 82 {
        affine.for %arg4 = #map1(%arg2) to min #map2(%arg2, %arg3) {
          affine.for %arg5 = max #map3(%arg3, %arg4) to min #map4(%arg3) {
            %12 = affine.load %3[%arg4, %arg5] : memref<2600x2600xf64>
            affine.store %12, %3[%arg5, %arg4] : memref<2600x2600xf64>
          }
        }
      }
    }
    call @polybench_timer_stop() : () -> ()
    call @polybench_timer_print() : () -> ()
    %10 = cmpi sgt, %arg0, %c42_i32 : i32
    %11 = scf.if %10 -> (i1) {
      %12 = llvm.getelementptr %arg1[%c0_i64] : (!llvm.ptr<ptr<i8>>, i64) -> !llvm.ptr<ptr<i8>>
      %13 = llvm.load %12 : !llvm.ptr<ptr<i8>>
      %14 = llvm.mlir.addressof @str0 : !llvm.ptr<array<1 x i8>>
      %15 = llvm.getelementptr %14[%c0_i64, %c0_i64] : (!llvm.ptr<array<1 x i8>>, i64, i64) -> !llvm.ptr<i8>
      %16 = llvm.call @strcmp(%13, %15) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
      %17 = trunci %16 : i32 to i1
      %18 = xor %17, %true : i1
      scf.yield %18 : i1
    } else {
      scf.yield %false : i1
    }
    scf.if %11 {
      call @print_array(%c2600_i32, %9) : (i32, memref<?x2600xf64>) -> ()
    }
    memref.dealloc %2 : memref<3000x2600xf64>
    memref.dealloc %3 : memref<2600x2600xf64>
    memref.dealloc %4 : memref<2600xf64>
    memref.dealloc %5 : memref<2600xf64>
    return %c0_i32 : i32
  }
  func private @polybench_timer_start()
  func private @polybench_timer_stop()
  func private @polybench_timer_print()
  func @kernel_correlation(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?x2600xf64>, %arg4: memref<?x2600xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: memref<?xf64>) {
    %c1 = constant 1 : index
    %cst = constant 0.000000e+00 : f64
    %cst_0 = constant 1.000000e-01 : f64
    %cst_1 = constant 1.000000e+00 : f64
    %0 = index_cast %arg0 : i32 to index
    %1 = index_cast %arg1 : i32 to index
    %2 = subi %0, %c1 : index
    memref.store %cst_1, %arg4[%2, %2] : memref<?x2600xf64>
    affine.for %arg8 = 0 to #map8()[%0] {
      affine.for %arg9 = #map0(%arg8) to #map9()[%0] {
        affine.for %arg10 = #map1(%arg8) to min #map10(%arg8, %arg9)[%0] {
          affine.for %arg11 = max #map3(%arg9, %arg10) to min #map11(%arg9)[%0] {
            affine.store %cst, %arg4[%arg10, %arg11] : memref<?x2600xf64>
          }
        }
      }
    }
    affine.for %arg8 = 0 to #map9()[%0] {
      affine.for %arg9 = #map1(%arg8) to min #map12(%arg8)[%0] {
        affine.store %cst_1, %arg4[%arg9, %arg9] : memref<?x2600xf64>
        affine.store %cst, %arg6[%arg9] : memref<?xf64>
        affine.store %cst, %arg5[%arg9] : memref<?xf64>
      }
      affine.if #set1(%arg8)[%0] {
        affine.store %cst, %arg6[symbol(%0) - 1] : memref<?xf64>
        affine.store %cst, %arg5[symbol(%0) - 1] : memref<?xf64>
      }
    }
    affine.if #set2()[%1] {
      affine.for %arg8 = 0 to #map9()[%0] {
        affine.for %arg9 = 0 to #map9()[%1] {
          affine.for %arg10 = #map1(%arg9) to min #map11(%arg9)[%1] {
            affine.for %arg11 = #map1(%arg8) to min #map11(%arg8)[%0] {
              %3 = affine.load %arg5[%arg11] : memref<?xf64>
              %4 = affine.load %arg3[%arg10, %arg11] : memref<?x2600xf64>
              %5 = addf %3, %4 : f64
              affine.store %5, %arg5[%arg11] : memref<?xf64>
            }
          }
        }
      }
    }
    affine.for %arg8 = 0 to #map9()[%0] {
      affine.for %arg9 = #map1(%arg8) to min #map11(%arg8)[%0] {
        %3 = affine.load %arg5[%arg9] : memref<?xf64>
        %4 = divf %3, %arg2 : f64
        affine.store %4, %arg5[%arg9] : memref<?xf64>
      }
    }
    affine.if #set2()[%1] {
      affine.for %arg8 = 0 to #map9()[%0] {
        affine.for %arg9 = 0 to #map9()[%1] {
          affine.for %arg10 = #map1(%arg9) to min #map11(%arg9)[%1] {
            affine.for %arg11 = #map1(%arg8) to min #map11(%arg8)[%0] {
              %3 = affine.load %arg6[%arg11] : memref<?xf64>
              %4 = affine.load %arg3[%arg10, %arg11] : memref<?x2600xf64>
              %5 = affine.load %arg5[%arg11] : memref<?xf64>
              %6 = subf %4, %5 : f64
              %7 = mulf %6, %6 : f64
              %8 = addf %3, %7 : f64
              affine.store %8, %arg6[%arg11] : memref<?xf64>
            }
          }
        }
      }
    }
    affine.for %arg8 = 0 to #map9()[%0] {
      affine.for %arg9 = #map1(%arg8) to min #map11(%arg8)[%0] {
        %3 = affine.load %arg6[%arg9] : memref<?xf64>
        %4 = divf %3, %arg2 : f64
        %5 = math.sqrt %4 : f64
        %6 = cmpf ule, %5, %cst_0 : f64
        %7 = select %6, %cst_1, %5 : f64
        affine.store %7, %arg6[%arg9] : memref<?xf64>
      }
    }
    affine.if #set2()[%0] {
      affine.for %arg8 = 0 to #map9()[%1] {
        affine.for %arg9 = 0 to #map9()[%0] {
          affine.for %arg10 = #map1(%arg8) to min #map11(%arg8)[%1] {
            affine.for %arg11 = #map1(%arg9) to min #map11(%arg9)[%0] {
              %3 = affine.load %arg3[%arg10, %arg11] : memref<?x2600xf64>
              %4 = affine.load %arg5[%arg11] : memref<?xf64>
              %5 = subf %3, %4 : f64
              %6 = math.sqrt %arg2 : f64
              %7 = affine.load %arg6[%arg11] : memref<?xf64>
              %8 = mulf %6, %7 : f64
              %9 = divf %5, %8 : f64
              affine.store %9, %arg3[%arg10, %arg11] : memref<?x2600xf64>
            }
          }
        }
      }
    }
    affine.if #set3()[%0] {
      affine.for %arg8 = 0 to #map9()[%1] {
        affine.for %arg9 = 0 to #map8()[%0] {
          affine.for %arg10 = #map1(%arg8) to min #map11(%arg8)[%1] {
            affine.for %arg11 = #map1(%arg9) to min #map12(%arg9)[%0] {
              affine.for %arg12 = #map7(%arg11) to %0 {
                %3 = affine.load %arg3[%arg10, %arg11] : memref<?x2600xf64>
                %4 = affine.load %arg3[%arg10, %arg12] : memref<?x2600xf64>
                %5 = mulf %3, %4 {scop.splittable = 0 : index} : f64
                affine.store %5, %arg7[%arg10] : memref<?xf64>
                %6 = affine.load %arg4[%arg11, %arg12] : memref<?x2600xf64>
                %7 = affine.load %arg7[%arg10] : memref<?xf64>
                %8 = addf %6, %7 : f64
                affine.store %8, %arg4[%arg11, %arg12] : memref<?x2600xf64>
              }
            }
          }
        }
      }
    }
    affine.for %arg8 = 0 to #map8()[%0] {
      affine.for %arg9 = #map0(%arg8) to #map9()[%0] {
        affine.for %arg10 = #map1(%arg8) to min #map10(%arg8, %arg9)[%0] {
          affine.for %arg11 = max #map3(%arg9, %arg10) to min #map11(%arg9)[%0] {
            %3 = affine.load %arg4[%arg10, %arg11] : memref<?x2600xf64>
            affine.store %3, %arg4[%arg11, %arg10] : memref<?x2600xf64>
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

