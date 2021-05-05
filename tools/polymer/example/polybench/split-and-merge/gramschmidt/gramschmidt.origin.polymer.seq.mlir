#map0 = affine_map<(d0) -> (d0 + 1)>
#map1 = affine_map<()[s0] -> ((s0 - 2) floordiv 32 + 1)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<()[s0] -> ((s0 - 1) floordiv 32 + 1)>
#map4 = affine_map<(d0) -> (d0 * 32)>
#map5 = affine_map<(d0, d1)[s0] -> (s0 - 1, d0 * 32 + 32, d1 * 32 + 31)>
#map6 = affine_map<(d0, d1) -> (d0 * 32, d1 + 1)>
#map7 = affine_map<(d0)[s0] -> (s0, d0 * 32 + 32)>
#map8 = affine_map<(d0) -> ((d0 - 30) ceildiv 32)>
#set0 = affine_set<()[s0, s1] : (s0 - 2 >= 0, s1 - 1 >= 0)>
#set1 = affine_set<(d0)[s0] : (-d0 + s0 - 2 >= 0)>
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
    %c0_i32 = constant 0 : i32
    %0 = memref.alloca() : memref<2000xf64>
    %1 = memref.cast %0 : memref<2000xf64> to memref<?xf64>
    %2 = memref.alloc() : memref<2000x2600xf64>
    %3 = memref.alloc() : memref<2600x2600xf64>
    %4 = memref.alloc() : memref<2000x2600xf64>
    %5 = memref.cast %2 : memref<2000x2600xf64> to memref<?x2600xf64>
    %6 = memref.cast %3 : memref<2600x2600xf64> to memref<?x2600xf64>
    %7 = memref.cast %4 : memref<2000x2600xf64> to memref<?x2600xf64>
    call @init_array(%c2000_i32, %c2600_i32, %5, %6, %7) : (i32, i32, memref<?x2600xf64>, memref<?x2600xf64>, memref<?x2600xf64>) -> ()
    call @polybench_timer_start() : () -> ()
    call @kernel_gramschmidt_opt(%c2000_i32, %c2600_i32, %5, %6, %7, %1) : (i32, i32, memref<?x2600xf64>, memref<?x2600xf64>, memref<?x2600xf64>, memref<?xf64>) -> ()
    call @polybench_timer_stop() : () -> ()
    call @polybench_timer_print() : () -> ()
    %8 = cmpi sgt, %arg0, %c42_i32 : i32
    %9 = scf.if %8 -> (i1) {
      %10 = llvm.getelementptr %arg1[%c0_i64] : (!llvm.ptr<ptr<i8>>, i64) -> !llvm.ptr<ptr<i8>>
      %11 = llvm.load %10 : !llvm.ptr<ptr<i8>>
      %12 = llvm.mlir.addressof @str0 : !llvm.ptr<array<1 x i8>>
      %13 = llvm.getelementptr %12[%c0_i64, %c0_i64] : (!llvm.ptr<array<1 x i8>>, i64, i64) -> !llvm.ptr<i8>
      %14 = llvm.call @strcmp(%11, %13) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
      %15 = trunci %14 : i32 to i1
      %16 = xor %15, %true : i1
      scf.yield %16 : i1
    } else {
      scf.yield %false : i1
    }
    scf.if %9 {
      call @print_array(%c2000_i32, %c2600_i32, %5, %6, %7) : (i32, i32, memref<?x2600xf64>, memref<?x2600xf64>, memref<?x2600xf64>) -> ()
    }
    memref.dealloc %2 : memref<2000x2600xf64>
    memref.dealloc %3 : memref<2600x2600xf64>
    memref.dealloc %4 : memref<2000x2600xf64>
    return %c0_i32 : i32
  }
  func private @init_array(%arg0: i32, %arg1: i32, %arg2: memref<?x2600xf64>, %arg3: memref<?x2600xf64>, %arg4: memref<?x2600xf64>) {
    %c100_i32 = constant 100 : i32
    %c10_i32 = constant 10 : i32
    %cst = constant 0.000000e+00 : f64
    %c1_i32 = constant 1 : i32
    %c0_i32 = constant 0 : i32
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %0:2 = scf.while (%arg5 = %c0_i32) : (i32) -> (i32, i32) {
      %4 = cmpi slt, %arg5, %arg0 : i32
      scf.condition(%4) %c0_i32, %arg5 : i32, i32
    } do {
    ^bb0(%arg5: i32, %arg6: i32):  // no predecessors
      %4 = index_cast %arg6 : i32 to index
      %5 = scf.while (%arg7 = %c0_i32) : (i32) -> i32 {
        %7 = cmpi slt, %arg7, %arg1 : i32
        scf.condition(%7) %arg7 : i32
      } do {
      ^bb0(%arg7: i32):  // no predecessors
        %7 = index_cast %arg7 : i32 to index
        %8 = muli %arg6, %arg7 : i32
        %9 = remi_signed %8, %arg0 : i32
        %10 = sitofp %9 : i32 to f64
        %11 = sitofp %arg0 : i32 to f64
        %12 = divf %10, %11 : f64
        %13 = sitofp %c100_i32 : i32 to f64
        %14 = mulf %12, %13 : f64
        %15 = sitofp %c10_i32 : i32 to f64
        %16 = addf %14, %15 : f64
        memref.store %16, %arg2[%4, %7] : memref<?x2600xf64>
        memref.store %cst, %arg4[%4, %7] : memref<?x2600xf64>
        %17 = addi %arg7, %c1_i32 : i32
        scf.yield %17 : i32
      }
      %6 = addi %arg6, %c1_i32 : i32
      scf.yield %6 : i32
    }
    %1 = index_cast %arg1 : i32 to index
    %2 = index_cast %0#0 : i32 to index
    %3 = scf.for %arg5 = %2 to %1 step %c1 iter_args(%arg6 = %0#0) -> (i32) {
      %4 = index_cast %arg6 : i32 to index
      scf.for %arg7 = %c0 to %1 step %c1 {
        memref.store %cst, %arg3[%4, %arg7] : memref<?x2600xf64>
      }
      %5 = addi %arg6, %c1_i32 : i32
      scf.yield %5 : i32
    }
    return
  }
  func private @polybench_timer_start()
  func private @kernel_gramschmidt(%arg0: i32, %arg1: i32, %arg2: memref<?x2600xf64>, %arg3: memref<?x2600xf64>, %arg4: memref<?x2600xf64>, %arg5: memref<?xf64>) {
    %0 = memref.alloca() {scop.scratchpad} : memref<1xf64>
    %1 = index_cast %arg0 : i32 to index
    %2 = memref.alloca() : memref<1xf64>
    %3 = index_cast %arg1 : i32 to index
    affine.for %arg6 = 0 to %3 {
      call @S0(%2) : (memref<1xf64>) -> ()
      affine.for %arg7 = 0 to %1 {
        call @S1(%arg5, %arg7, %arg2, %arg6) : (memref<?xf64>, index, memref<?x2600xf64>, index) -> ()
        call @S2(%2, %arg5, %arg7) : (memref<1xf64>, memref<?xf64>, index) -> ()
      }
      call @S3(%0, %2) : (memref<1xf64>, memref<1xf64>) -> ()
      call @S4(%arg3, %arg6, %0) : (memref<?x2600xf64>, index, memref<1xf64>) -> ()
      affine.for %arg7 = 0 to %1 {
        call @S5(%arg4, %arg7, %arg6, %0, %arg2) : (memref<?x2600xf64>, index, index, memref<1xf64>, memref<?x2600xf64>) -> ()
      }
      affine.for %arg7 = #map0(%arg6) to %3 {
        call @S6(%arg3, %arg6, %arg7) : (memref<?x2600xf64>, index, index) -> ()
        affine.for %arg8 = 0 to %1 {
          call @S7(%arg3, %arg6, %arg7, %arg2, %arg8, %arg4) : (memref<?x2600xf64>, index, index, memref<?x2600xf64>, index, memref<?x2600xf64>) -> ()
        }
        affine.for %arg8 = 0 to %1 {
          call @S8(%arg2, %arg8, %arg7, %arg3, %arg6, %arg4) : (memref<?x2600xf64>, index, index, memref<?x2600xf64>, index, memref<?x2600xf64>) -> ()
        }
      }
    }
    return
  }
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
  func private @kernel_gramschmidt_opt(%arg0: i32, %arg1: i32, %arg2: memref<?x2600xf64>, %arg3: memref<?x2600xf64>, %arg4: memref<?x2600xf64>, %arg5: memref<?xf64>) {
    %0 = memref.alloca() {scop.scratchpad} : memref<1xf64>
    %1 = memref.alloca() : memref<1xf64>
    %2 = index_cast %arg0 : i32 to index
    %3 = index_cast %arg1 : i32 to index
    affine.if #set0()[%3, %2] {
      affine.for %arg6 = 0 to #map1()[%3] {
        affine.for %arg7 = #map2(%arg6) to #map3()[%3] {
          affine.for %arg8 = #map4(%arg6) to min #map5(%arg6, %arg7)[%3] {
            affine.for %arg9 = max #map6(%arg7, %arg8) to min #map7(%arg7)[%3] {
              call @S6(%arg3, %arg8, %arg9) : (memref<?x2600xf64>, index, index) -> ()
            }
          }
        }
      }
      affine.for %arg6 = 0 to %3 {
        affine.for %arg7 = 0 to #map3()[%2] {
          affine.for %arg8 = #map4(%arg7) to min #map7(%arg7)[%2] {
            call @S1(%arg5, %arg8, %arg2, %arg6) : (memref<?xf64>, index, memref<?x2600xf64>, index) -> ()
          }
        }
        call @S0(%1) : (memref<1xf64>) -> ()
        affine.for %arg7 = 0 to %2 {
          call @S2(%1, %arg5, %arg7) : (memref<1xf64>, memref<?xf64>, index) -> ()
        }
        call @S3(%0, %1) : (memref<1xf64>, memref<1xf64>) -> ()
        affine.for %arg7 = 0 to #map3()[%2] {
          affine.for %arg8 = #map4(%arg7) to min #map7(%arg7)[%2] {
            call @S5(%arg4, %arg8, %arg6, %0, %arg2) : (memref<?x2600xf64>, index, index, memref<1xf64>, memref<?x2600xf64>) -> ()
          }
        }
        affine.if #set1(%arg6)[%3] {
          affine.for %arg7 = #map8(%arg6) to #map3()[%3] {
            affine.for %arg8 = 0 to #map3()[%2] {
              affine.for %arg9 = #map4(%arg8) to min #map7(%arg8)[%2] {
                affine.for %arg10 = max #map6(%arg7, %arg6) to min #map7(%arg7)[%3] {
                  call @S7(%arg3, %arg6, %arg10, %arg2, %arg9, %arg4) : (memref<?x2600xf64>, index, index, memref<?x2600xf64>, index, memref<?x2600xf64>) -> ()
                }
              }
            }
            affine.for %arg8 = 0 to #map3()[%2] {
              affine.for %arg9 = #map4(%arg8) to min #map7(%arg8)[%2] {
                affine.for %arg10 = max #map6(%arg7, %arg6) to min #map7(%arg7)[%3] {
                  call @S8(%arg2, %arg9, %arg10, %arg3, %arg6, %arg4) : (memref<?x2600xf64>, index, index, memref<?x2600xf64>, index, memref<?x2600xf64>) -> ()
                }
              }
            }
          }
        }
        call @S4(%arg3, %arg6, %0) : (memref<?x2600xf64>, index, memref<1xf64>) -> ()
      }
    }
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

