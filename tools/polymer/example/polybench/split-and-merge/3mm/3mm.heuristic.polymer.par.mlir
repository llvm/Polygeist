#map0 = affine_map<(d0) -> (d0 * 32)>
#map1 = affine_map<(d0) -> (1600, d0 * 32 + 32)>
#map2 = affine_map<(d0) -> (2200, d0 * 32 + 32)>
#map3 = affine_map<(d0) -> (1800, d0 * 32 + 32)>
#map4 = affine_map<(d0) -> (0, (d0 * 32 - 1999) ceildiv 32)>
#map5 = affine_map<(d0) -> (50, d0 + 1)>
#map6 = affine_map<(d0, d1) -> (d0 * 32 - d1 * 32)>
#map7 = affine_map<(d0, d1) -> (2000, d0 * 32 - d1 * 32 + 32)>
#map8 = affine_map<(d0) -> (0, (d0 * 32 - 1799) ceildiv 32)>
#map9 = affine_map<(d0) -> (75, d0 + 1)>
#map10 = affine_map<(d0, d1) -> (1800, d0 * 32 - d1 * 32 + 32)>
#map11 = affine_map<(d0) -> (2400, d0 * 32 + 32)>
#map12 = affine_map<()[s0] -> ((s0 - 1) floordiv 32 + 1)>
#map13 = affine_map<(d0)[s0] -> (s0, d0 * 32 + 32)>
#map14 = affine_map<()[s0, s1] -> ((s0 + s1 - 2) floordiv 32 + 1)>
#map15 = affine_map<(d0)[s0] -> (0, (d0 * 32 - s0 + 1) ceildiv 32)>
#map16 = affine_map<(d0)[s0] -> ((s0 - 1) floordiv 32 + 1, d0 + 1)>
#map17 = affine_map<(d0, d1)[s0] -> (s0, d0 * 32 - d1 * 32 + 32)>
#set0 = affine_set<()[s0] : (s0 - 1 >= 0)>
#set1 = affine_set<()[s0, s1, s2] : (s0 - 1 >= 0, s1 - 1 >= 0, s2 - 1 >= 0)>
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
    %c3_i32 = constant 3 : i32
    %c2_i32 = constant 2 : i32
    %c1_i32 = constant 1 : i32
    %c0_i32 = constant 0 : i32
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c8000_i32 = constant 8000 : i32
    %c1800 = constant 1800 : index
    %c9000_i32 = constant 9000 : i32
    %c11000_i32 = constant 11000 : i32
    %c10000_i32 = constant 10000 : i32
    %cst = constant 0.000000e+00 : f64
    %0 = memref.alloca() : memref<2400xf64>
    %1 = memref.alloc() : memref<1600x1800xf64>
    %2 = memref.alloc() : memref<1600x2000xf64>
    %3 = memref.alloc() : memref<2000x1800xf64>
    %4 = memref.alloc() : memref<1800x2200xf64>
    %5 = memref.alloc() : memref<1800x2400xf64>
    %6 = memref.alloc() : memref<2400x2200xf64>
    %7 = memref.alloc() : memref<1600x2200xf64>
    %8:2 = scf.while (%arg2 = %c0_i32) : (i32) -> (i32, i32) {
      %15 = cmpi slt, %arg2, %c1600_i32 : i32
      scf.condition(%15) %c0_i32, %arg2 : i32, i32
    } do {
    ^bb0(%arg2: i32, %arg3: i32):  // no predecessors
      %15 = index_cast %arg3 : i32 to index
      %16 = scf.while (%arg4 = %c0_i32) : (i32) -> i32 {
        %18 = cmpi slt, %arg4, %c2000_i32 : i32
        scf.condition(%18) %arg4 : i32
      } do {
      ^bb0(%arg4: i32):  // no predecessors
        %18 = index_cast %arg4 : i32 to index
        %19 = muli %arg3, %arg4 : i32
        %20 = addi %19, %c1_i32 : i32
        %21 = remi_signed %20, %c1600_i32 : i32
        %22 = sitofp %21 : i32 to f64
        %23 = sitofp %c8000_i32 : i32 to f64
        %24 = divf %22, %23 : f64
        memref.store %24, %2[%15, %18] : memref<1600x2000xf64>
        %25 = addi %arg4, %c1_i32 : i32
        scf.yield %25 : i32
      }
      %17 = addi %arg3, %c1_i32 : i32
      scf.yield %17 : i32
    }
    %9:2 = scf.while (%arg2 = %8#0) : (i32) -> (i32, i32) {
      %15 = cmpi slt, %arg2, %c2000_i32 : i32
      scf.condition(%15) %c0_i32, %arg2 : i32, i32
    } do {
    ^bb0(%arg2: i32, %arg3: i32):  // no predecessors
      %15 = index_cast %arg3 : i32 to index
      %16 = scf.for %arg4 = %c0 to %c1800 step %c1 iter_args(%arg5 = %c0_i32) -> (i32) {
        %18 = index_cast %arg5 : i32 to index
        %19 = addi %arg5, %c1_i32 : i32
        %20 = muli %arg3, %19 : i32
        %21 = addi %20, %c2_i32 : i32
        %22 = remi_signed %21, %c1800_i32 : i32
        %23 = sitofp %22 : i32 to f64
        %24 = sitofp %c9000_i32 : i32 to f64
        %25 = divf %23, %24 : f64
        memref.store %25, %3[%15, %18] : memref<2000x1800xf64>
        scf.yield %19 : i32
      }
      %17 = addi %arg3, %c1_i32 : i32
      scf.yield %17 : i32
    }
    %10:2 = scf.while (%arg2 = %9#0) : (i32) -> (i32, i32) {
      %15 = cmpi slt, %arg2, %c1800_i32 : i32
      scf.condition(%15) %c0_i32, %arg2 : i32, i32
    } do {
    ^bb0(%arg2: i32, %arg3: i32):  // no predecessors
      %15 = index_cast %arg3 : i32 to index
      %16 = scf.while (%arg4 = %c0_i32) : (i32) -> i32 {
        %18 = cmpi slt, %arg4, %c2400_i32 : i32
        scf.condition(%18) %arg4 : i32
      } do {
      ^bb0(%arg4: i32):  // no predecessors
        %18 = index_cast %arg4 : i32 to index
        %19 = addi %arg4, %c3_i32 : i32
        %20 = muli %arg3, %19 : i32
        %21 = remi_signed %20, %c2200_i32 : i32
        %22 = sitofp %21 : i32 to f64
        %23 = sitofp %c11000_i32 : i32 to f64
        %24 = divf %22, %23 : f64
        memref.store %24, %5[%15, %18] : memref<1800x2400xf64>
        %25 = addi %arg4, %c1_i32 : i32
        scf.yield %25 : i32
      }
      %17 = addi %arg3, %c1_i32 : i32
      scf.yield %17 : i32
    }
    %11 = scf.while (%arg2 = %10#0) : (i32) -> i32 {
      %15 = cmpi slt, %arg2, %c2400_i32 : i32
      scf.condition(%15) %arg2 : i32
    } do {
    ^bb0(%arg2: i32):  // no predecessors
      %15 = index_cast %arg2 : i32 to index
      %16 = scf.while (%arg3 = %c0_i32) : (i32) -> i32 {
        %18 = cmpi slt, %arg3, %c2200_i32 : i32
        scf.condition(%18) %arg3 : i32
      } do {
      ^bb0(%arg3: i32):  // no predecessors
        %18 = index_cast %arg3 : i32 to index
        %19 = addi %arg3, %c2_i32 : i32
        %20 = muli %arg2, %19 : i32
        %21 = addi %20, %c2_i32 : i32
        %22 = remi_signed %21, %c2000_i32 : i32
        %23 = sitofp %22 : i32 to f64
        %24 = sitofp %c10000_i32 : i32 to f64
        %25 = divf %23, %24 : f64
        memref.store %25, %6[%15, %18] : memref<2400x2200xf64>
        %26 = addi %arg3, %c1_i32 : i32
        scf.yield %26 : i32
      }
      %17 = addi %arg2, %c1_i32 : i32
      scf.yield %17 : i32
    }
    call @polybench_timer_start() : () -> ()
    %12 = memref.cast %7 : memref<1600x2200xf64> to memref<?x2200xf64>
    affine.for %arg2 = 0 to 50 {
      affine.for %arg3 = 0 to 69 {
        affine.for %arg4 = #map0(%arg2) to min #map1(%arg2) {
          affine.for %arg5 = #map0(%arg3) to min #map2(%arg3) {
            affine.store %cst, %7[%arg4, %arg5] : memref<1600x2200xf64>
          }
        }
      }
    } {scop.parallelizable}
    affine.for %arg2 = 0 to 57 {
      affine.for %arg3 = 0 to 69 {
        affine.for %arg4 = #map0(%arg2) to min #map3(%arg2) {
          affine.for %arg5 = #map0(%arg3) to min #map2(%arg3) {
            affine.store %cst, %4[%arg4, %arg5] : memref<1800x2200xf64>
          }
        }
      }
    } {scop.parallelizable}
    affine.for %arg2 = 0 to 50 {
      affine.for %arg3 = 0 to 57 {
        affine.for %arg4 = #map0(%arg2) to min #map1(%arg2) {
          affine.for %arg5 = #map0(%arg3) to min #map3(%arg3) {
            affine.store %cst, %1[%arg4, %arg5] : memref<1600x1800xf64>
          }
        }
      }
    } {scop.parallelizable}
    affine.for %arg2 = 0 to 113 {
      affine.for %arg3 = max #map4(%arg2) to min #map5(%arg2) {
        affine.for %arg4 = #map6(%arg2, %arg3) to min #map7(%arg2, %arg3) {
          affine.for %arg5 = #map0(%arg3) to min #map1(%arg3) {
            affine.for %arg6 = 0 to 1800 {
              %15 = affine.load %2[%arg5, %arg4] : memref<1600x2000xf64>
              %16 = affine.load %3[%arg4, %arg6] : memref<2000x1800xf64>
              %17 = mulf %15, %16 {scop.splittable = 0 : index} : f64
              affine.store %17, %0[%arg4] : memref<2400xf64>
              %18 = affine.load %1[%arg5, %arg6] : memref<1600x1800xf64>
              %19 = affine.load %0[%arg4] : memref<2400xf64>
              %20 = addf %18, %19 : f64
              affine.store %20, %1[%arg5, %arg6] : memref<1600x1800xf64>
            }
          }
        }
      } {scop.parallelizable}
    }
    affine.for %arg2 = 0 to 132 {
      affine.for %arg3 = max #map8(%arg2) to min #map9(%arg2) {
        affine.for %arg4 = #map6(%arg2, %arg3) to min #map10(%arg2, %arg3) {
          affine.for %arg5 = #map0(%arg3) to min #map11(%arg3) {
            affine.for %arg6 = 0 to 2200 {
              %15 = affine.load %5[%arg4, %arg5] : memref<1800x2400xf64>
              %16 = affine.load %6[%arg5, %arg6] : memref<2400x2200xf64>
              %17 = mulf %15, %16 {scop.splittable = 1 : index} : f64
              affine.store %17, %0[%arg5] : memref<2400xf64>
              %18 = affine.load %4[%arg4, %arg6] : memref<1800x2200xf64>
              %19 = affine.load %0[%arg5] : memref<2400xf64>
              %20 = addf %18, %19 : f64
              affine.store %20, %4[%arg4, %arg6] : memref<1800x2200xf64>
            }
          }
        }
      } {scop.parallelizable}
    }
    affine.for %arg2 = 0 to 107 {
      affine.for %arg3 = max #map8(%arg2) to min #map5(%arg2) {
        affine.for %arg4 = #map6(%arg2, %arg3) to min #map10(%arg2, %arg3) {
          affine.for %arg5 = #map0(%arg3) to min #map1(%arg3) {
            affine.for %arg6 = 0 to 2200 {
              %15 = affine.load %1[%arg5, %arg4] : memref<1600x1800xf64>
              %16 = affine.load %4[%arg4, %arg6] : memref<1800x2200xf64>
              %17 = mulf %15, %16 {scop.splittable = 2 : index} : f64
              affine.store %17, %0[%arg4] : memref<2400xf64>
              %18 = affine.load %7[%arg5, %arg6] : memref<1600x2200xf64>
              %19 = affine.load %0[%arg4] : memref<2400xf64>
              %20 = addf %18, %19 : f64
              affine.store %20, %7[%arg5, %arg6] : memref<1600x2200xf64>
            }
          }
        }
      } {scop.parallelizable}
    }
    call @polybench_timer_stop() : () -> ()
    call @polybench_timer_print() : () -> ()
    %13 = cmpi sgt, %arg0, %c42_i32 : i32
    %14 = scf.if %13 -> (i1) {
      %15 = llvm.getelementptr %arg1[%c0_i64] : (!llvm.ptr<ptr<i8>>, i64) -> !llvm.ptr<ptr<i8>>
      %16 = llvm.load %15 : !llvm.ptr<ptr<i8>>
      %17 = llvm.mlir.addressof @str0 : !llvm.ptr<array<1 x i8>>
      %18 = llvm.getelementptr %17[%c0_i64, %c0_i64] : (!llvm.ptr<array<1 x i8>>, i64, i64) -> !llvm.ptr<i8>
      %19 = llvm.call @strcmp(%16, %18) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
      %20 = trunci %19 : i32 to i1
      %21 = xor %20, %true : i1
      scf.yield %21 : i1
    } else {
      scf.yield %false : i1
    }
    scf.if %14 {
      call @print_array(%c1600_i32, %c2200_i32, %12) : (i32, i32, memref<?x2200xf64>) -> ()
    }
    memref.dealloc %1 : memref<1600x1800xf64>
    memref.dealloc %2 : memref<1600x2000xf64>
    memref.dealloc %3 : memref<2000x1800xf64>
    memref.dealloc %4 : memref<1800x2200xf64>
    memref.dealloc %5 : memref<1800x2400xf64>
    memref.dealloc %6 : memref<2400x2200xf64>
    memref.dealloc %7 : memref<1600x2200xf64>
    return %c0_i32 : i32
  }
  func private @polybench_timer_start()
  func private @polybench_timer_stop()
  func private @polybench_timer_print()
  func @kernel_3mm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: memref<?x1800xf64>, %arg6: memref<?x2000xf64>, %arg7: memref<?x1800xf64>, %arg8: memref<?x2200xf64>, %arg9: memref<?x2400xf64>, %arg10: memref<?x2200xf64>, %arg11: memref<?x2200xf64>, %arg12: memref<?xf64>, %arg13: memref<?xf64>, %arg14: memref<?xf64>) {
    %cst = constant 0.000000e+00 : f64
    %0 = index_cast %arg0 : i32 to index
    %1 = index_cast %arg1 : i32 to index
    %2 = index_cast %arg2 : i32 to index
    %3 = index_cast %arg3 : i32 to index
    %4 = index_cast %arg4 : i32 to index
    affine.if #set0()[%3] {
      affine.for %arg15 = 0 to #map12()[%0] {
        affine.for %arg16 = 0 to #map12()[%3] {
          affine.for %arg17 = #map0(%arg15) to min #map13(%arg15)[%0] {
            affine.for %arg18 = #map0(%arg16) to min #map13(%arg16)[%3] {
              affine.store %cst, %arg11[%arg17, %arg18] : memref<?x2200xf64>
            }
          }
        }
      } {scop.parallelizable}
    }
    affine.if #set0()[%3] {
      affine.for %arg15 = 0 to #map12()[%1] {
        affine.for %arg16 = 0 to #map12()[%3] {
          affine.for %arg17 = #map0(%arg15) to min #map13(%arg15)[%1] {
            affine.for %arg18 = #map0(%arg16) to min #map13(%arg16)[%3] {
              affine.store %cst, %arg8[%arg17, %arg18] : memref<?x2200xf64>
            }
          }
        }
      } {scop.parallelizable}
    }
    affine.if #set0()[%1] {
      affine.for %arg15 = 0 to #map12()[%0] {
        affine.for %arg16 = 0 to #map12()[%1] {
          affine.for %arg17 = #map0(%arg15) to min #map13(%arg15)[%0] {
            affine.for %arg18 = #map0(%arg16) to min #map13(%arg16)[%1] {
              affine.store %cst, %arg5[%arg17, %arg18] : memref<?x1800xf64>
            }
          }
        }
      } {scop.parallelizable}
    }
    affine.if #set1()[%0, %1, %2] {
      affine.for %arg15 = 0 to #map14()[%0, %2] {
        affine.for %arg16 = max #map15(%arg15)[%2] to min #map16(%arg15)[%0] {
          affine.for %arg17 = #map6(%arg15, %arg16) to min #map17(%arg15, %arg16)[%2] {
            affine.for %arg18 = #map0(%arg16) to min #map13(%arg16)[%0] {
              affine.for %arg19 = 0 to %1 {
                %5 = affine.load %arg6[%arg18, %arg17] : memref<?x2000xf64>
                %6 = affine.load %arg7[%arg17, %arg19] : memref<?x1800xf64>
                %7 = mulf %5, %6 {scop.splittable = 0 : index} : f64
                affine.store %7, %arg12[%arg17] : memref<?xf64>
                %8 = affine.load %arg5[%arg18, %arg19] : memref<?x1800xf64>
                %9 = affine.load %arg12[%arg17] : memref<?xf64>
                %10 = addf %8, %9 : f64
                affine.store %10, %arg5[%arg18, %arg19] : memref<?x1800xf64>
              }
            }
          }
        } {scop.parallelizable}
      }
    }
    affine.if #set1()[%1, %3, %4] {
      affine.for %arg15 = 0 to #map14()[%1, %4] {
        affine.for %arg16 = max #map15(%arg15)[%1] to min #map16(%arg15)[%4] {
          affine.for %arg17 = #map6(%arg15, %arg16) to min #map17(%arg15, %arg16)[%1] {
            affine.for %arg18 = #map0(%arg16) to min #map13(%arg16)[%4] {
              affine.for %arg19 = 0 to %3 {
                %5 = affine.load %arg9[%arg17, %arg18] : memref<?x2400xf64>
                %6 = affine.load %arg10[%arg18, %arg19] : memref<?x2200xf64>
                %7 = mulf %5, %6 {scop.splittable = 1 : index} : f64
                affine.store %7, %arg12[%arg18] : memref<?xf64>
                %8 = affine.load %arg8[%arg17, %arg19] : memref<?x2200xf64>
                %9 = affine.load %arg12[%arg18] : memref<?xf64>
                %10 = addf %8, %9 : f64
                affine.store %10, %arg8[%arg17, %arg19] : memref<?x2200xf64>
              }
            }
          }
        } {scop.parallelizable}
      }
    }
    affine.if #set1()[%0, %1, %3] {
      affine.for %arg15 = 0 to #map14()[%0, %1] {
        affine.for %arg16 = max #map15(%arg15)[%1] to min #map16(%arg15)[%0] {
          affine.for %arg17 = #map6(%arg15, %arg16) to min #map17(%arg15, %arg16)[%1] {
            affine.for %arg18 = #map0(%arg16) to min #map13(%arg16)[%0] {
              affine.for %arg19 = 0 to %3 {
                %5 = affine.load %arg5[%arg18, %arg17] : memref<?x1800xf64>
                %6 = affine.load %arg8[%arg17, %arg19] : memref<?x2200xf64>
                %7 = mulf %5, %6 {scop.splittable = 2 : index} : f64
                affine.store %7, %arg12[%arg17] : memref<?xf64>
                %8 = affine.load %arg11[%arg18, %arg19] : memref<?x2200xf64>
                %9 = affine.load %arg12[%arg17] : memref<?xf64>
                %10 = addf %8, %9 : f64
                affine.store %10, %arg11[%arg18, %arg19] : memref<?x2200xf64>
              }
            }
          }
        } {scop.parallelizable}
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

