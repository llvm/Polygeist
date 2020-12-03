#map0 = affine_map<(d0)[s0] -> (-d0 + s0)>
#map1 = affine_map<(d0) -> (d0 - 1)>
#map2 = affine_map<(d0)[s0] -> (-d0 + s0 - 1)>
#map3 = affine_map<(d0) -> (d0)>
#set0 = affine_set<(d0) : (d0 >= 0)>
#set1 = affine_set<(d0)[s0] : (-d0 + s0 - 1 >= 0)>
#set2 = affine_set<(d0, d1)[s0] : (d0 >= 0, -d1 + s0 - 1 >= 0)>
#set3 = affine_set<(d0, d1) : (d1 - d0 - 1 >= 0)>
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  llvm.mlir.global internal constant @str9("%0.6f\0A\00")
  global_memref "private" @polybench_t_end : memref<1xf64>
  llvm.mlir.global internal constant @str8("Error return from gettimeofday: %d\00")
  llvm.func @printf(!llvm.ptr<i8>, ...) -> !llvm.i32
  llvm.func @gettimeofday(!llvm.ptr<struct<"struct.timeval", (i64, i64)>>, !llvm.ptr<struct<"struct.timezone", (i32, i32)>>) -> !llvm.i32
  global_memref "private" @polybench_t_start : memref<1xf64>
  llvm.mlir.global internal constant @str7("==END   DUMP_ARRAYS==\0A\00")
  llvm.mlir.global internal constant @str6("\0Aend   dump: %s\0A\00")
  llvm.mlir.global internal constant @str5("%d \00")
  llvm.mlir.global internal constant @str4("\0A\00")
  llvm.mlir.global internal constant @str3("table\00")
  llvm.mlir.global internal constant @str2("begin dump: %s\00")
  llvm.mlir.global internal constant @str1("==BEGIN DUMP_ARRAYS==\0A\00")
  llvm.mlir.global external @stderr() : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>
  llvm.func @fprintf(!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, ...) -> !llvm.i32
  llvm.mlir.global internal constant @str0("\00")
  llvm.func @strcmp(!llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
  func @main(%arg0: i32, %arg1: !llvm.ptr<ptr<i8>>) -> i32 {
    %c2500_i32 = constant 2500 : i32
    %c42_i32 = constant 42 : i32
    %true = constant true
    %false = constant false
    %c0_i32 = constant 0 : i32
    %0 = alloc() : memref<2500xi8>
    %1 = alloc() : memref<2500x2500xi32>
    call @init_array(%c2500_i32, %0, %1) : (i32, memref<2500xi8>, memref<2500x2500xi32>) -> ()
    call @polybench_timer_start() : () -> ()
    call @kernel_nussinov(%c2500_i32, %0, %1) : (i32, memref<2500xi8>, memref<2500x2500xi32>) -> ()
    call @polybench_timer_stop() : () -> ()
    call @polybench_timer_print() : () -> ()
    %2 = cmpi "sgt", %arg0, %c42_i32 : i32
    %3 = scf.if %2 -> (i1) {
      %4 = llvm.load %arg1 : !llvm.ptr<ptr<i8>>
      %5 = llvm.mlir.addressof @str0 : !llvm.ptr<array<1 x i8>>
      %6 = llvm.mlir.constant(0 : index) : !llvm.i64
      %7 = llvm.getelementptr %5[%6, %6] : (!llvm.ptr<array<1 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
      %8 = llvm.call @strcmp(%4, %7) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
      %9 = llvm.mlir.cast %8 : !llvm.i32 to i32
      %10 = trunci %9 : i32 to i1
      %11 = xor %10, %true : i1
      scf.yield %11 : i1
    } else {
      scf.yield %false : i1
    }
    scf.if %3 {
      call @print_array(%c2500_i32, %1) : (i32, memref<2500x2500xi32>) -> ()
    }
    return %c0_i32 : i32
  }
  func @init_array(%arg0: i32, %arg1: memref<2500xi8>, %arg2: memref<2500x2500xi32>) {
    %c0_i32 = constant 0 : i32
    %c4_i32 = constant 4 : i32
    %c1_i32 = constant 1 : i32
    br ^bb1(%c0_i32 : i32)
  ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb2
    %1 = cmpi "slt", %0, %arg0 : i32
    %2 = index_cast %0 : i32 to index
    cond_br %1, ^bb2, ^bb3(%c0_i32 : i32)
  ^bb2:  // pred: ^bb1
    %3 = addi %0, %c1_i32 : i32
    %4 = remi_signed %3, %c4_i32 : i32
    %5 = trunci %4 : i32 to i8
    store %5, %arg1[%2] : memref<2500xi8>
    br ^bb1(%3 : i32)
  ^bb3(%6: i32):  // 2 preds: ^bb1, ^bb7
    %7 = cmpi "slt", %6, %arg0 : i32
    %8 = index_cast %6 : i32 to index
    cond_br %7, ^bb5(%c0_i32 : i32), ^bb4
  ^bb4:  // pred: ^bb3
    return
  ^bb5(%9: i32):  // 2 preds: ^bb3, ^bb6
    %10 = cmpi "slt", %9, %arg0 : i32
    %11 = index_cast %9 : i32 to index
    cond_br %10, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    store %c0_i32, %arg2[%8, %11] : memref<2500x2500xi32>
    %12 = addi %9, %c1_i32 : i32
    br ^bb5(%12 : i32)
  ^bb7:  // pred: ^bb5
    %13 = addi %6, %c1_i32 : i32
    br ^bb3(%13 : i32)
  }
  func @polybench_timer_start() {
    %c0 = constant 0 : index
    call @polybench_prepare_instruments() : () -> ()
    %0 = get_global_memref @polybench_t_start : memref<1xf64>
    %1 = call @rtclock() : () -> f64
    store %1, %0[%c0] : memref<1xf64>
    return
  }
  func @kernel_nussinov(%arg0: i32, %arg1: memref<2500xi8>, %arg2: memref<2500x2500xi32>) {
    %c0_i32 = constant 0 : i32
    %c1_i32 = constant 1 : i32
    %c3_i32 = constant 3 : i32
    %0 = index_cast %arg0 : i32 to index
    affine.for %arg3 = 0 to %0 {
      affine.for %arg4 = #map0(%arg3)[%0] to %0 {
        %1 = affine.apply #map1(%arg4)
        affine.if #set0(%1) {
          %3 = affine.load %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<2500x2500xi32>
          %4 = affine.load %arg2[-%arg3 + symbol(%0) - 1, %arg4 - 1] : memref<2500x2500xi32>
          %5 = cmpi "sge", %3, %4 : i32
          %6 = scf.if %5 -> (i32) {
            %7 = affine.load %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<2500x2500xi32>
            scf.yield %7 : i32
          } else {
            %7 = affine.load %arg2[-%arg3 + symbol(%0) - 1, %arg4 - 1] : memref<2500x2500xi32>
            scf.yield %7 : i32
          }
          affine.store %6, %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<2500x2500xi32>
        }
        %2 = affine.apply #map0(%arg3)[%0]
        affine.if #set1(%2)[%0] {
          %3 = affine.load %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<2500x2500xi32>
          %4 = affine.load %arg2[-%arg3 + symbol(%0), %arg4] : memref<2500x2500xi32>
          %5 = cmpi "sge", %3, %4 : i32
          %6 = scf.if %5 -> (i32) {
            %7 = affine.load %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<2500x2500xi32>
            scf.yield %7 : i32
          } else {
            %7 = affine.load %arg2[-%arg3 + symbol(%0), %arg4] : memref<2500x2500xi32>
            scf.yield %7 : i32
          }
          affine.store %6, %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<2500x2500xi32>
        }
        affine.if #set2(%1, %2)[%0] {
          %3 = affine.apply #map2(%arg3)[%0]
          affine.if #set3(%3, %1) {
            %4 = affine.load %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<2500x2500xi32>
            %5 = affine.load %arg2[-%arg3 + symbol(%0), %arg4 - 1] : memref<2500x2500xi32>
            %6 = affine.load %arg1[-%arg3 + symbol(%0) - 1] : memref<2500xi8>
            %7 = sexti %6 : i8 to i32
            %8 = affine.load %arg1[%arg4] : memref<2500xi8>
            %9 = sexti %8 : i8 to i32
            %10 = addi %7, %9 : i32
            %11 = cmpi "eq", %10, %c3_i32 : i32
            %12 = select %11, %c1_i32, %c0_i32 : i32
            %13 = addi %5, %12 : i32
            %14 = cmpi "sge", %4, %13 : i32
            %15 = scf.if %14 -> (i32) {
              %16 = affine.load %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<2500x2500xi32>
              scf.yield %16 : i32
            } else {
              %16 = affine.load %arg2[-%arg3 + symbol(%0), %arg4 - 1] : memref<2500x2500xi32>
              %17 = affine.load %arg1[-%arg3 + symbol(%0) - 1] : memref<2500xi8>
              %18 = sexti %17 : i8 to i32
              %19 = affine.load %arg1[%arg4] : memref<2500xi8>
              %20 = sexti %19 : i8 to i32
              %21 = addi %18, %20 : i32
              %22 = cmpi "eq", %21, %c3_i32 : i32
              %23 = select %22, %c1_i32, %c0_i32 : i32
              %24 = addi %16, %23 : i32
              scf.yield %24 : i32
            }
            affine.store %15, %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<2500x2500xi32>
          } else {
            %4 = affine.load %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<2500x2500xi32>
            %5 = affine.load %arg2[-%arg3 + symbol(%0), %arg4 - 1] : memref<2500x2500xi32>
            %6 = cmpi "sge", %4, %5 : i32
            %7 = scf.if %6 -> (i32) {
              %8 = affine.load %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<2500x2500xi32>
              scf.yield %8 : i32
            } else {
              %8 = affine.load %arg2[-%arg3 + symbol(%0), %arg4 - 1] : memref<2500x2500xi32>
              scf.yield %8 : i32
            }
            affine.store %7, %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<2500x2500xi32>
          }
        }
        affine.for %arg5 = #map0(%arg3)[%0] to #map3(%arg4) {
          %3 = affine.load %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<2500x2500xi32>
          %4 = affine.load %arg2[-%arg3 + symbol(%0) - 1, %arg5] : memref<2500x2500xi32>
          %5 = affine.load %arg2[%arg5 + 1, %arg4] : memref<2500x2500xi32>
          %6 = addi %4, %5 : i32
          %7 = cmpi "sge", %3, %6 : i32
          %8 = scf.if %7 -> (i32) {
            %9 = affine.load %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<2500x2500xi32>
            scf.yield %9 : i32
          } else {
            %9 = affine.load %arg2[-%arg3 + symbol(%0) - 1, %arg5] : memref<2500x2500xi32>
            %10 = affine.load %arg2[%arg5 + 1, %arg4] : memref<2500x2500xi32>
            %11 = addi %9, %10 : i32
            scf.yield %11 : i32
          }
          affine.store %8, %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<2500x2500xi32>
        }
      }
    }
    return
  }
  func @polybench_timer_stop() {
    %c0 = constant 0 : index
    %0 = get_global_memref @polybench_t_end : memref<1xf64>
    %1 = call @rtclock() : () -> f64
    store %1, %0[%c0] : memref<1xf64>
    return
  }
  func @polybench_timer_print() {
    %c0 = constant 0 : index
    %0 = llvm.mlir.addressof @str9 : !llvm.ptr<array<7 x i8>>
    %1 = llvm.mlir.constant(0 : index) : !llvm.i64
    %2 = llvm.getelementptr %0[%1, %1] : (!llvm.ptr<array<7 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %3 = get_global_memref @polybench_t_end : memref<1xf64>
    %4 = load %3[%c0] : memref<1xf64>
    %5 = get_global_memref @polybench_t_start : memref<1xf64>
    %6 = load %5[%c0] : memref<1xf64>
    %7 = subf %4, %6 : f64
    %8 = llvm.mlir.cast %7 : f64 to !llvm.double
    %9 = llvm.call @printf(%2, %8) : (!llvm.ptr<i8>, !llvm.double) -> !llvm.i32
    return
  }
  func @print_array(%arg0: i32, %arg1: memref<2500x2500xi32>) {
    %c0_i32 = constant 0 : i32
    %c20_i32 = constant 20 : i32
    %c1_i32 = constant 1 : i32
    %0 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %1 = llvm.load %0 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %2 = llvm.mlir.addressof @str1 : !llvm.ptr<array<23 x i8>>
    %3 = llvm.mlir.constant(0 : index) : !llvm.i64
    %4 = llvm.getelementptr %2[%3, %3] : (!llvm.ptr<array<23 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %5 = llvm.call @fprintf(%1, %4) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
    %6 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %7 = llvm.load %6 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %8 = llvm.mlir.addressof @str2 : !llvm.ptr<array<15 x i8>>
    %9 = llvm.getelementptr %8[%3, %3] : (!llvm.ptr<array<15 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %10 = llvm.mlir.addressof @str3 : !llvm.ptr<array<6 x i8>>
    %11 = llvm.getelementptr %10[%3, %3] : (!llvm.ptr<array<6 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %12 = llvm.call @fprintf(%7, %9, %11) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
    br ^bb1(%c0_i32, %c0_i32 : i32, i32)
  ^bb1(%13: i32, %14: i32):  // 2 preds: ^bb0, ^bb5
    %15 = cmpi "slt", %13, %arg0 : i32
    %16 = index_cast %13 : i32 to index
    cond_br %15, ^bb3(%13, %14 : i32, i32), ^bb2
  ^bb2:  // pred: ^bb1
    %17 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %18 = llvm.load %17 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %19 = llvm.mlir.addressof @str6 : !llvm.ptr<array<17 x i8>>
    %20 = llvm.getelementptr %19[%3, %3] : (!llvm.ptr<array<17 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %21 = llvm.mlir.addressof @str3 : !llvm.ptr<array<6 x i8>>
    %22 = llvm.getelementptr %21[%3, %3] : (!llvm.ptr<array<6 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %23 = llvm.call @fprintf(%18, %20, %22) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
    %24 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %25 = llvm.load %24 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %26 = llvm.mlir.addressof @str7 : !llvm.ptr<array<23 x i8>>
    %27 = llvm.getelementptr %26[%3, %3] : (!llvm.ptr<array<23 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %28 = llvm.call @fprintf(%25, %27) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
    return
  ^bb3(%29: i32, %30: i32):  // 2 preds: ^bb1, ^bb4
    %31 = cmpi "slt", %29, %arg0 : i32
    %32 = index_cast %29 : i32 to index
    cond_br %31, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %33 = remi_signed %30, %c20_i32 : i32
    %34 = cmpi "eq", %33, %c0_i32 : i32
    scf.if %34 {
      %45 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
      %46 = llvm.load %45 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
      %47 = llvm.mlir.addressof @str4 : !llvm.ptr<array<2 x i8>>
      %48 = llvm.getelementptr %47[%3, %3] : (!llvm.ptr<array<2 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
      %49 = llvm.call @fprintf(%46, %48) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
    }
    %35 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %36 = llvm.load %35 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %37 = llvm.mlir.addressof @str5 : !llvm.ptr<array<4 x i8>>
    %38 = llvm.getelementptr %37[%3, %3] : (!llvm.ptr<array<4 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
    %39 = load %arg1[%16, %32] : memref<2500x2500xi32>
    %40 = llvm.mlir.cast %39 : i32 to !llvm.i32
    %41 = llvm.call @fprintf(%36, %38, %40) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.i32) -> !llvm.i32
    %42 = addi %30, %c1_i32 : i32
    %43 = addi %29, %c1_i32 : i32
    br ^bb3(%43, %42 : i32, i32)
  ^bb5:  // pred: ^bb3
    %44 = addi %13, %c1_i32 : i32
    br ^bb1(%44, %30 : i32, i32)
  }
  func private @free(memref<?xi8>)
  func @polybench_prepare_instruments() {
    return
  }
  func @rtclock() -> f64 {
    %c0_i32 = constant 0 : i32
    %cst = constant 9.9999999999999995E-7 : f64
    %0 = llvm.mlir.constant(1 : index) : !llvm.i64
    %1 = llvm.alloca %0 x !llvm.struct<"struct.timeval", (i64, i64)> : (!llvm.i64) -> !llvm.ptr<struct<"struct.timeval", (i64, i64)>>
    %2 = llvm.mlir.null : !llvm.ptr<struct<"struct.timezone", (i32, i32)>>
    %3 = llvm.call @gettimeofday(%1, %2) : (!llvm.ptr<struct<"struct.timeval", (i64, i64)>>, !llvm.ptr<struct<"struct.timezone", (i32, i32)>>) -> !llvm.i32
    %4 = llvm.mlir.cast %3 : !llvm.i32 to i32
    %5 = llvm.load %1 : !llvm.ptr<struct<"struct.timeval", (i64, i64)>>
    %6 = llvm.extractvalue %5[0] : !llvm.struct<"struct.timeval", (i64, i64)>
    %7 = llvm.mlir.cast %6 : !llvm.i64 to i64
    %8 = llvm.extractvalue %5[1] : !llvm.struct<"struct.timeval", (i64, i64)>
    %9 = llvm.mlir.cast %8 : !llvm.i64 to i64
    %10 = cmpi "ne", %4, %c0_i32 : i32
    scf.if %10 {
      %15 = llvm.mlir.addressof @str8 : !llvm.ptr<array<35 x i8>>
      %16 = llvm.mlir.constant(0 : index) : !llvm.i64
      %17 = llvm.getelementptr %15[%16, %16] : (!llvm.ptr<array<35 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
      %18 = llvm.mlir.cast %4 : i32 to !llvm.i32
      %19 = llvm.call @printf(%17, %18) : (!llvm.ptr<i8>, !llvm.i32) -> !llvm.i32
    }
    %11 = sitofp %7 : i64 to f64
    %12 = sitofp %9 : i64 to f64
    %13 = mulf %12, %cst : f64
    %14 = addf %11, %13 : f64
    return %14 : f64
  }
}

