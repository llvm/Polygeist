

module attributes {llvm.data_layout = ""} {
  llvm.func @max_score(%arg0: !llvm.i32, %arg1: !llvm.i32) -> !llvm.i32 {
    %0 = llvm.icmp "sge" %arg0, %arg1 : !llvm.i32
    %1 = llvm.select %0, %arg0, %arg1 : !llvm.i1, !llvm.i32
    llvm.return %1 : !llvm.i32
  }
  llvm.func @_mlir_ciface_max_score(%arg0: !llvm.i32, %arg1: !llvm.i32) -> !llvm.i32 {
    %0 = llvm.call @max_score(%arg0, %arg1) : (!llvm.i32, !llvm.i32) -> !llvm.i32
    llvm.return %0 : !llvm.i32
  }
  llvm.func @match(%arg0: !llvm.i8, %arg1: !llvm.i8) -> !llvm.i32 {
    %0 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %2 = llvm.mlir.constant(3 : i8) : !llvm.i8
    %3 = llvm.add %arg0, %arg1 : !llvm.i8
    %4 = llvm.icmp "eq" %3, %2 : !llvm.i8
    %5 = llvm.select %4, %1, %0 : !llvm.i1, !llvm.i32
    llvm.return %5 : !llvm.i32
  }
  llvm.func @_mlir_ciface_match(%arg0: !llvm.i8, %arg1: !llvm.i8) -> !llvm.i32 {
    %0 = llvm.call @match(%arg0, %arg1) : (!llvm.i8, !llvm.i8) -> !llvm.i32
    llvm.return %0 : !llvm.i32
  }
  llvm.func @pb_nussinov(%arg0: !llvm.ptr<i8>, %arg1: !llvm.ptr<i8>, %arg2: !llvm.i64, %arg3: !llvm.i64, %arg4: !llvm.i64, %arg5: !llvm.ptr<i32>, %arg6: !llvm.ptr<i32>, %arg7: !llvm.i64, %arg8: !llvm.i64, %arg9: !llvm.i64, %arg10: !llvm.i64, %arg11: !llvm.i64) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %5 = llvm.insertvalue %arg4, %4[4, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %6 = llvm.mlir.undef : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.insertvalue %arg5, %6[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.insertvalue %arg6, %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.insertvalue %arg7, %8[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %10 = llvm.insertvalue %arg8, %9[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %11 = llvm.insertvalue %arg10, %10[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %12 = llvm.insertvalue %arg9, %11[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %13 = llvm.insertvalue %arg11, %12[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %14 = llvm.mlir.constant(0 : index) : !llvm.i64
    %15 = llvm.mlir.constant(-1 : index) : !llvm.i64
    %16 = llvm.mlir.constant(-2 : index) : !llvm.i64
    %17 = llvm.mlir.constant(1 : index) : !llvm.i64
    %18 = llvm.extractvalue %5[3, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.br ^bb1(%14 : !llvm.i64)
  ^bb1(%19: !llvm.i64):  // 2 preds: ^bb0, ^bb17
    %20 = llvm.icmp "slt" %19, %18 : !llvm.i64
    llvm.cond_br %20, ^bb2, ^bb18
  ^bb2:  // pred: ^bb1
    %21 = llvm.mul %19, %15 : !llvm.i64
    %22 = llvm.add %21, %18 : !llvm.i64
    %23 = llvm.add %22, %15 : !llvm.i64
    %24 = llvm.add %23, %17 : !llvm.i64
    llvm.br ^bb3(%24 : !llvm.i64)
  ^bb3(%25: !llvm.i64):  // 2 preds: ^bb2, ^bb16
    %26 = llvm.icmp "slt" %25, %18 : !llvm.i64
    llvm.cond_br %26, ^bb4, ^bb17
  ^bb4:  // pred: ^bb3
    %27 = llvm.add %25, %15 : !llvm.i64
    %28 = llvm.icmp "sge" %27, %14 : !llvm.i64
    llvm.cond_br %28, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %29 = llvm.extractvalue %13[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %30 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %31 = llvm.extractvalue %13[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %32 = llvm.extractvalue %13[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %33 = llvm.extractvalue %13[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %34 = llvm.extractvalue %13[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %35 = llvm.extractvalue %13[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S0(%29, %30, %31, %32, %33, %34, %35, %25, %19, %18) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.br ^bb6
  ^bb6:  // 2 preds: ^bb4, ^bb5
    %36 = llvm.mul %23, %15 : !llvm.i64
    %37 = llvm.add %36, %18 : !llvm.i64
    %38 = llvm.add %37, %16 : !llvm.i64
    %39 = llvm.icmp "sge" %38, %14 : !llvm.i64
    llvm.cond_br %39, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    %40 = llvm.extractvalue %13[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %41 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %42 = llvm.extractvalue %13[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %43 = llvm.extractvalue %13[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %44 = llvm.extractvalue %13[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %45 = llvm.extractvalue %13[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %46 = llvm.extractvalue %13[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S1(%40, %41, %42, %43, %44, %45, %46, %25, %19, %18) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.br ^bb8
  ^bb8:  // 2 preds: ^bb6, ^bb7
    %47 = llvm.add %25, %15 : !llvm.i64
    %48 = llvm.icmp "sge" %47, %14 : !llvm.i64
    llvm.cond_br %48, ^bb9, ^bb13
  ^bb9:  // pred: ^bb8
    %49 = llvm.mul %23, %15 : !llvm.i64
    %50 = llvm.add %49, %18 : !llvm.i64
    %51 = llvm.add %50, %16 : !llvm.i64
    %52 = llvm.icmp "sge" %51, %14 : !llvm.i64
    llvm.cond_br %52, ^bb10, ^bb13
  ^bb10:  // pred: ^bb9
    %53 = llvm.mul %23, %15 : !llvm.i64
    %54 = llvm.add %25, %53 : !llvm.i64
    %55 = llvm.add %54, %16 : !llvm.i64
    %56 = llvm.icmp "sge" %55, %14 : !llvm.i64
    llvm.cond_br %56, ^bb11, ^bb12
  ^bb11:  // pred: ^bb10
    %57 = llvm.extractvalue %13[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %58 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %59 = llvm.extractvalue %13[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %60 = llvm.extractvalue %13[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %61 = llvm.extractvalue %13[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %62 = llvm.extractvalue %13[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %63 = llvm.extractvalue %13[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %64 = llvm.extractvalue %5[0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %65 = llvm.extractvalue %5[1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %66 = llvm.extractvalue %5[2] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %67 = llvm.extractvalue %5[3, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %68 = llvm.extractvalue %5[4, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S2(%57, %58, %59, %60, %61, %62, %63, %25, %19, %18, %64, %65, %66, %67, %68) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.br ^bb13
  ^bb12:  // pred: ^bb10
    %69 = llvm.extractvalue %13[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %70 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %71 = llvm.extractvalue %13[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %72 = llvm.extractvalue %13[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %73 = llvm.extractvalue %13[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %74 = llvm.extractvalue %13[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %75 = llvm.extractvalue %13[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S3(%69, %70, %71, %72, %73, %74, %75, %25, %19, %18) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.br ^bb13
  ^bb13:  // 4 preds: ^bb8, ^bb9, ^bb11, ^bb12
    %76 = llvm.add %23, %17 : !llvm.i64
    llvm.br ^bb14(%76 : !llvm.i64)
  ^bb14(%77: !llvm.i64):  // 2 preds: ^bb13, ^bb15
    %78 = llvm.icmp "slt" %77, %25 : !llvm.i64
    llvm.cond_br %78, ^bb15, ^bb16
  ^bb15:  // pred: ^bb14
    %79 = llvm.extractvalue %13[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %80 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %81 = llvm.extractvalue %13[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %82 = llvm.extractvalue %13[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %83 = llvm.extractvalue %13[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %84 = llvm.extractvalue %13[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %85 = llvm.extractvalue %13[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S4(%79, %80, %81, %82, %83, %84, %85, %25, %19, %18, %77) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %86 = llvm.add %77, %17 : !llvm.i64
    llvm.br ^bb14(%86 : !llvm.i64)
  ^bb16:  // pred: ^bb14
    %87 = llvm.add %25, %17 : !llvm.i64
    llvm.br ^bb3(%87 : !llvm.i64)
  ^bb17:  // pred: ^bb3
    %88 = llvm.add %19, %17 : !llvm.i64
    llvm.br ^bb1(%88 : !llvm.i64)
  ^bb18:  // pred: ^bb1
    llvm.return
  }
  llvm.func @_mlir_ciface_pb_nussinov(%arg0: !llvm.ptr<struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>>, %arg1: !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>>) {
    %0 = llvm.load %arg0 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %5 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %6 = llvm.load %arg1 : !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>>
    %7 = llvm.extractvalue %6[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.extractvalue %6[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.extractvalue %6[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %10 = llvm.extractvalue %6[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %11 = llvm.extractvalue %6[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %12 = llvm.extractvalue %6[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %13 = llvm.extractvalue %6[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @pb_nussinov(%1, %2, %3, %4, %5, %7, %8, %9, %10, %11, %12, %13) : (!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.return
  }
  llvm.func @S0(%arg0: !llvm.ptr<i32>, %arg1: !llvm.ptr<i32>, %arg2: !llvm.i64, %arg3: !llvm.i64, %arg4: !llvm.i64, %arg5: !llvm.i64, %arg6: !llvm.i64, %arg7: !llvm.i64, %arg8: !llvm.i64, %arg9: !llvm.i64) attributes {scop.stmt} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.mlir.constant(-1 : index) : !llvm.i64
    %9 = llvm.mul %arg8, %8 : !llvm.i64
    %10 = llvm.add %9, %arg9 : !llvm.i64
    %11 = llvm.add %10, %8 : !llvm.i64
    %12 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %13 = llvm.mlir.constant(0 : index) : !llvm.i64
    %14 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %15 = llvm.mul %11, %14 : !llvm.i64
    %16 = llvm.add %13, %15 : !llvm.i64
    %17 = llvm.mlir.constant(1 : index) : !llvm.i64
    %18 = llvm.mul %arg7, %17 : !llvm.i64
    %19 = llvm.add %16, %18 : !llvm.i64
    %20 = llvm.getelementptr %12[%19] : (!llvm.ptr<i32>, !llvm.i64) -> !llvm.ptr<i32>
    %21 = llvm.load %20 : !llvm.ptr<i32>
    %22 = llvm.mul %arg8, %8 : !llvm.i64
    %23 = llvm.add %22, %arg9 : !llvm.i64
    %24 = llvm.add %23, %8 : !llvm.i64
    %25 = llvm.add %arg7, %8 : !llvm.i64
    %26 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %27 = llvm.mlir.constant(0 : index) : !llvm.i64
    %28 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %29 = llvm.mul %24, %28 : !llvm.i64
    %30 = llvm.add %27, %29 : !llvm.i64
    %31 = llvm.mlir.constant(1 : index) : !llvm.i64
    %32 = llvm.mul %25, %31 : !llvm.i64
    %33 = llvm.add %30, %32 : !llvm.i64
    %34 = llvm.getelementptr %26[%33] : (!llvm.ptr<i32>, !llvm.i64) -> !llvm.ptr<i32>
    %35 = llvm.load %34 : !llvm.ptr<i32>
    %36 = llvm.call @max_score(%21, %35) : (!llvm.i32, !llvm.i32) -> !llvm.i32
    %37 = llvm.mul %arg8, %8 : !llvm.i64
    %38 = llvm.add %37, %arg9 : !llvm.i64
    %39 = llvm.add %38, %8 : !llvm.i64
    %40 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %41 = llvm.mlir.constant(0 : index) : !llvm.i64
    %42 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %43 = llvm.mul %39, %42 : !llvm.i64
    %44 = llvm.add %41, %43 : !llvm.i64
    %45 = llvm.mlir.constant(1 : index) : !llvm.i64
    %46 = llvm.mul %arg7, %45 : !llvm.i64
    %47 = llvm.add %44, %46 : !llvm.i64
    %48 = llvm.getelementptr %40[%47] : (!llvm.ptr<i32>, !llvm.i64) -> !llvm.ptr<i32>
    llvm.store %36, %48 : !llvm.ptr<i32>
    llvm.return
  }
  llvm.func @_mlir_ciface_S0(%arg0: !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>>, %arg1: !llvm.i64, %arg2: !llvm.i64, %arg3: !llvm.i64) attributes {scop.stmt} {
    %0 = llvm.load %arg0 : !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S0(%1, %2, %3, %4, %5, %6, %7, %arg1, %arg2, %arg3) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.return
  }
  llvm.func @S1(%arg0: !llvm.ptr<i32>, %arg1: !llvm.ptr<i32>, %arg2: !llvm.i64, %arg3: !llvm.i64, %arg4: !llvm.i64, %arg5: !llvm.i64, %arg6: !llvm.i64, %arg7: !llvm.i64, %arg8: !llvm.i64, %arg9: !llvm.i64) attributes {scop.stmt} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.mlir.constant(-1 : index) : !llvm.i64
    %9 = llvm.mul %arg8, %8 : !llvm.i64
    %10 = llvm.add %9, %arg9 : !llvm.i64
    %11 = llvm.add %10, %8 : !llvm.i64
    %12 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %13 = llvm.mlir.constant(0 : index) : !llvm.i64
    %14 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %15 = llvm.mul %11, %14 : !llvm.i64
    %16 = llvm.add %13, %15 : !llvm.i64
    %17 = llvm.mlir.constant(1 : index) : !llvm.i64
    %18 = llvm.mul %arg7, %17 : !llvm.i64
    %19 = llvm.add %16, %18 : !llvm.i64
    %20 = llvm.getelementptr %12[%19] : (!llvm.ptr<i32>, !llvm.i64) -> !llvm.ptr<i32>
    %21 = llvm.load %20 : !llvm.ptr<i32>
    %22 = llvm.mul %arg8, %8 : !llvm.i64
    %23 = llvm.add %22, %arg9 : !llvm.i64
    %24 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %25 = llvm.mlir.constant(0 : index) : !llvm.i64
    %26 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %27 = llvm.mul %23, %26 : !llvm.i64
    %28 = llvm.add %25, %27 : !llvm.i64
    %29 = llvm.mlir.constant(1 : index) : !llvm.i64
    %30 = llvm.mul %arg7, %29 : !llvm.i64
    %31 = llvm.add %28, %30 : !llvm.i64
    %32 = llvm.getelementptr %24[%31] : (!llvm.ptr<i32>, !llvm.i64) -> !llvm.ptr<i32>
    %33 = llvm.load %32 : !llvm.ptr<i32>
    %34 = llvm.call @max_score(%21, %33) : (!llvm.i32, !llvm.i32) -> !llvm.i32
    %35 = llvm.mul %arg8, %8 : !llvm.i64
    %36 = llvm.add %35, %arg9 : !llvm.i64
    %37 = llvm.add %36, %8 : !llvm.i64
    %38 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %39 = llvm.mlir.constant(0 : index) : !llvm.i64
    %40 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %41 = llvm.mul %37, %40 : !llvm.i64
    %42 = llvm.add %39, %41 : !llvm.i64
    %43 = llvm.mlir.constant(1 : index) : !llvm.i64
    %44 = llvm.mul %arg7, %43 : !llvm.i64
    %45 = llvm.add %42, %44 : !llvm.i64
    %46 = llvm.getelementptr %38[%45] : (!llvm.ptr<i32>, !llvm.i64) -> !llvm.ptr<i32>
    llvm.store %34, %46 : !llvm.ptr<i32>
    llvm.return
  }
  llvm.func @_mlir_ciface_S1(%arg0: !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>>, %arg1: !llvm.i64, %arg2: !llvm.i64, %arg3: !llvm.i64) attributes {scop.stmt} {
    %0 = llvm.load %arg0 : !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S1(%1, %2, %3, %4, %5, %6, %7, %arg1, %arg2, %arg3) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.return
  }
  llvm.func @S2(%arg0: !llvm.ptr<i32>, %arg1: !llvm.ptr<i32>, %arg2: !llvm.i64, %arg3: !llvm.i64, %arg4: !llvm.i64, %arg5: !llvm.i64, %arg6: !llvm.i64, %arg7: !llvm.i64, %arg8: !llvm.i64, %arg9: !llvm.i64, %arg10: !llvm.ptr<i8>, %arg11: !llvm.ptr<i8>, %arg12: !llvm.i64, %arg13: !llvm.i64, %arg14: !llvm.i64) attributes {scop.stmt} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.insertvalue %arg10, %8[0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %10 = llvm.insertvalue %arg11, %9[1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.insertvalue %arg12, %10[2] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %12 = llvm.insertvalue %arg13, %11[3, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %13 = llvm.insertvalue %arg14, %12[4, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %14 = llvm.mlir.constant(-1 : index) : !llvm.i64
    %15 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %16 = llvm.mlir.constant(0 : index) : !llvm.i64
    %17 = llvm.mlir.constant(1 : index) : !llvm.i64
    %18 = llvm.mul %arg7, %17 : !llvm.i64
    %19 = llvm.add %16, %18 : !llvm.i64
    %20 = llvm.getelementptr %15[%19] : (!llvm.ptr<i8>, !llvm.i64) -> !llvm.ptr<i8>
    %21 = llvm.load %20 : !llvm.ptr<i8>
    %22 = llvm.mul %arg8, %14 : !llvm.i64
    %23 = llvm.add %22, %arg9 : !llvm.i64
    %24 = llvm.add %23, %14 : !llvm.i64
    %25 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %26 = llvm.mlir.constant(0 : index) : !llvm.i64
    %27 = llvm.mlir.constant(1 : index) : !llvm.i64
    %28 = llvm.mul %24, %27 : !llvm.i64
    %29 = llvm.add %26, %28 : !llvm.i64
    %30 = llvm.getelementptr %25[%29] : (!llvm.ptr<i8>, !llvm.i64) -> !llvm.ptr<i8>
    %31 = llvm.load %30 : !llvm.ptr<i8>
    %32 = llvm.call @match(%31, %21) : (!llvm.i8, !llvm.i8) -> !llvm.i32
    %33 = llvm.mul %arg8, %14 : !llvm.i64
    %34 = llvm.add %33, %arg9 : !llvm.i64
    %35 = llvm.add %arg7, %14 : !llvm.i64
    %36 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %37 = llvm.mlir.constant(0 : index) : !llvm.i64
    %38 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %39 = llvm.mul %34, %38 : !llvm.i64
    %40 = llvm.add %37, %39 : !llvm.i64
    %41 = llvm.mlir.constant(1 : index) : !llvm.i64
    %42 = llvm.mul %35, %41 : !llvm.i64
    %43 = llvm.add %40, %42 : !llvm.i64
    %44 = llvm.getelementptr %36[%43] : (!llvm.ptr<i32>, !llvm.i64) -> !llvm.ptr<i32>
    %45 = llvm.load %44 : !llvm.ptr<i32>
    %46 = llvm.add %45, %32 : !llvm.i32
    %47 = llvm.mul %arg8, %14 : !llvm.i64
    %48 = llvm.add %47, %arg9 : !llvm.i64
    %49 = llvm.add %48, %14 : !llvm.i64
    %50 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %51 = llvm.mlir.constant(0 : index) : !llvm.i64
    %52 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %53 = llvm.mul %49, %52 : !llvm.i64
    %54 = llvm.add %51, %53 : !llvm.i64
    %55 = llvm.mlir.constant(1 : index) : !llvm.i64
    %56 = llvm.mul %arg7, %55 : !llvm.i64
    %57 = llvm.add %54, %56 : !llvm.i64
    %58 = llvm.getelementptr %50[%57] : (!llvm.ptr<i32>, !llvm.i64) -> !llvm.ptr<i32>
    %59 = llvm.load %58 : !llvm.ptr<i32>
    %60 = llvm.call @max_score(%59, %46) : (!llvm.i32, !llvm.i32) -> !llvm.i32
    %61 = llvm.mul %arg8, %14 : !llvm.i64
    %62 = llvm.add %61, %arg9 : !llvm.i64
    %63 = llvm.add %62, %14 : !llvm.i64
    %64 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %65 = llvm.mlir.constant(0 : index) : !llvm.i64
    %66 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %67 = llvm.mul %63, %66 : !llvm.i64
    %68 = llvm.add %65, %67 : !llvm.i64
    %69 = llvm.mlir.constant(1 : index) : !llvm.i64
    %70 = llvm.mul %arg7, %69 : !llvm.i64
    %71 = llvm.add %68, %70 : !llvm.i64
    %72 = llvm.getelementptr %64[%71] : (!llvm.ptr<i32>, !llvm.i64) -> !llvm.ptr<i32>
    llvm.store %60, %72 : !llvm.ptr<i32>
    llvm.return
  }
  llvm.func @_mlir_ciface_S2(%arg0: !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>>, %arg1: !llvm.i64, %arg2: !llvm.i64, %arg3: !llvm.i64, %arg4: !llvm.ptr<struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>>) attributes {scop.stmt} {
    %0 = llvm.load %arg0 : !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.load %arg4 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>>
    %9 = llvm.extractvalue %8[0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %10 = llvm.extractvalue %8[1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.extractvalue %8[2] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %12 = llvm.extractvalue %8[3, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %13 = llvm.extractvalue %8[4, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S2(%1, %2, %3, %4, %5, %6, %7, %arg1, %arg2, %arg3, %9, %10, %11, %12, %13) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.return
  }
  llvm.func @S3(%arg0: !llvm.ptr<i32>, %arg1: !llvm.ptr<i32>, %arg2: !llvm.i64, %arg3: !llvm.i64, %arg4: !llvm.i64, %arg5: !llvm.i64, %arg6: !llvm.i64, %arg7: !llvm.i64, %arg8: !llvm.i64, %arg9: !llvm.i64) attributes {scop.stmt} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.mlir.constant(-1 : index) : !llvm.i64
    %9 = llvm.mul %arg8, %8 : !llvm.i64
    %10 = llvm.add %9, %arg9 : !llvm.i64
    %11 = llvm.add %10, %8 : !llvm.i64
    %12 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %13 = llvm.mlir.constant(0 : index) : !llvm.i64
    %14 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %15 = llvm.mul %11, %14 : !llvm.i64
    %16 = llvm.add %13, %15 : !llvm.i64
    %17 = llvm.mlir.constant(1 : index) : !llvm.i64
    %18 = llvm.mul %arg7, %17 : !llvm.i64
    %19 = llvm.add %16, %18 : !llvm.i64
    %20 = llvm.getelementptr %12[%19] : (!llvm.ptr<i32>, !llvm.i64) -> !llvm.ptr<i32>
    %21 = llvm.load %20 : !llvm.ptr<i32>
    %22 = llvm.mul %arg8, %8 : !llvm.i64
    %23 = llvm.add %22, %arg9 : !llvm.i64
    %24 = llvm.add %arg7, %8 : !llvm.i64
    %25 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %26 = llvm.mlir.constant(0 : index) : !llvm.i64
    %27 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %28 = llvm.mul %23, %27 : !llvm.i64
    %29 = llvm.add %26, %28 : !llvm.i64
    %30 = llvm.mlir.constant(1 : index) : !llvm.i64
    %31 = llvm.mul %24, %30 : !llvm.i64
    %32 = llvm.add %29, %31 : !llvm.i64
    %33 = llvm.getelementptr %25[%32] : (!llvm.ptr<i32>, !llvm.i64) -> !llvm.ptr<i32>
    %34 = llvm.load %33 : !llvm.ptr<i32>
    %35 = llvm.call @max_score(%21, %34) : (!llvm.i32, !llvm.i32) -> !llvm.i32
    %36 = llvm.mul %arg8, %8 : !llvm.i64
    %37 = llvm.add %36, %arg9 : !llvm.i64
    %38 = llvm.add %37, %8 : !llvm.i64
    %39 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %40 = llvm.mlir.constant(0 : index) : !llvm.i64
    %41 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %42 = llvm.mul %38, %41 : !llvm.i64
    %43 = llvm.add %40, %42 : !llvm.i64
    %44 = llvm.mlir.constant(1 : index) : !llvm.i64
    %45 = llvm.mul %arg7, %44 : !llvm.i64
    %46 = llvm.add %43, %45 : !llvm.i64
    %47 = llvm.getelementptr %39[%46] : (!llvm.ptr<i32>, !llvm.i64) -> !llvm.ptr<i32>
    llvm.store %35, %47 : !llvm.ptr<i32>
    llvm.return
  }
  llvm.func @_mlir_ciface_S3(%arg0: !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>>, %arg1: !llvm.i64, %arg2: !llvm.i64, %arg3: !llvm.i64) attributes {scop.stmt} {
    %0 = llvm.load %arg0 : !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S3(%1, %2, %3, %4, %5, %6, %7, %arg1, %arg2, %arg3) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.return
  }
  llvm.func @S4(%arg0: !llvm.ptr<i32>, %arg1: !llvm.ptr<i32>, %arg2: !llvm.i64, %arg3: !llvm.i64, %arg4: !llvm.i64, %arg5: !llvm.i64, %arg6: !llvm.i64, %arg7: !llvm.i64, %arg8: !llvm.i64, %arg9: !llvm.i64, %arg10: !llvm.i64) attributes {scop.stmt} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.mlir.constant(1 : index) : !llvm.i64
    %9 = llvm.mlir.constant(-1 : index) : !llvm.i64
    %10 = llvm.add %arg10, %8 : !llvm.i64
    %11 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %12 = llvm.mlir.constant(0 : index) : !llvm.i64
    %13 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %14 = llvm.mul %10, %13 : !llvm.i64
    %15 = llvm.add %12, %14 : !llvm.i64
    %16 = llvm.mlir.constant(1 : index) : !llvm.i64
    %17 = llvm.mul %arg7, %16 : !llvm.i64
    %18 = llvm.add %15, %17 : !llvm.i64
    %19 = llvm.getelementptr %11[%18] : (!llvm.ptr<i32>, !llvm.i64) -> !llvm.ptr<i32>
    %20 = llvm.load %19 : !llvm.ptr<i32>
    %21 = llvm.mul %arg8, %9 : !llvm.i64
    %22 = llvm.add %21, %arg9 : !llvm.i64
    %23 = llvm.add %22, %9 : !llvm.i64
    %24 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %25 = llvm.mlir.constant(0 : index) : !llvm.i64
    %26 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %27 = llvm.mul %23, %26 : !llvm.i64
    %28 = llvm.add %25, %27 : !llvm.i64
    %29 = llvm.mlir.constant(1 : index) : !llvm.i64
    %30 = llvm.mul %arg10, %29 : !llvm.i64
    %31 = llvm.add %28, %30 : !llvm.i64
    %32 = llvm.getelementptr %24[%31] : (!llvm.ptr<i32>, !llvm.i64) -> !llvm.ptr<i32>
    %33 = llvm.load %32 : !llvm.ptr<i32>
    %34 = llvm.add %33, %20 : !llvm.i32
    %35 = llvm.mul %arg8, %9 : !llvm.i64
    %36 = llvm.add %35, %arg9 : !llvm.i64
    %37 = llvm.add %36, %9 : !llvm.i64
    %38 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %39 = llvm.mlir.constant(0 : index) : !llvm.i64
    %40 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %41 = llvm.mul %37, %40 : !llvm.i64
    %42 = llvm.add %39, %41 : !llvm.i64
    %43 = llvm.mlir.constant(1 : index) : !llvm.i64
    %44 = llvm.mul %arg7, %43 : !llvm.i64
    %45 = llvm.add %42, %44 : !llvm.i64
    %46 = llvm.getelementptr %38[%45] : (!llvm.ptr<i32>, !llvm.i64) -> !llvm.ptr<i32>
    %47 = llvm.load %46 : !llvm.ptr<i32>
    %48 = llvm.call @max_score(%47, %34) : (!llvm.i32, !llvm.i32) -> !llvm.i32
    %49 = llvm.mul %arg8, %9 : !llvm.i64
    %50 = llvm.add %49, %arg9 : !llvm.i64
    %51 = llvm.add %50, %9 : !llvm.i64
    %52 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %53 = llvm.mlir.constant(0 : index) : !llvm.i64
    %54 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %55 = llvm.mul %51, %54 : !llvm.i64
    %56 = llvm.add %53, %55 : !llvm.i64
    %57 = llvm.mlir.constant(1 : index) : !llvm.i64
    %58 = llvm.mul %arg7, %57 : !llvm.i64
    %59 = llvm.add %56, %58 : !llvm.i64
    %60 = llvm.getelementptr %52[%59] : (!llvm.ptr<i32>, !llvm.i64) -> !llvm.ptr<i32>
    llvm.store %48, %60 : !llvm.ptr<i32>
    llvm.return
  }
  llvm.func @_mlir_ciface_S4(%arg0: !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>>, %arg1: !llvm.i64, %arg2: !llvm.i64, %arg3: !llvm.i64, %arg4: !llvm.i64) attributes {scop.stmt} {
    %0 = llvm.load %arg0 : !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S4(%1, %2, %3, %4, %5, %6, %7, %arg1, %arg2, %arg3, %arg4) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.return
  }
  llvm.func @pb_nussinov_new(%arg0: !llvm.ptr<i8>, %arg1: !llvm.ptr<i8>, %arg2: !llvm.i64, %arg3: !llvm.i64, %arg4: !llvm.i64, %arg5: !llvm.ptr<i32>, %arg6: !llvm.ptr<i32>, %arg7: !llvm.i64, %arg8: !llvm.i64, %arg9: !llvm.i64, %arg10: !llvm.i64, %arg11: !llvm.i64) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %5 = llvm.insertvalue %arg4, %4[4, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %6 = llvm.mlir.undef : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.insertvalue %arg5, %6[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.insertvalue %arg6, %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.insertvalue %arg7, %8[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %10 = llvm.insertvalue %arg8, %9[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %11 = llvm.insertvalue %arg10, %10[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %12 = llvm.insertvalue %arg9, %11[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %13 = llvm.insertvalue %arg11, %12[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %14 = llvm.mlir.constant(-62 : index) : !llvm.i64
    %15 = llvm.mlir.constant(-61 : index) : !llvm.i64
    %16 = llvm.mlir.constant(16 : index) : !llvm.i64
    %17 = llvm.mlir.constant(31 : index) : !llvm.i64
    %18 = llvm.mlir.constant(-31 : index) : !llvm.i64
    %19 = llvm.mlir.constant(2 : index) : !llvm.i64
    %20 = llvm.mlir.constant(-30 : index) : !llvm.i64
    %21 = llvm.mlir.constant(-32 : index) : !llvm.i64
    %22 = llvm.mlir.constant(0 : index) : !llvm.i64
    %23 = llvm.mlir.constant(32 : index) : !llvm.i64
    %24 = llvm.mlir.constant(-1 : index) : !llvm.i64
    %25 = llvm.mlir.constant(1 : index) : !llvm.i64
    %26 = llvm.extractvalue %5[3, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %27 = llvm.add %26, %14 : !llvm.i64
    %28 = llvm.icmp "sge" %27, %22 : !llvm.i64
    llvm.cond_br %28, ^bb1, ^bb5
  ^bb1:  // pred: ^bb0
    %29 = llvm.add %26, %19 : !llvm.i64
    %30 = llvm.srem %29, %23 : !llvm.i64
    %31 = llvm.icmp "slt" %30, %22 : !llvm.i64
    %32 = llvm.add %30, %23 : !llvm.i64
    %33 = llvm.select %31, %32, %30 : !llvm.i1, !llvm.i64
    %34 = llvm.icmp "eq" %33, %22 : !llvm.i64
    llvm.cond_br %34, ^bb2, ^bb5
  ^bb2:  // pred: ^bb1
    %35 = llvm.add %26, %14 : !llvm.i64
    %36 = llvm.icmp "slt" %35, %22 : !llvm.i64
    %37 = llvm.sub %24, %35 : !llvm.i64
    %38 = llvm.select %36, %37, %35 : !llvm.i1, !llvm.i64
    %39 = llvm.sdiv %38, %23 : !llvm.i64
    %40 = llvm.sub %24, %39 : !llvm.i64
    %41 = llvm.select %36, %40, %39 : !llvm.i1, !llvm.i64
    %42 = llvm.add %41, %25 : !llvm.i64
    llvm.br ^bb3(%22 : !llvm.i64)
  ^bb3(%43: !llvm.i64):  // 2 preds: ^bb2, ^bb4
    %44 = llvm.icmp "slt" %43, %42 : !llvm.i64
    llvm.cond_br %44, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %45 = llvm.mul %43, %21 : !llvm.i64
    %46 = llvm.add %45, %26 : !llvm.i64
    %47 = llvm.add %46, %18 : !llvm.i64
    %48 = llvm.mul %43, %23 : !llvm.i64
    %49 = llvm.add %48, %17 : !llvm.i64
    %50 = llvm.extractvalue %13[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %51 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %52 = llvm.extractvalue %13[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %53 = llvm.extractvalue %13[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %54 = llvm.extractvalue %13[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %55 = llvm.extractvalue %13[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %56 = llvm.extractvalue %13[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S0(%50, %51, %52, %53, %54, %55, %56, %47, %49, %26) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %57 = llvm.mul %43, %21 : !llvm.i64
    %58 = llvm.add %57, %26 : !llvm.i64
    %59 = llvm.add %58, %18 : !llvm.i64
    %60 = llvm.mul %43, %23 : !llvm.i64
    %61 = llvm.add %60, %17 : !llvm.i64
    %62 = llvm.extractvalue %13[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %63 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %64 = llvm.extractvalue %13[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %65 = llvm.extractvalue %13[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %66 = llvm.extractvalue %13[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %67 = llvm.extractvalue %13[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %68 = llvm.extractvalue %13[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S1(%62, %63, %64, %65, %66, %67, %68, %59, %61, %26) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %69 = llvm.mul %43, %21 : !llvm.i64
    %70 = llvm.add %69, %26 : !llvm.i64
    %71 = llvm.add %70, %18 : !llvm.i64
    %72 = llvm.mul %43, %23 : !llvm.i64
    %73 = llvm.add %72, %17 : !llvm.i64
    %74 = llvm.extractvalue %13[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %75 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %76 = llvm.extractvalue %13[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %77 = llvm.extractvalue %13[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %78 = llvm.extractvalue %13[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %79 = llvm.extractvalue %13[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %80 = llvm.extractvalue %13[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %81 = llvm.extractvalue %5[0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %82 = llvm.extractvalue %5[1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %83 = llvm.extractvalue %5[2] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %84 = llvm.extractvalue %5[3, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %85 = llvm.extractvalue %5[4, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S2(%74, %75, %76, %77, %78, %79, %80, %71, %73, %26, %81, %82, %83, %84, %85) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %86 = llvm.mul %43, %21 : !llvm.i64
    %87 = llvm.add %86, %26 : !llvm.i64
    %88 = llvm.add %87, %18 : !llvm.i64
    %89 = llvm.mul %43, %23 : !llvm.i64
    %90 = llvm.add %89, %17 : !llvm.i64
    %91 = llvm.extractvalue %13[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %92 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %93 = llvm.extractvalue %13[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %94 = llvm.extractvalue %13[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %95 = llvm.extractvalue %13[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %96 = llvm.extractvalue %13[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %97 = llvm.extractvalue %13[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S3(%91, %92, %93, %94, %95, %96, %97, %88, %90, %26) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %98 = llvm.add %43, %25 : !llvm.i64
    llvm.br ^bb3(%98 : !llvm.i64)
  ^bb5:  // 3 preds: ^bb0, ^bb1, ^bb3
    %99 = llvm.add %26, %15 : !llvm.i64
    %100 = llvm.icmp "sle" %99, %22 : !llvm.i64
    %101 = llvm.sub %22, %99 : !llvm.i64
    %102 = llvm.sub %99, %25 : !llvm.i64
    %103 = llvm.select %100, %101, %102 : !llvm.i1, !llvm.i64
    %104 = llvm.sdiv %103, %23 : !llvm.i64
    %105 = llvm.sub %22, %104 : !llvm.i64
    %106 = llvm.add %104, %25 : !llvm.i64
    %107 = llvm.select %100, %105, %106 : !llvm.i1, !llvm.i64
    %108 = llvm.icmp "sgt" %22, %107 : !llvm.i64
    %109 = llvm.select %108, %22, %107 : !llvm.i1, !llvm.i64
    %110 = llvm.add %26, %24 : !llvm.i64
    %111 = llvm.icmp "slt" %110, %22 : !llvm.i64
    %112 = llvm.sub %24, %110 : !llvm.i64
    %113 = llvm.select %111, %112, %110 : !llvm.i1, !llvm.i64
    %114 = llvm.sdiv %113, %16 : !llvm.i64
    %115 = llvm.sub %24, %114 : !llvm.i64
    %116 = llvm.select %111, %115, %114 : !llvm.i1, !llvm.i64
    %117 = llvm.add %116, %25 : !llvm.i64
    llvm.br ^bb6(%109 : !llvm.i64)
  ^bb6(%118: !llvm.i64):  // 2 preds: ^bb5, ^bb25
    %119 = llvm.icmp "slt" %118, %117 : !llvm.i64
    llvm.cond_br %119, ^bb7, ^bb26
  ^bb7:  // pred: ^bb6
    %120 = llvm.mul %118, %23 : !llvm.i64
    %121 = llvm.mul %26, %24 : !llvm.i64
    %122 = llvm.add %120, %121 : !llvm.i64
    %123 = llvm.add %122, %25 : !llvm.i64
    %124 = llvm.icmp "sle" %123, %22 : !llvm.i64
    %125 = llvm.sub %22, %123 : !llvm.i64
    %126 = llvm.sub %123, %25 : !llvm.i64
    %127 = llvm.select %124, %125, %126 : !llvm.i1, !llvm.i64
    %128 = llvm.sdiv %127, %23 : !llvm.i64
    %129 = llvm.sub %22, %128 : !llvm.i64
    %130 = llvm.add %128, %25 : !llvm.i64
    %131 = llvm.select %124, %129, %130 : !llvm.i1, !llvm.i64
    %132 = llvm.icmp "sgt" %22, %131 : !llvm.i64
    %133 = llvm.select %132, %22, %131 : !llvm.i1, !llvm.i64
    %134 = llvm.add %26, %24 : !llvm.i64
    %135 = llvm.icmp "slt" %134, %22 : !llvm.i64
    %136 = llvm.sub %24, %134 : !llvm.i64
    %137 = llvm.select %135, %136, %134 : !llvm.i1, !llvm.i64
    %138 = llvm.sdiv %137, %23 : !llvm.i64
    %139 = llvm.sub %24, %138 : !llvm.i64
    %140 = llvm.select %135, %139, %138 : !llvm.i1, !llvm.i64
    %141 = llvm.add %140, %25 : !llvm.i64
    %142 = llvm.add %118, %25 : !llvm.i64
    %143 = llvm.icmp "slt" %141, %142 : !llvm.i64
    %144 = llvm.select %143, %141, %142 : !llvm.i1, !llvm.i64
    llvm.br ^bb8(%133 : !llvm.i64)
  ^bb8(%145: !llvm.i64):  // 2 preds: ^bb7, ^bb24
    %146 = llvm.icmp "slt" %145, %144 : !llvm.i64
    llvm.cond_br %146, ^bb9, ^bb25
  ^bb9:  // pred: ^bb8
    %147 = llvm.mul %118, %24 : !llvm.i64
    %148 = llvm.add %26, %18 : !llvm.i64
    %149 = llvm.icmp "slt" %148, %22 : !llvm.i64
    %150 = llvm.sub %24, %148 : !llvm.i64
    %151 = llvm.select %149, %150, %148 : !llvm.i1, !llvm.i64
    %152 = llvm.sdiv %151, %23 : !llvm.i64
    %153 = llvm.sub %24, %152 : !llvm.i64
    %154 = llvm.select %149, %153, %152 : !llvm.i1, !llvm.i64
    %155 = llvm.add %147, %154 : !llvm.i64
    %156 = llvm.icmp "sge" %155, %22 : !llvm.i64
    %157 = llvm.mul %145, %24 : !llvm.i64
    %158 = llvm.icmp "slt" %26, %22 : !llvm.i64
    %159 = llvm.sub %24, %26 : !llvm.i64
    %160 = llvm.select %158, %159, %26 : !llvm.i1, !llvm.i64
    %161 = llvm.sdiv %160, %23 : !llvm.i64
    %162 = llvm.sub %24, %161 : !llvm.i64
    %163 = llvm.select %158, %162, %161 : !llvm.i1, !llvm.i64
    %164 = llvm.add %157, %163 : !llvm.i64
    %165 = llvm.add %164, %24 : !llvm.i64
    %166 = llvm.icmp "sge" %165, %22 : !llvm.i64
    %167 = llvm.and %156, %166 : !llvm.i1
    llvm.cond_br %167, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    %168 = llvm.mul %145, %21 : !llvm.i64
    %169 = llvm.add %168, %26 : !llvm.i64
    %170 = llvm.add %169, %18 : !llvm.i64
    %171 = llvm.mul %145, %23 : !llvm.i64
    %172 = llvm.add %171, %17 : !llvm.i64
    %173 = llvm.extractvalue %13[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %174 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %175 = llvm.extractvalue %13[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %176 = llvm.extractvalue %13[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %177 = llvm.extractvalue %13[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %178 = llvm.extractvalue %13[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %179 = llvm.extractvalue %13[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S0(%173, %174, %175, %176, %177, %178, %179, %170, %172, %26) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %180 = llvm.mul %145, %21 : !llvm.i64
    %181 = llvm.add %180, %26 : !llvm.i64
    %182 = llvm.add %181, %18 : !llvm.i64
    %183 = llvm.mul %145, %23 : !llvm.i64
    %184 = llvm.add %183, %17 : !llvm.i64
    %185 = llvm.extractvalue %13[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %186 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %187 = llvm.extractvalue %13[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %188 = llvm.extractvalue %13[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %189 = llvm.extractvalue %13[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %190 = llvm.extractvalue %13[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %191 = llvm.extractvalue %13[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S1(%185, %186, %187, %188, %189, %190, %191, %182, %184, %26) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %192 = llvm.mul %145, %21 : !llvm.i64
    %193 = llvm.add %192, %26 : !llvm.i64
    %194 = llvm.add %193, %18 : !llvm.i64
    %195 = llvm.mul %145, %23 : !llvm.i64
    %196 = llvm.add %195, %17 : !llvm.i64
    %197 = llvm.extractvalue %13[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %198 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %199 = llvm.extractvalue %13[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %200 = llvm.extractvalue %13[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %201 = llvm.extractvalue %13[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %202 = llvm.extractvalue %13[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %203 = llvm.extractvalue %13[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %204 = llvm.extractvalue %5[0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %205 = llvm.extractvalue %5[1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %206 = llvm.extractvalue %5[2] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %207 = llvm.extractvalue %5[3, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %208 = llvm.extractvalue %5[4, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S2(%197, %198, %199, %200, %201, %202, %203, %194, %196, %26, %204, %205, %206, %207, %208) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %209 = llvm.mul %145, %21 : !llvm.i64
    %210 = llvm.add %209, %26 : !llvm.i64
    %211 = llvm.add %210, %18 : !llvm.i64
    %212 = llvm.mul %145, %23 : !llvm.i64
    %213 = llvm.add %212, %17 : !llvm.i64
    %214 = llvm.extractvalue %13[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %215 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %216 = llvm.extractvalue %13[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %217 = llvm.extractvalue %13[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %218 = llvm.extractvalue %13[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %219 = llvm.extractvalue %13[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %220 = llvm.extractvalue %13[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S3(%214, %215, %216, %217, %218, %219, %220, %211, %213, %26) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.br ^bb11
  ^bb11:  // 2 preds: ^bb9, ^bb10
    %221 = llvm.mul %145, %24 : !llvm.i64
    %222 = llvm.add %118, %221 : !llvm.i64
    %223 = llvm.icmp "eq" %222, %22 : !llvm.i64
    %224 = llvm.add %26, %18 : !llvm.i64
    %225 = llvm.icmp "sle" %224, %22 : !llvm.i64
    %226 = llvm.sub %22, %224 : !llvm.i64
    %227 = llvm.sub %224, %25 : !llvm.i64
    %228 = llvm.select %225, %226, %227 : !llvm.i1, !llvm.i64
    %229 = llvm.sdiv %228, %23 : !llvm.i64
    %230 = llvm.sub %22, %229 : !llvm.i64
    %231 = llvm.add %229, %25 : !llvm.i64
    %232 = llvm.select %225, %230, %231 : !llvm.i1, !llvm.i64
    %233 = llvm.mul %232, %24 : !llvm.i64
    %234 = llvm.add %118, %233 : !llvm.i64
    %235 = llvm.icmp "sge" %234, %22 : !llvm.i64
    %236 = llvm.and %223, %235 : !llvm.i1
    llvm.cond_br %236, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    %237 = llvm.add %26, %24 : !llvm.i64
    %238 = llvm.extractvalue %13[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %239 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %240 = llvm.extractvalue %13[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %241 = llvm.extractvalue %13[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %242 = llvm.extractvalue %13[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %243 = llvm.extractvalue %13[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %244 = llvm.extractvalue %13[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S0(%238, %239, %240, %241, %242, %243, %244, %25, %237, %26) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %245 = llvm.add %26, %24 : !llvm.i64
    %246 = llvm.extractvalue %13[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %247 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %248 = llvm.extractvalue %13[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %249 = llvm.extractvalue %13[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %250 = llvm.extractvalue %13[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %251 = llvm.extractvalue %13[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %252 = llvm.extractvalue %13[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S1(%246, %247, %248, %249, %250, %251, %252, %25, %245, %26) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %253 = llvm.add %26, %24 : !llvm.i64
    %254 = llvm.extractvalue %13[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %255 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %256 = llvm.extractvalue %13[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %257 = llvm.extractvalue %13[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %258 = llvm.extractvalue %13[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %259 = llvm.extractvalue %13[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %260 = llvm.extractvalue %13[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %261 = llvm.extractvalue %5[0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %262 = llvm.extractvalue %5[1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %263 = llvm.extractvalue %5[2] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %264 = llvm.extractvalue %5[3, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %265 = llvm.extractvalue %5[4, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S2(%254, %255, %256, %257, %258, %259, %260, %25, %253, %26, %261, %262, %263, %264, %265) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %266 = llvm.add %26, %24 : !llvm.i64
    %267 = llvm.extractvalue %13[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %268 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %269 = llvm.extractvalue %13[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %270 = llvm.extractvalue %13[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %271 = llvm.extractvalue %13[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %272 = llvm.extractvalue %13[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %273 = llvm.extractvalue %13[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S3(%267, %268, %269, %270, %271, %272, %273, %25, %266, %26) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.br ^bb13
  ^bb13:  // 2 preds: ^bb11, ^bb12
    %274 = llvm.mul %118, %23 : !llvm.i64
    %275 = llvm.mul %145, %21 : !llvm.i64
    %276 = llvm.add %274, %275 : !llvm.i64
    %277 = llvm.mul %145, %21 : !llvm.i64
    %278 = llvm.add %277, %26 : !llvm.i64
    %279 = llvm.add %278, %20 : !llvm.i64
    %280 = llvm.icmp "sgt" %19, %276 : !llvm.i64
    %281 = llvm.select %280, %19, %276 : !llvm.i1, !llvm.i64
    %282 = llvm.icmp "sgt" %281, %279 : !llvm.i64
    %283 = llvm.select %282, %281, %279 : !llvm.i1, !llvm.i64
    %284 = llvm.mul %118, %23 : !llvm.i64
    %285 = llvm.mul %145, %21 : !llvm.i64
    %286 = llvm.add %284, %285 : !llvm.i64
    %287 = llvm.add %286, %23 : !llvm.i64
    %288 = llvm.icmp "slt" %26, %287 : !llvm.i64
    %289 = llvm.select %288, %26, %287 : !llvm.i1, !llvm.i64
    llvm.br ^bb14(%283 : !llvm.i64)
  ^bb14(%290: !llvm.i64):  // 2 preds: ^bb13, ^bb23
    %291 = llvm.icmp "slt" %290, %289 : !llvm.i64
    llvm.cond_br %291, ^bb15, ^bb24
  ^bb15:  // pred: ^bb14
    %292 = llvm.mul %290, %24 : !llvm.i64
    %293 = llvm.add %292, %26 : !llvm.i64
    %294 = llvm.icmp "slt" %293, %22 : !llvm.i64
    %295 = llvm.sub %24, %293 : !llvm.i64
    %296 = llvm.select %294, %295, %293 : !llvm.i1, !llvm.i64
    %297 = llvm.sdiv %296, %23 : !llvm.i64
    %298 = llvm.sub %24, %297 : !llvm.i64
    %299 = llvm.select %294, %298, %297 : !llvm.i1, !llvm.i64
    %300 = llvm.mul %145, %24 : !llvm.i64
    %301 = llvm.add %299, %300 : !llvm.i64
    %302 = llvm.icmp "sge" %301, %22 : !llvm.i64
    llvm.cond_br %302, ^bb16, ^bb17
  ^bb16:  // pred: ^bb15
    %303 = llvm.mul %290, %24 : !llvm.i64
    %304 = llvm.add %303, %26 : !llvm.i64
    %305 = llvm.extractvalue %13[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %306 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %307 = llvm.extractvalue %13[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %308 = llvm.extractvalue %13[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %309 = llvm.extractvalue %13[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %310 = llvm.extractvalue %13[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %311 = llvm.extractvalue %13[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S0(%305, %306, %307, %308, %309, %310, %311, %290, %304, %26) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %312 = llvm.mul %290, %24 : !llvm.i64
    %313 = llvm.add %312, %26 : !llvm.i64
    %314 = llvm.extractvalue %13[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %315 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %316 = llvm.extractvalue %13[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %317 = llvm.extractvalue %13[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %318 = llvm.extractvalue %13[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %319 = llvm.extractvalue %13[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %320 = llvm.extractvalue %13[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S1(%314, %315, %316, %317, %318, %319, %320, %290, %313, %26) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %321 = llvm.mul %290, %24 : !llvm.i64
    %322 = llvm.add %321, %26 : !llvm.i64
    %323 = llvm.extractvalue %13[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %324 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %325 = llvm.extractvalue %13[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %326 = llvm.extractvalue %13[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %327 = llvm.extractvalue %13[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %328 = llvm.extractvalue %13[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %329 = llvm.extractvalue %13[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %330 = llvm.extractvalue %5[0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %331 = llvm.extractvalue %5[1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %332 = llvm.extractvalue %5[2] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %333 = llvm.extractvalue %5[3, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %334 = llvm.extractvalue %5[4, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S2(%323, %324, %325, %326, %327, %328, %329, %290, %322, %26, %330, %331, %332, %333, %334) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %335 = llvm.mul %290, %24 : !llvm.i64
    %336 = llvm.add %335, %26 : !llvm.i64
    %337 = llvm.extractvalue %13[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %338 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %339 = llvm.extractvalue %13[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %340 = llvm.extractvalue %13[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %341 = llvm.extractvalue %13[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %342 = llvm.extractvalue %13[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %343 = llvm.extractvalue %13[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S3(%337, %338, %339, %340, %341, %342, %343, %290, %336, %26) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.br ^bb17
  ^bb17:  // 2 preds: ^bb15, ^bb16
    %344 = llvm.mul %145, %23 : !llvm.i64
    %345 = llvm.mul %290, %24 : !llvm.i64
    %346 = llvm.add %345, %26 : !llvm.i64
    %347 = llvm.add %346, %25 : !llvm.i64
    %348 = llvm.icmp "sgt" %344, %347 : !llvm.i64
    %349 = llvm.select %348, %344, %347 : !llvm.i1, !llvm.i64
    %350 = llvm.mul %145, %23 : !llvm.i64
    %351 = llvm.add %350, %23 : !llvm.i64
    %352 = llvm.icmp "slt" %26, %351 : !llvm.i64
    %353 = llvm.select %352, %26, %351 : !llvm.i1, !llvm.i64
    llvm.br ^bb18(%349 : !llvm.i64)
  ^bb18(%354: !llvm.i64):  // 2 preds: ^bb17, ^bb22
    %355 = llvm.icmp "slt" %354, %353 : !llvm.i64
    llvm.cond_br %355, ^bb19, ^bb23
  ^bb19:  // pred: ^bb18
    %356 = llvm.extractvalue %13[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %357 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %358 = llvm.extractvalue %13[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %359 = llvm.extractvalue %13[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %360 = llvm.extractvalue %13[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %361 = llvm.extractvalue %13[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %362 = llvm.extractvalue %13[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S0(%356, %357, %358, %359, %360, %361, %362, %290, %354, %26) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %363 = llvm.extractvalue %13[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %364 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %365 = llvm.extractvalue %13[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %366 = llvm.extractvalue %13[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %367 = llvm.extractvalue %13[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %368 = llvm.extractvalue %13[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %369 = llvm.extractvalue %13[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S1(%363, %364, %365, %366, %367, %368, %369, %290, %354, %26) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %370 = llvm.extractvalue %13[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %371 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %372 = llvm.extractvalue %13[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %373 = llvm.extractvalue %13[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %374 = llvm.extractvalue %13[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %375 = llvm.extractvalue %13[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %376 = llvm.extractvalue %13[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %377 = llvm.extractvalue %5[0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %378 = llvm.extractvalue %5[1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %379 = llvm.extractvalue %5[2] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %380 = llvm.extractvalue %5[3, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %381 = llvm.extractvalue %5[4, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S2(%370, %371, %372, %373, %374, %375, %376, %290, %354, %26, %377, %378, %379, %380, %381) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %382 = llvm.extractvalue %13[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %383 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %384 = llvm.extractvalue %13[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %385 = llvm.extractvalue %13[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %386 = llvm.extractvalue %13[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %387 = llvm.extractvalue %13[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %388 = llvm.extractvalue %13[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S3(%382, %383, %384, %385, %386, %387, %388, %290, %354, %26) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %389 = llvm.mul %290, %24 : !llvm.i64
    %390 = llvm.add %389, %26 : !llvm.i64
    llvm.br ^bb20(%390 : !llvm.i64)
  ^bb20(%391: !llvm.i64):  // 2 preds: ^bb19, ^bb21
    %392 = llvm.icmp "slt" %391, %354 : !llvm.i64
    llvm.cond_br %392, ^bb21, ^bb22
  ^bb21:  // pred: ^bb20
    %393 = llvm.extractvalue %13[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %394 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %395 = llvm.extractvalue %13[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %396 = llvm.extractvalue %13[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %397 = llvm.extractvalue %13[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %398 = llvm.extractvalue %13[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %399 = llvm.extractvalue %13[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S4(%393, %394, %395, %396, %397, %398, %399, %290, %354, %26, %391) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %400 = llvm.add %391, %25 : !llvm.i64
    llvm.br ^bb20(%400 : !llvm.i64)
  ^bb22:  // pred: ^bb20
    %401 = llvm.add %354, %25 : !llvm.i64
    llvm.br ^bb18(%401 : !llvm.i64)
  ^bb23:  // pred: ^bb18
    %402 = llvm.add %290, %25 : !llvm.i64
    llvm.br ^bb14(%402 : !llvm.i64)
  ^bb24:  // pred: ^bb14
    %403 = llvm.add %145, %25 : !llvm.i64
    llvm.br ^bb8(%403 : !llvm.i64)
  ^bb25:  // pred: ^bb8
    %404 = llvm.add %118, %25 : !llvm.i64
    llvm.br ^bb6(%404 : !llvm.i64)
  ^bb26:  // pred: ^bb6
    llvm.return
  }
  llvm.func @_mlir_ciface_pb_nussinov_new(%arg0: !llvm.ptr<struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>>, %arg1: !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>>) {
    %0 = llvm.load %arg0 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %5 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %6 = llvm.load %arg1 : !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>>
    %7 = llvm.extractvalue %6[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.extractvalue %6[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.extractvalue %6[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %10 = llvm.extractvalue %6[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %11 = llvm.extractvalue %6[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %12 = llvm.extractvalue %6[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %13 = llvm.extractvalue %6[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @pb_nussinov_new(%1, %2, %3, %4, %5, %7, %8, %9, %10, %11, %12, %13) : (!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.return
  }
}
