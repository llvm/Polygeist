

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
    %22 = llvm.add %arg7, %8 : !llvm.i64
    %23 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %24 = llvm.mlir.constant(0 : index) : !llvm.i64
    %25 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %26 = llvm.mul %11, %25 : !llvm.i64
    %27 = llvm.add %24, %26 : !llvm.i64
    %28 = llvm.mlir.constant(1 : index) : !llvm.i64
    %29 = llvm.mul %22, %28 : !llvm.i64
    %30 = llvm.add %27, %29 : !llvm.i64
    %31 = llvm.getelementptr %23[%30] : (!llvm.ptr<i32>, !llvm.i64) -> !llvm.ptr<i32>
    %32 = llvm.load %31 : !llvm.ptr<i32>
    %33 = llvm.call @max_score(%21, %32) : (!llvm.i32, !llvm.i32) -> !llvm.i32
    %34 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %35 = llvm.mlir.constant(0 : index) : !llvm.i64
    %36 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %37 = llvm.mul %11, %36 : !llvm.i64
    %38 = llvm.add %35, %37 : !llvm.i64
    %39 = llvm.mlir.constant(1 : index) : !llvm.i64
    %40 = llvm.mul %arg7, %39 : !llvm.i64
    %41 = llvm.add %38, %40 : !llvm.i64
    %42 = llvm.getelementptr %34[%41] : (!llvm.ptr<i32>, !llvm.i64) -> !llvm.ptr<i32>
    llvm.store %33, %42 : !llvm.ptr<i32>
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
    %9 = llvm.mlir.constant(1 : index) : !llvm.i64
    %10 = llvm.mul %arg8, %8 : !llvm.i64
    %11 = llvm.add %10, %arg9 : !llvm.i64
    %12 = llvm.add %11, %8 : !llvm.i64
    %13 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %14 = llvm.mlir.constant(0 : index) : !llvm.i64
    %15 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %16 = llvm.mul %12, %15 : !llvm.i64
    %17 = llvm.add %14, %16 : !llvm.i64
    %18 = llvm.mlir.constant(1 : index) : !llvm.i64
    %19 = llvm.mul %arg7, %18 : !llvm.i64
    %20 = llvm.add %17, %19 : !llvm.i64
    %21 = llvm.getelementptr %13[%20] : (!llvm.ptr<i32>, !llvm.i64) -> !llvm.ptr<i32>
    %22 = llvm.load %21 : !llvm.ptr<i32>
    %23 = llvm.add %12, %9 : !llvm.i64
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
    %34 = llvm.call @max_score(%22, %33) : (!llvm.i32, !llvm.i32) -> !llvm.i32
    %35 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %36 = llvm.mlir.constant(0 : index) : !llvm.i64
    %37 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %38 = llvm.mul %12, %37 : !llvm.i64
    %39 = llvm.add %36, %38 : !llvm.i64
    %40 = llvm.mlir.constant(1 : index) : !llvm.i64
    %41 = llvm.mul %arg7, %40 : !llvm.i64
    %42 = llvm.add %39, %41 : !llvm.i64
    %43 = llvm.getelementptr %35[%42] : (!llvm.ptr<i32>, !llvm.i64) -> !llvm.ptr<i32>
    llvm.store %34, %43 : !llvm.ptr<i32>
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
    %14 = llvm.mlir.constant(1 : index) : !llvm.i64
    %15 = llvm.mlir.constant(-1 : index) : !llvm.i64
    %16 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %17 = llvm.mlir.constant(0 : index) : !llvm.i64
    %18 = llvm.mlir.constant(1 : index) : !llvm.i64
    %19 = llvm.mul %arg7, %18 : !llvm.i64
    %20 = llvm.add %17, %19 : !llvm.i64
    %21 = llvm.getelementptr %16[%20] : (!llvm.ptr<i8>, !llvm.i64) -> !llvm.ptr<i8>
    %22 = llvm.load %21 : !llvm.ptr<i8>
    %23 = llvm.mul %arg8, %15 : !llvm.i64
    %24 = llvm.add %23, %arg9 : !llvm.i64
    %25 = llvm.add %24, %15 : !llvm.i64
    %26 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %27 = llvm.mlir.constant(0 : index) : !llvm.i64
    %28 = llvm.mlir.constant(1 : index) : !llvm.i64
    %29 = llvm.mul %25, %28 : !llvm.i64
    %30 = llvm.add %27, %29 : !llvm.i64
    %31 = llvm.getelementptr %26[%30] : (!llvm.ptr<i8>, !llvm.i64) -> !llvm.ptr<i8>
    %32 = llvm.load %31 : !llvm.ptr<i8>
    %33 = llvm.call @match(%32, %22) : (!llvm.i8, !llvm.i8) -> !llvm.i32
    %34 = llvm.add %25, %14 : !llvm.i64
    %35 = llvm.add %arg7, %15 : !llvm.i64
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
    %46 = llvm.add %45, %33 : !llvm.i32
    %47 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %48 = llvm.mlir.constant(0 : index) : !llvm.i64
    %49 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %50 = llvm.mul %25, %49 : !llvm.i64
    %51 = llvm.add %48, %50 : !llvm.i64
    %52 = llvm.mlir.constant(1 : index) : !llvm.i64
    %53 = llvm.mul %arg7, %52 : !llvm.i64
    %54 = llvm.add %51, %53 : !llvm.i64
    %55 = llvm.getelementptr %47[%54] : (!llvm.ptr<i32>, !llvm.i64) -> !llvm.ptr<i32>
    %56 = llvm.load %55 : !llvm.ptr<i32>
    %57 = llvm.call @max_score(%56, %46) : (!llvm.i32, !llvm.i32) -> !llvm.i32
    %58 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %59 = llvm.mlir.constant(0 : index) : !llvm.i64
    %60 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %61 = llvm.mul %25, %60 : !llvm.i64
    %62 = llvm.add %59, %61 : !llvm.i64
    %63 = llvm.mlir.constant(1 : index) : !llvm.i64
    %64 = llvm.mul %arg7, %63 : !llvm.i64
    %65 = llvm.add %62, %64 : !llvm.i64
    %66 = llvm.getelementptr %58[%65] : (!llvm.ptr<i32>, !llvm.i64) -> !llvm.ptr<i32>
    llvm.store %57, %66 : !llvm.ptr<i32>
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
    %8 = llvm.mlir.constant(1 : index) : !llvm.i64
    %9 = llvm.mlir.constant(-1 : index) : !llvm.i64
    %10 = llvm.mul %arg8, %9 : !llvm.i64
    %11 = llvm.add %10, %arg9 : !llvm.i64
    %12 = llvm.add %11, %9 : !llvm.i64
    %13 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %14 = llvm.mlir.constant(0 : index) : !llvm.i64
    %15 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %16 = llvm.mul %12, %15 : !llvm.i64
    %17 = llvm.add %14, %16 : !llvm.i64
    %18 = llvm.mlir.constant(1 : index) : !llvm.i64
    %19 = llvm.mul %arg7, %18 : !llvm.i64
    %20 = llvm.add %17, %19 : !llvm.i64
    %21 = llvm.getelementptr %13[%20] : (!llvm.ptr<i32>, !llvm.i64) -> !llvm.ptr<i32>
    %22 = llvm.load %21 : !llvm.ptr<i32>
    %23 = llvm.add %12, %8 : !llvm.i64
    %24 = llvm.add %arg7, %9 : !llvm.i64
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
    %35 = llvm.call @max_score(%22, %34) : (!llvm.i32, !llvm.i32) -> !llvm.i32
    %36 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %37 = llvm.mlir.constant(0 : index) : !llvm.i64
    %38 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %39 = llvm.mul %12, %38 : !llvm.i64
    %40 = llvm.add %37, %39 : !llvm.i64
    %41 = llvm.mlir.constant(1 : index) : !llvm.i64
    %42 = llvm.mul %arg7, %41 : !llvm.i64
    %43 = llvm.add %40, %42 : !llvm.i64
    %44 = llvm.getelementptr %36[%43] : (!llvm.ptr<i32>, !llvm.i64) -> !llvm.ptr<i32>
    llvm.store %35, %44 : !llvm.ptr<i32>
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
    %35 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %36 = llvm.mlir.constant(0 : index) : !llvm.i64
    %37 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %38 = llvm.mul %23, %37 : !llvm.i64
    %39 = llvm.add %36, %38 : !llvm.i64
    %40 = llvm.mlir.constant(1 : index) : !llvm.i64
    %41 = llvm.mul %arg7, %40 : !llvm.i64
    %42 = llvm.add %39, %41 : !llvm.i64
    %43 = llvm.getelementptr %35[%42] : (!llvm.ptr<i32>, !llvm.i64) -> !llvm.ptr<i32>
    %44 = llvm.load %43 : !llvm.ptr<i32>
    %45 = llvm.call @max_score(%44, %34) : (!llvm.i32, !llvm.i32) -> !llvm.i32
    %46 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %47 = llvm.mlir.constant(0 : index) : !llvm.i64
    %48 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %49 = llvm.mul %23, %48 : !llvm.i64
    %50 = llvm.add %47, %49 : !llvm.i64
    %51 = llvm.mlir.constant(1 : index) : !llvm.i64
    %52 = llvm.mul %arg7, %51 : !llvm.i64
    %53 = llvm.add %50, %52 : !llvm.i64
    %54 = llvm.getelementptr %46[%53] : (!llvm.ptr<i32>, !llvm.i64) -> !llvm.ptr<i32>
    llvm.store %45, %54 : !llvm.ptr<i32>
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
  llvm.func @pb_nussinov_new(%arg0: !llvm.ptr<i32>, %arg1: !llvm.ptr<i32>, %arg2: !llvm.i64, %arg3: !llvm.i64, %arg4: !llvm.i64, %arg5: !llvm.i64, %arg6: !llvm.i64, %arg7: !llvm.ptr<i8>, %arg8: !llvm.ptr<i8>, %arg9: !llvm.i64, %arg10: !llvm.i64, %arg11: !llvm.i64) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.insertvalue %arg7, %8[0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %10 = llvm.insertvalue %arg8, %9[1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.insertvalue %arg9, %10[2] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %12 = llvm.insertvalue %arg10, %11[3, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %13 = llvm.insertvalue %arg11, %12[4, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
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
    %26 = llvm.extractvalue %13[3, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
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
    %50 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %51 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %52 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %53 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %54 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %55 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %56 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S0(%50, %51, %52, %53, %54, %55, %56, %47, %49, %26) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %57 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %58 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %59 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %60 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %61 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %62 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %63 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S1(%57, %58, %59, %60, %61, %62, %63, %47, %49, %26) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %64 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %65 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %66 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %67 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %68 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %69 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %70 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %71 = llvm.extractvalue %13[0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %72 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %73 = llvm.extractvalue %13[2] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %74 = llvm.extractvalue %13[3, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %75 = llvm.extractvalue %13[4, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S2(%64, %65, %66, %67, %68, %69, %70, %47, %49, %26, %71, %72, %73, %74, %75) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %76 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %77 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %78 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %79 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %80 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %81 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %82 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S3(%76, %77, %78, %79, %80, %81, %82, %47, %49, %26) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %83 = llvm.add %43, %25 : !llvm.i64
    llvm.br ^bb3(%83 : !llvm.i64)
  ^bb5:  // 3 preds: ^bb0, ^bb1, ^bb3
    %84 = llvm.add %26, %15 : !llvm.i64
    %85 = llvm.icmp "sle" %84, %22 : !llvm.i64
    %86 = llvm.sub %22, %84 : !llvm.i64
    %87 = llvm.sub %84, %25 : !llvm.i64
    %88 = llvm.select %85, %86, %87 : !llvm.i1, !llvm.i64
    %89 = llvm.sdiv %88, %23 : !llvm.i64
    %90 = llvm.sub %22, %89 : !llvm.i64
    %91 = llvm.add %89, %25 : !llvm.i64
    %92 = llvm.select %85, %90, %91 : !llvm.i1, !llvm.i64
    %93 = llvm.icmp "sgt" %22, %92 : !llvm.i64
    %94 = llvm.select %93, %22, %92 : !llvm.i1, !llvm.i64
    %95 = llvm.add %26, %24 : !llvm.i64
    %96 = llvm.icmp "slt" %95, %22 : !llvm.i64
    %97 = llvm.sub %24, %95 : !llvm.i64
    %98 = llvm.select %96, %97, %95 : !llvm.i1, !llvm.i64
    %99 = llvm.sdiv %98, %16 : !llvm.i64
    %100 = llvm.sub %24, %99 : !llvm.i64
    %101 = llvm.select %96, %100, %99 : !llvm.i1, !llvm.i64
    %102 = llvm.add %101, %25 : !llvm.i64
    llvm.br ^bb6(%94 : !llvm.i64)
  ^bb6(%103: !llvm.i64):  // 2 preds: ^bb5, ^bb25
    %104 = llvm.icmp "slt" %103, %102 : !llvm.i64
    llvm.cond_br %104, ^bb7, ^bb26
  ^bb7:  // pred: ^bb6
    %105 = llvm.mul %103, %23 : !llvm.i64
    %106 = llvm.mul %26, %24 : !llvm.i64
    %107 = llvm.add %105, %106 : !llvm.i64
    %108 = llvm.add %107, %25 : !llvm.i64
    %109 = llvm.icmp "sle" %108, %22 : !llvm.i64
    %110 = llvm.sub %22, %108 : !llvm.i64
    %111 = llvm.sub %108, %25 : !llvm.i64
    %112 = llvm.select %109, %110, %111 : !llvm.i1, !llvm.i64
    %113 = llvm.sdiv %112, %23 : !llvm.i64
    %114 = llvm.sub %22, %113 : !llvm.i64
    %115 = llvm.add %113, %25 : !llvm.i64
    %116 = llvm.select %109, %114, %115 : !llvm.i1, !llvm.i64
    %117 = llvm.icmp "sgt" %22, %116 : !llvm.i64
    %118 = llvm.select %117, %22, %116 : !llvm.i1, !llvm.i64
    %119 = llvm.add %26, %24 : !llvm.i64
    %120 = llvm.icmp "slt" %119, %22 : !llvm.i64
    %121 = llvm.sub %24, %119 : !llvm.i64
    %122 = llvm.select %120, %121, %119 : !llvm.i1, !llvm.i64
    %123 = llvm.sdiv %122, %23 : !llvm.i64
    %124 = llvm.sub %24, %123 : !llvm.i64
    %125 = llvm.select %120, %124, %123 : !llvm.i1, !llvm.i64
    %126 = llvm.add %125, %25 : !llvm.i64
    %127 = llvm.add %103, %25 : !llvm.i64
    %128 = llvm.icmp "slt" %126, %127 : !llvm.i64
    %129 = llvm.select %128, %126, %127 : !llvm.i1, !llvm.i64
    llvm.br ^bb8(%118 : !llvm.i64)
  ^bb8(%130: !llvm.i64):  // 2 preds: ^bb7, ^bb24
    %131 = llvm.icmp "slt" %130, %129 : !llvm.i64
    llvm.cond_br %131, ^bb9, ^bb25
  ^bb9:  // pred: ^bb8
    %132 = llvm.mul %103, %24 : !llvm.i64
    %133 = llvm.add %26, %18 : !llvm.i64
    %134 = llvm.icmp "slt" %133, %22 : !llvm.i64
    %135 = llvm.sub %24, %133 : !llvm.i64
    %136 = llvm.select %134, %135, %133 : !llvm.i1, !llvm.i64
    %137 = llvm.sdiv %136, %23 : !llvm.i64
    %138 = llvm.sub %24, %137 : !llvm.i64
    %139 = llvm.select %134, %138, %137 : !llvm.i1, !llvm.i64
    %140 = llvm.add %132, %139 : !llvm.i64
    %141 = llvm.icmp "sge" %140, %22 : !llvm.i64
    %142 = llvm.mul %130, %24 : !llvm.i64
    %143 = llvm.icmp "slt" %26, %22 : !llvm.i64
    %144 = llvm.sub %24, %26 : !llvm.i64
    %145 = llvm.select %143, %144, %26 : !llvm.i1, !llvm.i64
    %146 = llvm.sdiv %145, %23 : !llvm.i64
    %147 = llvm.sub %24, %146 : !llvm.i64
    %148 = llvm.select %143, %147, %146 : !llvm.i1, !llvm.i64
    %149 = llvm.add %142, %148 : !llvm.i64
    %150 = llvm.add %149, %24 : !llvm.i64
    %151 = llvm.icmp "sge" %150, %22 : !llvm.i64
    %152 = llvm.and %141, %151 : !llvm.i1
    llvm.cond_br %152, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    %153 = llvm.mul %130, %21 : !llvm.i64
    %154 = llvm.add %153, %26 : !llvm.i64
    %155 = llvm.add %154, %18 : !llvm.i64
    %156 = llvm.mul %130, %23 : !llvm.i64
    %157 = llvm.add %156, %17 : !llvm.i64
    %158 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %159 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %160 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %161 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %162 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %163 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %164 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S0(%158, %159, %160, %161, %162, %163, %164, %155, %157, %26) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %165 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %166 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %167 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %168 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %169 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %170 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %171 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S1(%165, %166, %167, %168, %169, %170, %171, %155, %157, %26) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %172 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %173 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %174 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %175 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %176 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %177 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %178 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %179 = llvm.extractvalue %13[0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %180 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %181 = llvm.extractvalue %13[2] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %182 = llvm.extractvalue %13[3, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %183 = llvm.extractvalue %13[4, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S2(%172, %173, %174, %175, %176, %177, %178, %155, %157, %26, %179, %180, %181, %182, %183) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %184 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %185 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %186 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %187 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %188 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %189 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %190 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S3(%184, %185, %186, %187, %188, %189, %190, %155, %157, %26) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.br ^bb11
  ^bb11:  // 2 preds: ^bb9, ^bb10
    %191 = llvm.mul %130, %24 : !llvm.i64
    %192 = llvm.add %103, %191 : !llvm.i64
    %193 = llvm.icmp "eq" %192, %22 : !llvm.i64
    %194 = llvm.add %26, %18 : !llvm.i64
    %195 = llvm.icmp "sle" %194, %22 : !llvm.i64
    %196 = llvm.sub %22, %194 : !llvm.i64
    %197 = llvm.sub %194, %25 : !llvm.i64
    %198 = llvm.select %195, %196, %197 : !llvm.i1, !llvm.i64
    %199 = llvm.sdiv %198, %23 : !llvm.i64
    %200 = llvm.sub %22, %199 : !llvm.i64
    %201 = llvm.add %199, %25 : !llvm.i64
    %202 = llvm.select %195, %200, %201 : !llvm.i1, !llvm.i64
    %203 = llvm.mul %202, %24 : !llvm.i64
    %204 = llvm.add %103, %203 : !llvm.i64
    %205 = llvm.icmp "sge" %204, %22 : !llvm.i64
    %206 = llvm.and %193, %205 : !llvm.i1
    llvm.cond_br %206, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    %207 = llvm.add %26, %24 : !llvm.i64
    %208 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %209 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %210 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %211 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %212 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %213 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %214 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S0(%208, %209, %210, %211, %212, %213, %214, %25, %207, %26) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %215 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %216 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %217 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %218 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %219 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %220 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %221 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S1(%215, %216, %217, %218, %219, %220, %221, %25, %207, %26) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %222 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %223 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %224 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %225 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %226 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %227 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %228 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %229 = llvm.extractvalue %13[0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %230 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %231 = llvm.extractvalue %13[2] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %232 = llvm.extractvalue %13[3, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %233 = llvm.extractvalue %13[4, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S2(%222, %223, %224, %225, %226, %227, %228, %25, %207, %26, %229, %230, %231, %232, %233) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %234 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %235 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %236 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %237 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %238 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %239 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %240 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S3(%234, %235, %236, %237, %238, %239, %240, %25, %207, %26) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.br ^bb13
  ^bb13:  // 2 preds: ^bb11, ^bb12
    %241 = llvm.mul %103, %23 : !llvm.i64
    %242 = llvm.mul %130, %21 : !llvm.i64
    %243 = llvm.add %241, %242 : !llvm.i64
    %244 = llvm.mul %130, %21 : !llvm.i64
    %245 = llvm.add %244, %26 : !llvm.i64
    %246 = llvm.add %245, %20 : !llvm.i64
    %247 = llvm.icmp "sgt" %19, %243 : !llvm.i64
    %248 = llvm.select %247, %19, %243 : !llvm.i1, !llvm.i64
    %249 = llvm.icmp "sgt" %248, %246 : !llvm.i64
    %250 = llvm.select %249, %248, %246 : !llvm.i1, !llvm.i64
    %251 = llvm.mul %103, %23 : !llvm.i64
    %252 = llvm.mul %130, %21 : !llvm.i64
    %253 = llvm.add %251, %252 : !llvm.i64
    %254 = llvm.add %253, %23 : !llvm.i64
    %255 = llvm.icmp "slt" %26, %254 : !llvm.i64
    %256 = llvm.select %255, %26, %254 : !llvm.i1, !llvm.i64
    llvm.br ^bb14(%250 : !llvm.i64)
  ^bb14(%257: !llvm.i64):  // 2 preds: ^bb13, ^bb23
    %258 = llvm.icmp "slt" %257, %256 : !llvm.i64
    llvm.cond_br %258, ^bb15, ^bb24
  ^bb15:  // pred: ^bb14
    %259 = llvm.mul %257, %24 : !llvm.i64
    %260 = llvm.add %259, %26 : !llvm.i64
    %261 = llvm.icmp "slt" %260, %22 : !llvm.i64
    %262 = llvm.sub %24, %260 : !llvm.i64
    %263 = llvm.select %261, %262, %260 : !llvm.i1, !llvm.i64
    %264 = llvm.sdiv %263, %23 : !llvm.i64
    %265 = llvm.sub %24, %264 : !llvm.i64
    %266 = llvm.select %261, %265, %264 : !llvm.i1, !llvm.i64
    %267 = llvm.mul %130, %24 : !llvm.i64
    %268 = llvm.add %266, %267 : !llvm.i64
    %269 = llvm.icmp "sge" %268, %22 : !llvm.i64
    llvm.cond_br %269, ^bb16, ^bb17
  ^bb16:  // pred: ^bb15
    %270 = llvm.mul %257, %24 : !llvm.i64
    %271 = llvm.add %270, %26 : !llvm.i64
    %272 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %273 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %274 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %275 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %276 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %277 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %278 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S0(%272, %273, %274, %275, %276, %277, %278, %257, %271, %26) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %279 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %280 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %281 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %282 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %283 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %284 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %285 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S1(%279, %280, %281, %282, %283, %284, %285, %257, %271, %26) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %286 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %287 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %288 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %289 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %290 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %291 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %292 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %293 = llvm.extractvalue %13[0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %294 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %295 = llvm.extractvalue %13[2] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %296 = llvm.extractvalue %13[3, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %297 = llvm.extractvalue %13[4, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S2(%286, %287, %288, %289, %290, %291, %292, %257, %271, %26, %293, %294, %295, %296, %297) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %298 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %299 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %300 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %301 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %302 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %303 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %304 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S3(%298, %299, %300, %301, %302, %303, %304, %257, %271, %26) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.br ^bb17
  ^bb17:  // 2 preds: ^bb15, ^bb16
    %305 = llvm.mul %130, %23 : !llvm.i64
    %306 = llvm.mul %257, %24 : !llvm.i64
    %307 = llvm.add %306, %26 : !llvm.i64
    %308 = llvm.add %307, %25 : !llvm.i64
    %309 = llvm.icmp "sgt" %305, %308 : !llvm.i64
    %310 = llvm.select %309, %305, %308 : !llvm.i1, !llvm.i64
    %311 = llvm.mul %130, %23 : !llvm.i64
    %312 = llvm.add %311, %23 : !llvm.i64
    %313 = llvm.icmp "slt" %26, %312 : !llvm.i64
    %314 = llvm.select %313, %26, %312 : !llvm.i1, !llvm.i64
    llvm.br ^bb18(%310 : !llvm.i64)
  ^bb18(%315: !llvm.i64):  // 2 preds: ^bb17, ^bb22
    %316 = llvm.icmp "slt" %315, %314 : !llvm.i64
    llvm.cond_br %316, ^bb19, ^bb23
  ^bb19:  // pred: ^bb18
    %317 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %318 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %319 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %320 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %321 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %322 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %323 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S0(%317, %318, %319, %320, %321, %322, %323, %257, %315, %26) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %324 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %325 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %326 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %327 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %328 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %329 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %330 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S1(%324, %325, %326, %327, %328, %329, %330, %257, %315, %26) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %331 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %332 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %333 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %334 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %335 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %336 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %337 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %338 = llvm.extractvalue %13[0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %339 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %340 = llvm.extractvalue %13[2] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %341 = llvm.extractvalue %13[3, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %342 = llvm.extractvalue %13[4, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S2(%331, %332, %333, %334, %335, %336, %337, %257, %315, %26, %338, %339, %340, %341, %342) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %343 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %344 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %345 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %346 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %347 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %348 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %349 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S3(%343, %344, %345, %346, %347, %348, %349, %257, %315, %26) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %350 = llvm.mul %257, %24 : !llvm.i64
    %351 = llvm.add %350, %26 : !llvm.i64
    llvm.br ^bb20(%351 : !llvm.i64)
  ^bb20(%352: !llvm.i64):  // 2 preds: ^bb19, ^bb21
    %353 = llvm.icmp "slt" %352, %315 : !llvm.i64
    llvm.cond_br %353, ^bb21, ^bb22
  ^bb21:  // pred: ^bb20
    %354 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %355 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %356 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %357 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %358 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %359 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %360 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S4(%354, %355, %356, %357, %358, %359, %360, %257, %315, %26, %352) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %361 = llvm.add %352, %25 : !llvm.i64
    llvm.br ^bb20(%361 : !llvm.i64)
  ^bb22:  // pred: ^bb20
    %362 = llvm.add %315, %25 : !llvm.i64
    llvm.br ^bb18(%362 : !llvm.i64)
  ^bb23:  // pred: ^bb18
    %363 = llvm.add %257, %25 : !llvm.i64
    llvm.br ^bb14(%363 : !llvm.i64)
  ^bb24:  // pred: ^bb14
    %364 = llvm.add %130, %25 : !llvm.i64
    llvm.br ^bb8(%364 : !llvm.i64)
  ^bb25:  // pred: ^bb8
    %365 = llvm.add %103, %25 : !llvm.i64
    llvm.br ^bb6(%365 : !llvm.i64)
  ^bb26:  // pred: ^bb6
    llvm.return
  }
  llvm.func @_mlir_ciface_pb_nussinov_new(%arg0: !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>>, %arg1: !llvm.ptr<struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>>) {
    %0 = llvm.load %arg0 : !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.load %arg1 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>>
    %9 = llvm.extractvalue %8[0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %10 = llvm.extractvalue %8[1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.extractvalue %8[2] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %12 = llvm.extractvalue %8[3, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    %13 = llvm.extractvalue %8[4, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @pb_nussinov_new(%1, %2, %3, %4, %5, %6, %7, %9, %10, %11, %12, %13) : (!llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.return
  }
}
