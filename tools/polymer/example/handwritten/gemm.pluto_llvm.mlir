

module attributes {llvm.data_layout = ""} {
  llvm.func @gemm(%arg0: !llvm.float, %arg1: !llvm.float, %arg2: !llvm.ptr<float>, %arg3: !llvm.ptr<float>, %arg4: !llvm.i64, %arg5: !llvm.i64, %arg6: !llvm.i64, %arg7: !llvm.i64, %arg8: !llvm.i64, %arg9: !llvm.ptr<float>, %arg10: !llvm.ptr<float>, %arg11: !llvm.i64, %arg12: !llvm.i64, %arg13: !llvm.i64, %arg14: !llvm.i64, %arg15: !llvm.i64, %arg16: !llvm.ptr<float>, %arg17: !llvm.ptr<float>, %arg18: !llvm.i64, %arg19: !llvm.i64, %arg20: !llvm.i64, %arg21: !llvm.i64, %arg22: !llvm.i64) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg2, %0[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.insertvalue %arg3, %1[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.insertvalue %arg4, %2[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.insertvalue %arg5, %3[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.insertvalue %arg7, %4[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.insertvalue %arg6, %5[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.insertvalue %arg8, %6[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.insertvalue %arg9, %8[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %10 = llvm.insertvalue %arg10, %9[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %11 = llvm.insertvalue %arg11, %10[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %12 = llvm.insertvalue %arg12, %11[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %13 = llvm.insertvalue %arg14, %12[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %14 = llvm.insertvalue %arg13, %13[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %15 = llvm.insertvalue %arg15, %14[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %16 = llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %17 = llvm.insertvalue %arg16, %16[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %18 = llvm.insertvalue %arg17, %17[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %19 = llvm.insertvalue %arg18, %18[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %20 = llvm.insertvalue %arg19, %19[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %21 = llvm.insertvalue %arg21, %20[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %22 = llvm.insertvalue %arg20, %21[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %23 = llvm.insertvalue %arg22, %22[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %24 = llvm.mlir.constant(0 : index) : !llvm.i64
    %25 = llvm.mlir.constant(1 : index) : !llvm.i64
    %26 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %27 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %28 = llvm.extractvalue %15[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.br ^bb1(%24 : !llvm.i64)
  ^bb1(%29: !llvm.i64):  // 2 preds: ^bb0, ^bb8
    %30 = llvm.icmp "slt" %29, %26 : !llvm.i64
    llvm.cond_br %30, ^bb2(%24 : !llvm.i64), ^bb9
  ^bb2(%31: !llvm.i64):  // 2 preds: ^bb1, ^bb3
    %32 = llvm.icmp "slt" %31, %27 : !llvm.i64
    llvm.cond_br %32, ^bb3, ^bb4(%24 : !llvm.i64)
  ^bb3:  // pred: ^bb2
    %33 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %34 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %35 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %36 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %37 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %38 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %39 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S0(%33, %34, %35, %36, %37, %38, %39, %29, %31, %arg1) : (!llvm.ptr<float>, !llvm.ptr<float>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.float) -> ()
    %40 = llvm.add %31, %25 : !llvm.i64
    llvm.br ^bb2(%40 : !llvm.i64)
  ^bb4(%41: !llvm.i64):  // 2 preds: ^bb2, ^bb7
    %42 = llvm.icmp "slt" %41, %27 : !llvm.i64
    llvm.cond_br %42, ^bb5(%24 : !llvm.i64), ^bb8
  ^bb5(%43: !llvm.i64):  // 2 preds: ^bb4, ^bb6
    %44 = llvm.icmp "slt" %43, %28 : !llvm.i64
    llvm.cond_br %44, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %45 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %46 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %47 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %48 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %49 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %50 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %51 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %52 = llvm.extractvalue %23[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %53 = llvm.extractvalue %23[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %54 = llvm.extractvalue %23[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %55 = llvm.extractvalue %23[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %56 = llvm.extractvalue %23[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %57 = llvm.extractvalue %23[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %58 = llvm.extractvalue %23[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %59 = llvm.extractvalue %15[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %60 = llvm.extractvalue %15[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %61 = llvm.extractvalue %15[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %62 = llvm.extractvalue %15[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %63 = llvm.extractvalue %15[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %64 = llvm.extractvalue %15[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %65 = llvm.extractvalue %15[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S1(%45, %46, %47, %48, %49, %50, %51, %29, %41, %52, %53, %54, %55, %56, %57, %58, %43, %arg0, %59, %60, %61, %62, %63, %64, %65) : (!llvm.ptr<float>, !llvm.ptr<float>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<float>, !llvm.ptr<float>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.float, !llvm.ptr<float>, !llvm.ptr<float>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %66 = llvm.add %43, %25 : !llvm.i64
    llvm.br ^bb5(%66 : !llvm.i64)
  ^bb7:  // pred: ^bb5
    %67 = llvm.add %41, %25 : !llvm.i64
    llvm.br ^bb4(%67 : !llvm.i64)
  ^bb8:  // pred: ^bb4
    %68 = llvm.add %29, %25 : !llvm.i64
    llvm.br ^bb1(%68 : !llvm.i64)
  ^bb9:  // pred: ^bb1
    llvm.return
  }
  llvm.func @_mlir_ciface_gemm(%arg0: !llvm.float, %arg1: !llvm.float, %arg2: !llvm.ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>, %arg3: !llvm.ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>, %arg4: !llvm.ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>) {
    %0 = llvm.load %arg2 : !llvm.ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.load %arg3 : !llvm.ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>
    %9 = llvm.extractvalue %8[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %10 = llvm.extractvalue %8[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %11 = llvm.extractvalue %8[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %12 = llvm.extractvalue %8[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %13 = llvm.extractvalue %8[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %14 = llvm.extractvalue %8[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %15 = llvm.extractvalue %8[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %16 = llvm.load %arg4 : !llvm.ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>
    %17 = llvm.extractvalue %16[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %18 = llvm.extractvalue %16[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %19 = llvm.extractvalue %16[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %20 = llvm.extractvalue %16[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %21 = llvm.extractvalue %16[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %22 = llvm.extractvalue %16[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %23 = llvm.extractvalue %16[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @gemm(%arg0, %arg1, %1, %2, %3, %4, %5, %6, %7, %9, %10, %11, %12, %13, %14, %15, %17, %18, %19, %20, %21, %22, %23) : (!llvm.float, !llvm.float, !llvm.ptr<float>, !llvm.ptr<float>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<float>, !llvm.ptr<float>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<float>, !llvm.ptr<float>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.return
  }
  llvm.func @S0(%arg0: !llvm.ptr<float>, %arg1: !llvm.ptr<float>, %arg2: !llvm.i64, %arg3: !llvm.i64, %arg4: !llvm.i64, %arg5: !llvm.i64, %arg6: !llvm.i64, %arg7: !llvm.i64, %arg8: !llvm.i64, %arg9: !llvm.float) attributes {scop.stmt} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.mlir.constant(0 : index) : !llvm.i64
    %10 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %11 = llvm.mul %arg7, %10 : !llvm.i64
    %12 = llvm.add %9, %11 : !llvm.i64
    %13 = llvm.mlir.constant(1 : index) : !llvm.i64
    %14 = llvm.mul %arg8, %13 : !llvm.i64
    %15 = llvm.add %12, %14 : !llvm.i64
    %16 = llvm.getelementptr %8[%15] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
    %17 = llvm.load %16 : !llvm.ptr<float>
    %18 = llvm.fmul %17, %arg9 : !llvm.float
    %19 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %20 = llvm.mlir.constant(0 : index) : !llvm.i64
    %21 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %22 = llvm.mul %arg7, %21 : !llvm.i64
    %23 = llvm.add %20, %22 : !llvm.i64
    %24 = llvm.mlir.constant(1 : index) : !llvm.i64
    %25 = llvm.mul %arg8, %24 : !llvm.i64
    %26 = llvm.add %23, %25 : !llvm.i64
    %27 = llvm.getelementptr %19[%26] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
    llvm.store %18, %27 : !llvm.ptr<float>
    llvm.return
  }
  llvm.func @_mlir_ciface_S0(%arg0: !llvm.ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>, %arg1: !llvm.i64, %arg2: !llvm.i64, %arg3: !llvm.float) attributes {scop.stmt} {
    %0 = llvm.load %arg0 : !llvm.ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S0(%1, %2, %3, %4, %5, %6, %7, %arg1, %arg2, %arg3) : (!llvm.ptr<float>, !llvm.ptr<float>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.float) -> ()
    llvm.return
  }
  llvm.func @S1(%arg0: !llvm.ptr<float>, %arg1: !llvm.ptr<float>, %arg2: !llvm.i64, %arg3: !llvm.i64, %arg4: !llvm.i64, %arg5: !llvm.i64, %arg6: !llvm.i64, %arg7: !llvm.i64, %arg8: !llvm.i64, %arg9: !llvm.ptr<float>, %arg10: !llvm.ptr<float>, %arg11: !llvm.i64, %arg12: !llvm.i64, %arg13: !llvm.i64, %arg14: !llvm.i64, %arg15: !llvm.i64, %arg16: !llvm.i64, %arg17: !llvm.float, %arg18: !llvm.ptr<float>, %arg19: !llvm.ptr<float>, %arg20: !llvm.i64, %arg21: !llvm.i64, %arg22: !llvm.i64, %arg23: !llvm.i64, %arg24: !llvm.i64) attributes {scop.stmt} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.insertvalue %arg9, %8[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %10 = llvm.insertvalue %arg10, %9[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %11 = llvm.insertvalue %arg11, %10[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %12 = llvm.insertvalue %arg12, %11[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %13 = llvm.insertvalue %arg14, %12[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %14 = llvm.insertvalue %arg13, %13[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %15 = llvm.insertvalue %arg15, %14[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %16 = llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %17 = llvm.insertvalue %arg18, %16[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %18 = llvm.insertvalue %arg19, %17[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %19 = llvm.insertvalue %arg20, %18[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %20 = llvm.insertvalue %arg21, %19[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %21 = llvm.insertvalue %arg23, %20[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %22 = llvm.insertvalue %arg22, %21[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %23 = llvm.insertvalue %arg24, %22[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %24 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %25 = llvm.mlir.constant(0 : index) : !llvm.i64
    %26 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %27 = llvm.mul %arg7, %26 : !llvm.i64
    %28 = llvm.add %25, %27 : !llvm.i64
    %29 = llvm.mlir.constant(1 : index) : !llvm.i64
    %30 = llvm.mul %arg8, %29 : !llvm.i64
    %31 = llvm.add %28, %30 : !llvm.i64
    %32 = llvm.getelementptr %24[%31] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
    %33 = llvm.load %32 : !llvm.ptr<float>
    %34 = llvm.extractvalue %23[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %35 = llvm.mlir.constant(0 : index) : !llvm.i64
    %36 = llvm.extractvalue %23[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %37 = llvm.mul %arg7, %36 : !llvm.i64
    %38 = llvm.add %35, %37 : !llvm.i64
    %39 = llvm.mlir.constant(1 : index) : !llvm.i64
    %40 = llvm.mul %arg16, %39 : !llvm.i64
    %41 = llvm.add %38, %40 : !llvm.i64
    %42 = llvm.getelementptr %34[%41] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
    %43 = llvm.load %42 : !llvm.ptr<float>
    %44 = llvm.fmul %arg17, %43 : !llvm.float
    %45 = llvm.extractvalue %15[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %46 = llvm.mlir.constant(0 : index) : !llvm.i64
    %47 = llvm.extractvalue %15[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %48 = llvm.mul %arg16, %47 : !llvm.i64
    %49 = llvm.add %46, %48 : !llvm.i64
    %50 = llvm.mlir.constant(1 : index) : !llvm.i64
    %51 = llvm.mul %arg8, %50 : !llvm.i64
    %52 = llvm.add %49, %51 : !llvm.i64
    %53 = llvm.getelementptr %45[%52] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
    %54 = llvm.load %53 : !llvm.ptr<float>
    %55 = llvm.fmul %44, %54 : !llvm.float
    %56 = llvm.fadd %33, %55 : !llvm.float
    %57 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %58 = llvm.mlir.constant(0 : index) : !llvm.i64
    %59 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %60 = llvm.mul %arg7, %59 : !llvm.i64
    %61 = llvm.add %58, %60 : !llvm.i64
    %62 = llvm.mlir.constant(1 : index) : !llvm.i64
    %63 = llvm.mul %arg8, %62 : !llvm.i64
    %64 = llvm.add %61, %63 : !llvm.i64
    %65 = llvm.getelementptr %57[%64] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
    llvm.store %56, %65 : !llvm.ptr<float>
    llvm.return
  }
  llvm.func @_mlir_ciface_S1(%arg0: !llvm.ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>, %arg1: !llvm.i64, %arg2: !llvm.i64, %arg3: !llvm.ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>, %arg4: !llvm.i64, %arg5: !llvm.float, %arg6: !llvm.ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>) attributes {scop.stmt} {
    %0 = llvm.load %arg0 : !llvm.ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.load %arg3 : !llvm.ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>
    %9 = llvm.extractvalue %8[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %10 = llvm.extractvalue %8[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %11 = llvm.extractvalue %8[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %12 = llvm.extractvalue %8[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %13 = llvm.extractvalue %8[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %14 = llvm.extractvalue %8[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %15 = llvm.extractvalue %8[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %16 = llvm.load %arg6 : !llvm.ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>
    %17 = llvm.extractvalue %16[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %18 = llvm.extractvalue %16[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %19 = llvm.extractvalue %16[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %20 = llvm.extractvalue %16[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %21 = llvm.extractvalue %16[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %22 = llvm.extractvalue %16[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %23 = llvm.extractvalue %16[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S1(%1, %2, %3, %4, %5, %6, %7, %arg1, %arg2, %9, %10, %11, %12, %13, %14, %15, %arg4, %arg5, %17, %18, %19, %20, %21, %22, %23) : (!llvm.ptr<float>, !llvm.ptr<float>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<float>, !llvm.ptr<float>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.float, !llvm.ptr<float>, !llvm.ptr<float>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.return
  }
  llvm.func @gemm_new(%arg0: !llvm.ptr<float>, %arg1: !llvm.ptr<float>, %arg2: !llvm.i64, %arg3: !llvm.i64, %arg4: !llvm.i64, %arg5: !llvm.i64, %arg6: !llvm.i64, %arg7: !llvm.ptr<float>, %arg8: !llvm.ptr<float>, %arg9: !llvm.i64, %arg10: !llvm.i64, %arg11: !llvm.i64, %arg12: !llvm.i64, %arg13: !llvm.i64, %arg14: !llvm.ptr<float>, %arg15: !llvm.ptr<float>, %arg16: !llvm.i64, %arg17: !llvm.i64, %arg18: !llvm.i64, %arg19: !llvm.i64, %arg20: !llvm.i64, %arg21: !llvm.float, %arg22: !llvm.float) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.insertvalue %arg7, %8[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %10 = llvm.insertvalue %arg8, %9[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %11 = llvm.insertvalue %arg9, %10[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %12 = llvm.insertvalue %arg10, %11[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %13 = llvm.insertvalue %arg12, %12[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %14 = llvm.insertvalue %arg11, %13[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %15 = llvm.insertvalue %arg13, %14[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %16 = llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %17 = llvm.insertvalue %arg14, %16[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %18 = llvm.insertvalue %arg15, %17[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %19 = llvm.insertvalue %arg16, %18[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %20 = llvm.insertvalue %arg17, %19[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %21 = llvm.insertvalue %arg19, %20[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %22 = llvm.insertvalue %arg18, %21[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %23 = llvm.insertvalue %arg20, %22[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %24 = llvm.mlir.constant(0 : index) : !llvm.i64
    %25 = llvm.mlir.constant(-1 : index) : !llvm.i64
    %26 = llvm.mlir.constant(32 : index) : !llvm.i64
    %27 = llvm.mlir.constant(1 : index) : !llvm.i64
    %28 = llvm.extractvalue %23[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %29 = llvm.extractvalue %23[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %30 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %31 = llvm.add %29, %25 : !llvm.i64
    %32 = llvm.icmp "slt" %31, %24 : !llvm.i64
    %33 = llvm.sub %25, %31 : !llvm.i64
    %34 = llvm.select %32, %33, %31 : !llvm.i1, !llvm.i64
    %35 = llvm.sdiv %34, %26 : !llvm.i64
    %36 = llvm.sub %25, %35 : !llvm.i64
    %37 = llvm.select %32, %36, %35 : !llvm.i1, !llvm.i64
    %38 = llvm.add %37, %27 : !llvm.i64
    llvm.br ^bb1(%24 : !llvm.i64)
  ^bb1(%39: !llvm.i64):  // 2 preds: ^bb0, ^bb11
    %40 = llvm.icmp "slt" %39, %38 : !llvm.i64
    llvm.cond_br %40, ^bb2, ^bb12
  ^bb2:  // pred: ^bb1
    %41 = llvm.add %28, %25 : !llvm.i64
    %42 = llvm.icmp "slt" %41, %24 : !llvm.i64
    %43 = llvm.sub %25, %41 : !llvm.i64
    %44 = llvm.select %42, %43, %41 : !llvm.i1, !llvm.i64
    %45 = llvm.sdiv %44, %26 : !llvm.i64
    %46 = llvm.sub %25, %45 : !llvm.i64
    %47 = llvm.select %42, %46, %45 : !llvm.i1, !llvm.i64
    %48 = llvm.add %47, %27 : !llvm.i64
    llvm.br ^bb3(%24 : !llvm.i64)
  ^bb3(%49: !llvm.i64):  // 2 preds: ^bb2, ^bb10
    %50 = llvm.icmp "slt" %49, %48 : !llvm.i64
    llvm.cond_br %50, ^bb4, ^bb11
  ^bb4:  // pred: ^bb3
    %51 = llvm.mul %39, %26 : !llvm.i64
    %52 = llvm.mul %39, %26 : !llvm.i64
    %53 = llvm.add %52, %26 : !llvm.i64
    %54 = llvm.icmp "slt" %29, %53 : !llvm.i64
    %55 = llvm.select %54, %29, %53 : !llvm.i1, !llvm.i64
    llvm.br ^bb5(%51 : !llvm.i64)
  ^bb5(%56: !llvm.i64):  // 2 preds: ^bb4, ^bb9
    %57 = llvm.icmp "slt" %56, %55 : !llvm.i64
    llvm.cond_br %57, ^bb6, ^bb10
  ^bb6:  // pred: ^bb5
    %58 = llvm.mul %49, %26 : !llvm.i64
    %59 = llvm.mul %49, %26 : !llvm.i64
    %60 = llvm.add %59, %26 : !llvm.i64
    %61 = llvm.icmp "slt" %28, %60 : !llvm.i64
    %62 = llvm.select %61, %28, %60 : !llvm.i1, !llvm.i64
    llvm.br ^bb7(%58 : !llvm.i64)
  ^bb7(%63: !llvm.i64):  // 2 preds: ^bb6, ^bb8
    %64 = llvm.icmp "slt" %63, %62 : !llvm.i64
    llvm.cond_br %64, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %65 = llvm.extractvalue %23[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %66 = llvm.extractvalue %23[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %67 = llvm.extractvalue %23[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %68 = llvm.extractvalue %23[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %69 = llvm.extractvalue %23[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %70 = llvm.extractvalue %23[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %71 = llvm.extractvalue %23[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S0(%65, %66, %67, %68, %69, %70, %71, %56, %63, %arg22) : (!llvm.ptr<float>, !llvm.ptr<float>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.float) -> ()
    %72 = llvm.add %63, %27 : !llvm.i64
    llvm.br ^bb7(%72 : !llvm.i64)
  ^bb9:  // pred: ^bb7
    %73 = llvm.add %56, %27 : !llvm.i64
    llvm.br ^bb5(%73 : !llvm.i64)
  ^bb10:  // pred: ^bb5
    %74 = llvm.add %49, %27 : !llvm.i64
    llvm.br ^bb3(%74 : !llvm.i64)
  ^bb11:  // pred: ^bb3
    %75 = llvm.add %39, %27 : !llvm.i64
    llvm.br ^bb1(%75 : !llvm.i64)
  ^bb12:  // pred: ^bb1
    %76 = llvm.add %29, %25 : !llvm.i64
    %77 = llvm.icmp "slt" %76, %24 : !llvm.i64
    %78 = llvm.sub %25, %76 : !llvm.i64
    %79 = llvm.select %77, %78, %76 : !llvm.i1, !llvm.i64
    %80 = llvm.sdiv %79, %26 : !llvm.i64
    %81 = llvm.sub %25, %80 : !llvm.i64
    %82 = llvm.select %77, %81, %80 : !llvm.i1, !llvm.i64
    %83 = llvm.add %82, %27 : !llvm.i64
    llvm.br ^bb13(%24 : !llvm.i64)
  ^bb13(%84: !llvm.i64):  // 2 preds: ^bb12, ^bb29
    %85 = llvm.icmp "slt" %84, %83 : !llvm.i64
    llvm.cond_br %85, ^bb14, ^bb30
  ^bb14:  // pred: ^bb13
    %86 = llvm.add %28, %25 : !llvm.i64
    %87 = llvm.icmp "slt" %86, %24 : !llvm.i64
    %88 = llvm.sub %25, %86 : !llvm.i64
    %89 = llvm.select %87, %88, %86 : !llvm.i1, !llvm.i64
    %90 = llvm.sdiv %89, %26 : !llvm.i64
    %91 = llvm.sub %25, %90 : !llvm.i64
    %92 = llvm.select %87, %91, %90 : !llvm.i1, !llvm.i64
    %93 = llvm.add %92, %27 : !llvm.i64
    llvm.br ^bb15(%24 : !llvm.i64)
  ^bb15(%94: !llvm.i64):  // 2 preds: ^bb14, ^bb28
    %95 = llvm.icmp "slt" %94, %93 : !llvm.i64
    llvm.cond_br %95, ^bb16, ^bb29
  ^bb16:  // pred: ^bb15
    %96 = llvm.add %30, %25 : !llvm.i64
    %97 = llvm.icmp "slt" %96, %24 : !llvm.i64
    %98 = llvm.sub %25, %96 : !llvm.i64
    %99 = llvm.select %97, %98, %96 : !llvm.i1, !llvm.i64
    %100 = llvm.sdiv %99, %26 : !llvm.i64
    %101 = llvm.sub %25, %100 : !llvm.i64
    %102 = llvm.select %97, %101, %100 : !llvm.i1, !llvm.i64
    %103 = llvm.add %102, %27 : !llvm.i64
    llvm.br ^bb17(%24 : !llvm.i64)
  ^bb17(%104: !llvm.i64):  // 2 preds: ^bb16, ^bb27
    %105 = llvm.icmp "slt" %104, %103 : !llvm.i64
    llvm.cond_br %105, ^bb18, ^bb28
  ^bb18:  // pred: ^bb17
    %106 = llvm.mul %84, %26 : !llvm.i64
    %107 = llvm.mul %84, %26 : !llvm.i64
    %108 = llvm.add %107, %26 : !llvm.i64
    %109 = llvm.icmp "slt" %29, %108 : !llvm.i64
    %110 = llvm.select %109, %29, %108 : !llvm.i1, !llvm.i64
    llvm.br ^bb19(%106 : !llvm.i64)
  ^bb19(%111: !llvm.i64):  // 2 preds: ^bb18, ^bb26
    %112 = llvm.icmp "slt" %111, %110 : !llvm.i64
    llvm.cond_br %112, ^bb20, ^bb27
  ^bb20:  // pred: ^bb19
    %113 = llvm.mul %94, %26 : !llvm.i64
    %114 = llvm.mul %94, %26 : !llvm.i64
    %115 = llvm.add %114, %26 : !llvm.i64
    %116 = llvm.icmp "slt" %28, %115 : !llvm.i64
    %117 = llvm.select %116, %28, %115 : !llvm.i1, !llvm.i64
    llvm.br ^bb21(%113 : !llvm.i64)
  ^bb21(%118: !llvm.i64):  // 2 preds: ^bb20, ^bb25
    %119 = llvm.icmp "slt" %118, %117 : !llvm.i64
    llvm.cond_br %119, ^bb22, ^bb26
  ^bb22:  // pred: ^bb21
    %120 = llvm.mul %104, %26 : !llvm.i64
    %121 = llvm.mul %104, %26 : !llvm.i64
    %122 = llvm.add %121, %26 : !llvm.i64
    %123 = llvm.icmp "slt" %30, %122 : !llvm.i64
    %124 = llvm.select %123, %30, %122 : !llvm.i1, !llvm.i64
    llvm.br ^bb23(%120 : !llvm.i64)
  ^bb23(%125: !llvm.i64):  // 2 preds: ^bb22, ^bb24
    %126 = llvm.icmp "slt" %125, %124 : !llvm.i64
    llvm.cond_br %126, ^bb24, ^bb25
  ^bb24:  // pred: ^bb23
    %127 = llvm.extractvalue %23[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %128 = llvm.extractvalue %23[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %129 = llvm.extractvalue %23[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %130 = llvm.extractvalue %23[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %131 = llvm.extractvalue %23[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %132 = llvm.extractvalue %23[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %133 = llvm.extractvalue %23[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %134 = llvm.extractvalue %15[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %135 = llvm.extractvalue %15[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %136 = llvm.extractvalue %15[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %137 = llvm.extractvalue %15[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %138 = llvm.extractvalue %15[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %139 = llvm.extractvalue %15[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %140 = llvm.extractvalue %15[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %141 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %142 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %143 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %144 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %145 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %146 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %147 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S1(%127, %128, %129, %130, %131, %132, %133, %111, %118, %134, %135, %136, %137, %138, %139, %140, %125, %arg21, %141, %142, %143, %144, %145, %146, %147) : (!llvm.ptr<float>, !llvm.ptr<float>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<float>, !llvm.ptr<float>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.float, !llvm.ptr<float>, !llvm.ptr<float>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %148 = llvm.add %125, %27 : !llvm.i64
    llvm.br ^bb23(%148 : !llvm.i64)
  ^bb25:  // pred: ^bb23
    %149 = llvm.add %118, %27 : !llvm.i64
    llvm.br ^bb21(%149 : !llvm.i64)
  ^bb26:  // pred: ^bb21
    %150 = llvm.add %111, %27 : !llvm.i64
    llvm.br ^bb19(%150 : !llvm.i64)
  ^bb27:  // pred: ^bb19
    %151 = llvm.add %104, %27 : !llvm.i64
    llvm.br ^bb17(%151 : !llvm.i64)
  ^bb28:  // pred: ^bb17
    %152 = llvm.add %94, %27 : !llvm.i64
    llvm.br ^bb15(%152 : !llvm.i64)
  ^bb29:  // pred: ^bb15
    %153 = llvm.add %84, %27 : !llvm.i64
    llvm.br ^bb13(%153 : !llvm.i64)
  ^bb30:  // pred: ^bb13
    llvm.return
  }
  llvm.func @_mlir_ciface_gemm_new(%arg0: !llvm.ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>, %arg1: !llvm.ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>, %arg2: !llvm.ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>, %arg3: !llvm.float, %arg4: !llvm.float) {
    %0 = llvm.load %arg0 : !llvm.ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.load %arg1 : !llvm.ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>
    %9 = llvm.extractvalue %8[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %10 = llvm.extractvalue %8[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %11 = llvm.extractvalue %8[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %12 = llvm.extractvalue %8[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %13 = llvm.extractvalue %8[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %14 = llvm.extractvalue %8[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %15 = llvm.extractvalue %8[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %16 = llvm.load %arg2 : !llvm.ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>
    %17 = llvm.extractvalue %16[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %18 = llvm.extractvalue %16[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %19 = llvm.extractvalue %16[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %20 = llvm.extractvalue %16[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %21 = llvm.extractvalue %16[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %22 = llvm.extractvalue %16[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %23 = llvm.extractvalue %16[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @gemm_new(%1, %2, %3, %4, %5, %6, %7, %9, %10, %11, %12, %13, %14, %15, %17, %18, %19, %20, %21, %22, %23, %arg3, %arg4) : (!llvm.ptr<float>, !llvm.ptr<float>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<float>, !llvm.ptr<float>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<float>, !llvm.ptr<float>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.float, !llvm.float) -> ()
    llvm.return
  }
}
