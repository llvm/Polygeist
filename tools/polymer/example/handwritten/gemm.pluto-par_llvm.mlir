module attributes {llvm.data_layout = ""}  {
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
  llvm.func @S0(%arg0: !llvm.ptr<float>, %arg1: !llvm.ptr<float>, %arg2: !llvm.i64, %arg3: !llvm.i64, %arg4: !llvm.i64, %arg5: !llvm.i64, %arg6: !llvm.i64, %arg7: !llvm.i64, %arg8: !llvm.i64, %arg9: !llvm.float) attributes {scop.stmt, sym_visibility = "private"} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %10 = llvm.mul %arg7, %9 : !llvm.i64
    %11 = llvm.add %10, %arg8 : !llvm.i64
    %12 = llvm.getelementptr %8[%11] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
    %13 = llvm.load %12 : !llvm.ptr<float>
    %14 = llvm.fmul %13, %arg9 : !llvm.float
    %15 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %16 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %17 = llvm.mul %arg7, %16 : !llvm.i64
    %18 = llvm.add %17, %arg8 : !llvm.i64
    %19 = llvm.getelementptr %15[%18] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
    llvm.store %14, %19 : !llvm.ptr<float>
    llvm.return
  }
  llvm.func @_mlir_ciface_S0(%arg0: !llvm.ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>, %arg1: !llvm.i64, %arg2: !llvm.i64, %arg3: !llvm.float) attributes {scop.stmt, sym_visibility = "private"} {
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
  llvm.func @S1(%arg0: !llvm.ptr<float>, %arg1: !llvm.ptr<float>, %arg2: !llvm.i64, %arg3: !llvm.i64, %arg4: !llvm.i64, %arg5: !llvm.i64, %arg6: !llvm.i64, %arg7: !llvm.i64, %arg8: !llvm.i64, %arg9: !llvm.ptr<float>, %arg10: !llvm.ptr<float>, %arg11: !llvm.i64, %arg12: !llvm.i64, %arg13: !llvm.i64, %arg14: !llvm.i64, %arg15: !llvm.i64, %arg16: !llvm.i64, %arg17: !llvm.float, %arg18: !llvm.ptr<float>, %arg19: !llvm.ptr<float>, %arg20: !llvm.i64, %arg21: !llvm.i64, %arg22: !llvm.i64, %arg23: !llvm.i64, %arg24: !llvm.i64) attributes {scop.stmt, sym_visibility = "private"} {
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
    %25 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %26 = llvm.mul %arg7, %25 : !llvm.i64
    %27 = llvm.add %26, %arg8 : !llvm.i64
    %28 = llvm.getelementptr %24[%27] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
    %29 = llvm.load %28 : !llvm.ptr<float>
    %30 = llvm.extractvalue %23[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %31 = llvm.extractvalue %23[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %32 = llvm.mul %arg7, %31 : !llvm.i64
    %33 = llvm.add %32, %arg16 : !llvm.i64
    %34 = llvm.getelementptr %30[%33] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
    %35 = llvm.load %34 : !llvm.ptr<float>
    %36 = llvm.fmul %arg17, %35 : !llvm.float
    %37 = llvm.extractvalue %15[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %38 = llvm.extractvalue %15[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %39 = llvm.mul %arg16, %38 : !llvm.i64
    %40 = llvm.add %39, %arg8 : !llvm.i64
    %41 = llvm.getelementptr %37[%40] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
    %42 = llvm.load %41 : !llvm.ptr<float>
    %43 = llvm.fmul %36, %42 : !llvm.float
    %44 = llvm.fadd %29, %43 : !llvm.float
    %45 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %46 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %47 = llvm.mul %arg7, %46 : !llvm.i64
    %48 = llvm.add %47, %arg8 : !llvm.i64
    %49 = llvm.getelementptr %45[%48] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
    llvm.store %44, %49 : !llvm.ptr<float>
    llvm.return
  }
  llvm.func @_mlir_ciface_S1(%arg0: !llvm.ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>, %arg1: !llvm.i64, %arg2: !llvm.i64, %arg3: !llvm.ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>, %arg4: !llvm.i64, %arg5: !llvm.float, %arg6: !llvm.ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>) attributes {scop.stmt, sym_visibility = "private"} {
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
  llvm.func @gemm_new(%arg0: !llvm.float, %arg1: !llvm.float, %arg2: !llvm.ptr<float>, %arg3: !llvm.ptr<float>, %arg4: !llvm.i64, %arg5: !llvm.i64, %arg6: !llvm.i64, %arg7: !llvm.i64, %arg8: !llvm.i64, %arg9: !llvm.ptr<float>, %arg10: !llvm.ptr<float>, %arg11: !llvm.i64, %arg12: !llvm.i64, %arg13: !llvm.i64, %arg14: !llvm.i64, %arg15: !llvm.i64, %arg16: !llvm.ptr<float>, %arg17: !llvm.ptr<float>, %arg18: !llvm.i64, %arg19: !llvm.i64, %arg20: !llvm.i64, %arg21: !llvm.i64, %arg22: !llvm.i64) {
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
    %26 = llvm.extractvalue %15[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %27 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %28 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.br ^bb1(%24 : !llvm.i64)
  ^bb1(%29: !llvm.i64):  // 2 preds: ^bb0, ^bb4
    %30 = llvm.icmp "slt" %29, %28 : !llvm.i64
    llvm.cond_br %30, ^bb2(%24 : !llvm.i64), ^bb5(%24 : !llvm.i64)
  ^bb2(%31: !llvm.i64):  // 2 preds: ^bb1, ^bb3
    %32 = llvm.icmp "slt" %31, %27 : !llvm.i64
    llvm.cond_br %32, ^bb3, ^bb4
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
  ^bb4:  // pred: ^bb2
    %41 = llvm.add %29, %25 : !llvm.i64
    llvm.br ^bb1(%41 : !llvm.i64)
  ^bb5(%42: !llvm.i64):  // 2 preds: ^bb1, ^bb10
    %43 = llvm.icmp "slt" %42, %27 : !llvm.i64
    llvm.cond_br %43, ^bb6(%24 : !llvm.i64), ^bb11
  ^bb6(%44: !llvm.i64):  // 2 preds: ^bb5, ^bb9
    %45 = llvm.icmp "slt" %44, %28 : !llvm.i64
    llvm.cond_br %45, ^bb7(%24 : !llvm.i64), ^bb10
  ^bb7(%46: !llvm.i64):  // 2 preds: ^bb6, ^bb8
    %47 = llvm.icmp "slt" %46, %26 : !llvm.i64
    llvm.cond_br %47, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %48 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %49 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %50 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %51 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %52 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %53 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %54 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %55 = llvm.extractvalue %23[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %56 = llvm.extractvalue %23[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %57 = llvm.extractvalue %23[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %58 = llvm.extractvalue %23[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %59 = llvm.extractvalue %23[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %60 = llvm.extractvalue %23[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %61 = llvm.extractvalue %23[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %62 = llvm.extractvalue %15[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %63 = llvm.extractvalue %15[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %64 = llvm.extractvalue %15[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %65 = llvm.extractvalue %15[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %66 = llvm.extractvalue %15[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %67 = llvm.extractvalue %15[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %68 = llvm.extractvalue %15[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S1(%48, %49, %50, %51, %52, %53, %54, %44, %42, %55, %56, %57, %58, %59, %60, %61, %46, %arg0, %62, %63, %64, %65, %66, %67, %68) : (!llvm.ptr<float>, !llvm.ptr<float>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<float>, !llvm.ptr<float>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.float, !llvm.ptr<float>, !llvm.ptr<float>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %69 = llvm.add %46, %25 : !llvm.i64
    llvm.br ^bb7(%69 : !llvm.i64)
  ^bb9:  // pred: ^bb7
    %70 = llvm.add %44, %25 : !llvm.i64
    llvm.br ^bb6(%70 : !llvm.i64)
  ^bb10:  // pred: ^bb6
    %71 = llvm.add %42, %25 : !llvm.i64
    llvm.br ^bb5(%71 : !llvm.i64)
  ^bb11:  // pred: ^bb5
    llvm.return
  }
  llvm.func @_mlir_ciface_gemm_new(%arg0: !llvm.float, %arg1: !llvm.float, %arg2: !llvm.ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>, %arg3: !llvm.ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>, %arg4: !llvm.ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>) {
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
    llvm.call @gemm_new(%arg0, %arg1, %1, %2, %3, %4, %5, %6, %7, %9, %10, %11, %12, %13, %14, %15, %17, %18, %19, %20, %21, %22, %23) : (!llvm.float, !llvm.float, !llvm.ptr<float>, !llvm.ptr<float>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<float>, !llvm.ptr<float>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<float>, !llvm.ptr<float>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.return
  }
}

