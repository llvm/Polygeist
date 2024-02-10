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
    %24 = llvm.mlir.constant(-2 : index) : !llvm.i64
    %25 = llvm.mlir.constant(0 : index) : !llvm.i64
    %26 = llvm.mlir.constant(-1 : index) : !llvm.i64
    %27 = llvm.mlir.constant(-32 : index) : !llvm.i64
    %28 = llvm.mlir.constant(32 : index) : !llvm.i64
    %29 = llvm.mlir.constant(1 : index) : !llvm.i64
    %30 = llvm.extractvalue %15[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %31 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %32 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %33 = llvm.add %32, %26 : !llvm.i64
    %34 = llvm.icmp "slt" %33, %25 : !llvm.i64
    %35 = llvm.sub %26, %33 : !llvm.i64
    %36 = llvm.select %34, %35, %33 : !llvm.i1, !llvm.i64
    %37 = llvm.sdiv %36, %28 : !llvm.i64
    %38 = llvm.sub %26, %37 : !llvm.i64
    %39 = llvm.select %34, %38, %37 : !llvm.i1, !llvm.i64
    %40 = llvm.add %39, %29 : !llvm.i64
    llvm.br ^bb1(%25 : !llvm.i64)
  ^bb1(%41: !llvm.i64):  // 2 preds: ^bb0, ^bb11
    %42 = llvm.icmp "slt" %41, %40 : !llvm.i64
    llvm.cond_br %42, ^bb2, ^bb12
  ^bb2:  // pred: ^bb1
    %43 = llvm.add %31, %26 : !llvm.i64
    %44 = llvm.icmp "slt" %43, %25 : !llvm.i64
    %45 = llvm.sub %26, %43 : !llvm.i64
    %46 = llvm.select %44, %45, %43 : !llvm.i1, !llvm.i64
    %47 = llvm.sdiv %46, %28 : !llvm.i64
    %48 = llvm.sub %26, %47 : !llvm.i64
    %49 = llvm.select %44, %48, %47 : !llvm.i1, !llvm.i64
    %50 = llvm.add %49, %29 : !llvm.i64
    llvm.br ^bb3(%25 : !llvm.i64)
  ^bb3(%51: !llvm.i64):  // 2 preds: ^bb2, ^bb10
    %52 = llvm.icmp "slt" %51, %50 : !llvm.i64
    llvm.cond_br %52, ^bb4, ^bb11
  ^bb4:  // pred: ^bb3
    %53 = llvm.mul %41, %28 : !llvm.i64
    %54 = llvm.mul %41, %28 : !llvm.i64
    %55 = llvm.add %54, %28 : !llvm.i64
    %56 = llvm.icmp "slt" %32, %55 : !llvm.i64
    %57 = llvm.select %56, %32, %55 : !llvm.i1, !llvm.i64
    llvm.br ^bb5(%53 : !llvm.i64)
  ^bb5(%58: !llvm.i64):  // 2 preds: ^bb4, ^bb9
    %59 = llvm.icmp "slt" %58, %57 : !llvm.i64
    llvm.cond_br %59, ^bb6, ^bb10
  ^bb6:  // pred: ^bb5
    %60 = llvm.mul %51, %28 : !llvm.i64
    %61 = llvm.mul %51, %28 : !llvm.i64
    %62 = llvm.add %61, %28 : !llvm.i64
    %63 = llvm.icmp "slt" %31, %62 : !llvm.i64
    %64 = llvm.select %63, %31, %62 : !llvm.i1, !llvm.i64
    llvm.br ^bb7(%60 : !llvm.i64)
  ^bb7(%65: !llvm.i64):  // 2 preds: ^bb6, ^bb8
    %66 = llvm.icmp "slt" %65, %64 : !llvm.i64
    llvm.cond_br %66, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %67 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %68 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %69 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %70 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %71 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %72 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %73 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S0(%67, %68, %69, %70, %71, %72, %73, %58, %65, %arg1) : (!llvm.ptr<float>, !llvm.ptr<float>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.float) -> ()
    %74 = llvm.add %65, %29 : !llvm.i64
    llvm.br ^bb7(%74 : !llvm.i64)
  ^bb9:  // pred: ^bb7
    %75 = llvm.add %58, %29 : !llvm.i64
    llvm.br ^bb5(%75 : !llvm.i64)
  ^bb10:  // pred: ^bb5
    %76 = llvm.add %51, %29 : !llvm.i64
    llvm.br ^bb3(%76 : !llvm.i64)
  ^bb11:  // pred: ^bb3
    %77 = llvm.add %41, %29 : !llvm.i64
    llvm.br ^bb1(%77 : !llvm.i64)
  ^bb12:  // pred: ^bb1
    %78 = llvm.add %31, %30 : !llvm.i64
    %79 = llvm.add %78, %24 : !llvm.i64
    %80 = llvm.icmp "slt" %79, %25 : !llvm.i64
    %81 = llvm.sub %26, %79 : !llvm.i64
    %82 = llvm.select %80, %81, %79 : !llvm.i1, !llvm.i64
    %83 = llvm.sdiv %82, %28 : !llvm.i64
    %84 = llvm.sub %26, %83 : !llvm.i64
    %85 = llvm.select %80, %84, %83 : !llvm.i1, !llvm.i64
    %86 = llvm.add %85, %29 : !llvm.i64
    llvm.br ^bb13(%25 : !llvm.i64)
  ^bb13(%87: !llvm.i64):  // 2 preds: ^bb12, ^bb29
    %88 = llvm.icmp "slt" %87, %86 : !llvm.i64
    llvm.cond_br %88, ^bb14, ^bb30
  ^bb14:  // pred: ^bb13
    %89 = llvm.mul %87, %28 : !llvm.i64
    %90 = llvm.mul %30, %26 : !llvm.i64
    %91 = llvm.add %89, %90 : !llvm.i64
    %92 = llvm.add %91, %29 : !llvm.i64
    %93 = llvm.icmp "sle" %92, %25 : !llvm.i64
    %94 = llvm.sub %25, %92 : !llvm.i64
    %95 = llvm.sub %92, %29 : !llvm.i64
    %96 = llvm.select %93, %94, %95 : !llvm.i1, !llvm.i64
    %97 = llvm.sdiv %96, %28 : !llvm.i64
    %98 = llvm.sub %25, %97 : !llvm.i64
    %99 = llvm.add %97, %29 : !llvm.i64
    %100 = llvm.select %93, %98, %99 : !llvm.i1, !llvm.i64
    %101 = llvm.icmp "sgt" %25, %100 : !llvm.i64
    %102 = llvm.select %101, %25, %100 : !llvm.i1, !llvm.i64
    %103 = llvm.add %31, %26 : !llvm.i64
    %104 = llvm.icmp "slt" %103, %25 : !llvm.i64
    %105 = llvm.sub %26, %103 : !llvm.i64
    %106 = llvm.select %104, %105, %103 : !llvm.i1, !llvm.i64
    %107 = llvm.sdiv %106, %28 : !llvm.i64
    %108 = llvm.sub %26, %107 : !llvm.i64
    %109 = llvm.select %104, %108, %107 : !llvm.i1, !llvm.i64
    %110 = llvm.add %109, %29 : !llvm.i64
    %111 = llvm.add %87, %29 : !llvm.i64
    %112 = llvm.icmp "slt" %110, %111 : !llvm.i64
    %113 = llvm.select %112, %110, %111 : !llvm.i1, !llvm.i64
    llvm.br ^bb15(%102 : !llvm.i64)
  ^bb15(%114: !llvm.i64):  // 2 preds: ^bb14, ^bb28
    %115 = llvm.icmp "slt" %114, %113 : !llvm.i64
    llvm.cond_br %115, ^bb16, ^bb29
  ^bb16:  // pred: ^bb15
    %116 = llvm.add %32, %26 : !llvm.i64
    %117 = llvm.icmp "slt" %116, %25 : !llvm.i64
    %118 = llvm.sub %26, %116 : !llvm.i64
    %119 = llvm.select %117, %118, %116 : !llvm.i1, !llvm.i64
    %120 = llvm.sdiv %119, %28 : !llvm.i64
    %121 = llvm.sub %26, %120 : !llvm.i64
    %122 = llvm.select %117, %121, %120 : !llvm.i1, !llvm.i64
    %123 = llvm.add %122, %29 : !llvm.i64
    llvm.br ^bb17(%25 : !llvm.i64)
  ^bb17(%124: !llvm.i64):  // 2 preds: ^bb16, ^bb27
    %125 = llvm.icmp "slt" %124, %123 : !llvm.i64
    llvm.cond_br %125, ^bb18, ^bb28
  ^bb18:  // pred: ^bb17
    %126 = llvm.mul %114, %28 : !llvm.i64
    %127 = llvm.mul %114, %28 : !llvm.i64
    %128 = llvm.add %127, %28 : !llvm.i64
    %129 = llvm.icmp "slt" %31, %128 : !llvm.i64
    %130 = llvm.select %129, %31, %128 : !llvm.i1, !llvm.i64
    llvm.br ^bb19(%126 : !llvm.i64)
  ^bb19(%131: !llvm.i64):  // 2 preds: ^bb18, ^bb26
    %132 = llvm.icmp "slt" %131, %130 : !llvm.i64
    llvm.cond_br %132, ^bb20, ^bb27
  ^bb20:  // pred: ^bb19
    %133 = llvm.mul %124, %28 : !llvm.i64
    %134 = llvm.mul %124, %28 : !llvm.i64
    %135 = llvm.add %134, %28 : !llvm.i64
    %136 = llvm.icmp "slt" %32, %135 : !llvm.i64
    %137 = llvm.select %136, %32, %135 : !llvm.i1, !llvm.i64
    llvm.br ^bb21(%133 : !llvm.i64)
  ^bb21(%138: !llvm.i64):  // 2 preds: ^bb20, ^bb25
    %139 = llvm.icmp "slt" %138, %137 : !llvm.i64
    llvm.cond_br %139, ^bb22, ^bb26
  ^bb22:  // pred: ^bb21
    %140 = llvm.mul %87, %28 : !llvm.i64
    %141 = llvm.mul %114, %27 : !llvm.i64
    %142 = llvm.add %140, %141 : !llvm.i64
    %143 = llvm.mul %87, %28 : !llvm.i64
    %144 = llvm.mul %114, %27 : !llvm.i64
    %145 = llvm.add %143, %144 : !llvm.i64
    %146 = llvm.add %145, %28 : !llvm.i64
    %147 = llvm.icmp "slt" %30, %146 : !llvm.i64
    %148 = llvm.select %147, %30, %146 : !llvm.i1, !llvm.i64
    llvm.br ^bb23(%142 : !llvm.i64)
  ^bb23(%149: !llvm.i64):  // 2 preds: ^bb22, ^bb24
    %150 = llvm.icmp "slt" %149, %148 : !llvm.i64
    llvm.cond_br %150, ^bb24, ^bb25
  ^bb24:  // pred: ^bb23
    %151 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %152 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %153 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %154 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %155 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %156 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %157 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %158 = llvm.extractvalue %23[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %159 = llvm.extractvalue %23[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %160 = llvm.extractvalue %23[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %161 = llvm.extractvalue %23[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %162 = llvm.extractvalue %23[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %163 = llvm.extractvalue %23[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %164 = llvm.extractvalue %23[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %165 = llvm.extractvalue %15[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %166 = llvm.extractvalue %15[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %167 = llvm.extractvalue %15[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %168 = llvm.extractvalue %15[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %169 = llvm.extractvalue %15[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %170 = llvm.extractvalue %15[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %171 = llvm.extractvalue %15[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S1(%151, %152, %153, %154, %155, %156, %157, %138, %131, %158, %159, %160, %161, %162, %163, %164, %149, %arg0, %165, %166, %167, %168, %169, %170, %171) : (!llvm.ptr<float>, !llvm.ptr<float>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<float>, !llvm.ptr<float>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.float, !llvm.ptr<float>, !llvm.ptr<float>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %172 = llvm.add %149, %29 : !llvm.i64
    llvm.br ^bb23(%172 : !llvm.i64)
  ^bb25:  // pred: ^bb23
    %173 = llvm.add %138, %29 : !llvm.i64
    llvm.br ^bb21(%173 : !llvm.i64)
  ^bb26:  // pred: ^bb21
    %174 = llvm.add %131, %29 : !llvm.i64
    llvm.br ^bb19(%174 : !llvm.i64)
  ^bb27:  // pred: ^bb19
    %175 = llvm.add %124, %29 : !llvm.i64
    llvm.br ^bb17(%175 : !llvm.i64)
  ^bb28:  // pred: ^bb17
    %176 = llvm.add %114, %29 : !llvm.i64
    llvm.br ^bb15(%176 : !llvm.i64)
  ^bb29:  // pred: ^bb15
    %177 = llvm.add %87, %29 : !llvm.i64
    llvm.br ^bb13(%177 : !llvm.i64)
  ^bb30:  // pred: ^bb13
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

