module attributes {llvm.data_layout = ""}  {
  llvm.func @kernel_gemver(%arg0: !llvm.i32, %arg1: !llvm.double, %arg2: !llvm.double, %arg3: !llvm.ptr<double>, %arg4: !llvm.ptr<double>, %arg5: !llvm.i64, %arg6: !llvm.i64, %arg7: !llvm.i64, %arg8: !llvm.i64, %arg9: !llvm.i64, %arg10: !llvm.ptr<double>, %arg11: !llvm.ptr<double>, %arg12: !llvm.i64, %arg13: !llvm.i64, %arg14: !llvm.i64, %arg15: !llvm.ptr<double>, %arg16: !llvm.ptr<double>, %arg17: !llvm.i64, %arg18: !llvm.i64, %arg19: !llvm.i64, %arg20: !llvm.ptr<double>, %arg21: !llvm.ptr<double>, %arg22: !llvm.i64, %arg23: !llvm.i64, %arg24: !llvm.i64, %arg25: !llvm.ptr<double>, %arg26: !llvm.ptr<double>, %arg27: !llvm.i64, %arg28: !llvm.i64, %arg29: !llvm.i64, %arg30: !llvm.ptr<double>, %arg31: !llvm.ptr<double>, %arg32: !llvm.i64, %arg33: !llvm.i64, %arg34: !llvm.i64, %arg35: !llvm.ptr<double>, %arg36: !llvm.ptr<double>, %arg37: !llvm.i64, %arg38: !llvm.i64, %arg39: !llvm.i64, %arg40: !llvm.ptr<double>, %arg41: !llvm.ptr<double>, %arg42: !llvm.i64, %arg43: !llvm.i64, %arg44: !llvm.i64, %arg45: !llvm.ptr<double>, %arg46: !llvm.ptr<double>, %arg47: !llvm.i64, %arg48: !llvm.i64, %arg49: !llvm.i64) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg3, %0[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.insertvalue %arg4, %1[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.insertvalue %arg5, %2[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.insertvalue %arg6, %3[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.insertvalue %arg8, %4[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.insertvalue %arg7, %5[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.insertvalue %arg9, %6[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.insertvalue %arg10, %8[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %10 = llvm.insertvalue %arg11, %9[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.insertvalue %arg12, %10[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %12 = llvm.insertvalue %arg13, %11[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %13 = llvm.insertvalue %arg14, %12[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %14 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %15 = llvm.insertvalue %arg15, %14[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %16 = llvm.insertvalue %arg16, %15[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %17 = llvm.insertvalue %arg17, %16[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %18 = llvm.insertvalue %arg18, %17[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %19 = llvm.insertvalue %arg19, %18[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %20 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %21 = llvm.insertvalue %arg20, %20[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %22 = llvm.insertvalue %arg21, %21[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %23 = llvm.insertvalue %arg22, %22[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %24 = llvm.insertvalue %arg23, %23[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %25 = llvm.insertvalue %arg24, %24[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %26 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %27 = llvm.insertvalue %arg25, %26[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %28 = llvm.insertvalue %arg26, %27[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %29 = llvm.insertvalue %arg27, %28[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %30 = llvm.insertvalue %arg28, %29[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %31 = llvm.insertvalue %arg29, %30[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %32 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %33 = llvm.insertvalue %arg30, %32[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %34 = llvm.insertvalue %arg31, %33[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %35 = llvm.insertvalue %arg32, %34[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %36 = llvm.insertvalue %arg33, %35[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %37 = llvm.insertvalue %arg34, %36[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %38 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %39 = llvm.insertvalue %arg35, %38[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %40 = llvm.insertvalue %arg36, %39[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %41 = llvm.insertvalue %arg37, %40[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %42 = llvm.insertvalue %arg38, %41[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %43 = llvm.insertvalue %arg39, %42[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %44 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %45 = llvm.insertvalue %arg40, %44[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %46 = llvm.insertvalue %arg41, %45[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %47 = llvm.insertvalue %arg42, %46[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %48 = llvm.insertvalue %arg43, %47[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %49 = llvm.insertvalue %arg44, %48[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %50 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %51 = llvm.insertvalue %arg45, %50[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %52 = llvm.insertvalue %arg46, %51[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %53 = llvm.insertvalue %arg47, %52[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %54 = llvm.insertvalue %arg48, %53[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %55 = llvm.insertvalue %arg49, %54[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %56 = llvm.mlir.constant(0 : index) : !llvm.i64
    %57 = llvm.mlir.constant(1 : index) : !llvm.i64
    %58 = llvm.sext %arg0 : !llvm.i32 to !llvm.i64
    llvm.br ^bb1(%56 : !llvm.i64)
  ^bb1(%59: !llvm.i64):  // 2 preds: ^bb0, ^bb5
    %60 = llvm.icmp "slt" %59, %58 : !llvm.i64
    llvm.cond_br %60, ^bb2, ^bb6(%56 : !llvm.i64)
  ^bb2:  // pred: ^bb1
    %61 = llvm.mlir.constant(1 : index) : !llvm.i64
    %62 = llvm.mlir.constant(1 : index) : !llvm.i64
    %63 = llvm.mlir.null : !llvm.ptr<double>
    %64 = llvm.getelementptr %63[%61] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    %65 = llvm.ptrtoint %64 : !llvm.ptr<double> to !llvm.i64
    %66 = llvm.alloca %65 x !llvm.double : (!llvm.i64) -> !llvm.ptr<double>
    %67 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %68 = llvm.insertvalue %66, %67[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %69 = llvm.insertvalue %66, %68[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %70 = llvm.mlir.constant(0 : index) : !llvm.i64
    %71 = llvm.insertvalue %70, %69[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %72 = llvm.insertvalue %61, %71[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %73 = llvm.insertvalue %62, %72[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %74 = llvm.extractvalue %73[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %75 = llvm.extractvalue %73[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %76 = llvm.extractvalue %73[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %77 = llvm.extractvalue %73[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %78 = llvm.extractvalue %73[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %79 = llvm.extractvalue %13[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %80 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %81 = llvm.extractvalue %13[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %82 = llvm.extractvalue %13[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %83 = llvm.extractvalue %13[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S0(%74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %59) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %84 = llvm.mlir.constant(1 : index) : !llvm.i64
    %85 = llvm.mlir.constant(1 : index) : !llvm.i64
    %86 = llvm.mlir.null : !llvm.ptr<double>
    %87 = llvm.getelementptr %86[%84] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    %88 = llvm.ptrtoint %87 : !llvm.ptr<double> to !llvm.i64
    %89 = llvm.alloca %88 x !llvm.double : (!llvm.i64) -> !llvm.ptr<double>
    %90 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %91 = llvm.insertvalue %89, %90[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %92 = llvm.insertvalue %89, %91[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %93 = llvm.mlir.constant(0 : index) : !llvm.i64
    %94 = llvm.insertvalue %93, %92[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %95 = llvm.insertvalue %84, %94[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %96 = llvm.insertvalue %85, %95[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %97 = llvm.extractvalue %96[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %98 = llvm.extractvalue %96[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %99 = llvm.extractvalue %96[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %100 = llvm.extractvalue %96[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %101 = llvm.extractvalue %96[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %102 = llvm.extractvalue %25[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %103 = llvm.extractvalue %25[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %104 = llvm.extractvalue %25[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %105 = llvm.extractvalue %25[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %106 = llvm.extractvalue %25[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S1(%97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %59) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.br ^bb3(%56 : !llvm.i64)
  ^bb3(%107: !llvm.i64):  // 2 preds: ^bb2, ^bb4
    %108 = llvm.icmp "slt" %107, %58 : !llvm.i64
    llvm.cond_br %108, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %109 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %110 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %111 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %112 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %113 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %114 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %115 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %116 = llvm.extractvalue %31[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %117 = llvm.extractvalue %31[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %118 = llvm.extractvalue %31[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %119 = llvm.extractvalue %31[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %120 = llvm.extractvalue %31[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %121 = llvm.extractvalue %96[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %122 = llvm.extractvalue %96[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %123 = llvm.extractvalue %96[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %124 = llvm.extractvalue %96[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %125 = llvm.extractvalue %96[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %126 = llvm.extractvalue %19[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %127 = llvm.extractvalue %19[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %128 = llvm.extractvalue %19[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %129 = llvm.extractvalue %19[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %130 = llvm.extractvalue %19[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %131 = llvm.extractvalue %73[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %132 = llvm.extractvalue %73[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %133 = llvm.extractvalue %73[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %134 = llvm.extractvalue %73[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %135 = llvm.extractvalue %73[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S2(%109, %110, %111, %112, %113, %114, %115, %59, %107, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127, %128, %129, %130, %131, %132, %133, %134, %135) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %136 = llvm.add %107, %57 : !llvm.i64
    llvm.br ^bb3(%136 : !llvm.i64)
  ^bb5:  // pred: ^bb3
    %137 = llvm.add %59, %57 : !llvm.i64
    llvm.br ^bb1(%137 : !llvm.i64)
  ^bb6(%138: !llvm.i64):  // 2 preds: ^bb1, ^bb10
    %139 = llvm.icmp "slt" %138, %58 : !llvm.i64
    llvm.cond_br %139, ^bb7, ^bb11(%56 : !llvm.i64)
  ^bb7:  // pred: ^bb6
    %140 = llvm.mlir.constant(1 : index) : !llvm.i64
    %141 = llvm.mlir.constant(1 : index) : !llvm.i64
    %142 = llvm.mlir.null : !llvm.ptr<double>
    %143 = llvm.getelementptr %142[%140] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    %144 = llvm.ptrtoint %143 : !llvm.ptr<double> to !llvm.i64
    %145 = llvm.alloca %144 x !llvm.double : (!llvm.i64) -> !llvm.ptr<double>
    %146 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %147 = llvm.insertvalue %145, %146[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %148 = llvm.insertvalue %145, %147[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %149 = llvm.mlir.constant(0 : index) : !llvm.i64
    %150 = llvm.insertvalue %149, %148[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %151 = llvm.insertvalue %140, %150[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %152 = llvm.insertvalue %141, %151[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %153 = llvm.extractvalue %152[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %154 = llvm.extractvalue %152[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %155 = llvm.extractvalue %152[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %156 = llvm.extractvalue %152[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %157 = llvm.extractvalue %152[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %158 = llvm.extractvalue %43[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %159 = llvm.extractvalue %43[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %160 = llvm.extractvalue %43[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %161 = llvm.extractvalue %43[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %162 = llvm.extractvalue %43[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S3(%153, %154, %155, %156, %157, %158, %159, %160, %161, %162, %138) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.br ^bb8(%56 : !llvm.i64)
  ^bb8(%163: !llvm.i64):  // 2 preds: ^bb7, ^bb9
    %164 = llvm.icmp "slt" %163, %58 : !llvm.i64
    llvm.cond_br %164, ^bb9, ^bb10
  ^bb9:  // pred: ^bb8
    %165 = llvm.extractvalue %43[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %166 = llvm.extractvalue %43[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %167 = llvm.extractvalue %43[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %168 = llvm.extractvalue %43[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %169 = llvm.extractvalue %43[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %170 = llvm.extractvalue %49[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %171 = llvm.extractvalue %49[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %172 = llvm.extractvalue %49[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %173 = llvm.extractvalue %49[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %174 = llvm.extractvalue %49[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %175 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %176 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %177 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %178 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %179 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %180 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %181 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %182 = llvm.extractvalue %152[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %183 = llvm.extractvalue %152[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %184 = llvm.extractvalue %152[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %185 = llvm.extractvalue %152[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %186 = llvm.extractvalue %152[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S4(%165, %166, %167, %168, %169, %138, %170, %171, %172, %173, %174, %163, %arg2, %175, %176, %177, %178, %179, %180, %181, %182, %183, %184, %185, %186) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.double, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %187 = llvm.add %163, %57 : !llvm.i64
    llvm.br ^bb8(%187 : !llvm.i64)
  ^bb10:  // pred: ^bb8
    %188 = llvm.add %138, %57 : !llvm.i64
    llvm.br ^bb6(%188 : !llvm.i64)
  ^bb11(%189: !llvm.i64):  // 2 preds: ^bb6, ^bb12
    %190 = llvm.icmp "slt" %189, %58 : !llvm.i64
    llvm.cond_br %190, ^bb12, ^bb13(%56 : !llvm.i64)
  ^bb12:  // pred: ^bb11
    %191 = llvm.extractvalue %43[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %192 = llvm.extractvalue %43[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %193 = llvm.extractvalue %43[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %194 = llvm.extractvalue %43[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %195 = llvm.extractvalue %43[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %196 = llvm.extractvalue %55[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %197 = llvm.extractvalue %55[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %198 = llvm.extractvalue %55[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %199 = llvm.extractvalue %55[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %200 = llvm.extractvalue %55[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S5(%191, %192, %193, %194, %195, %189, %196, %197, %198, %199, %200) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %201 = llvm.add %189, %57 : !llvm.i64
    llvm.br ^bb11(%201 : !llvm.i64)
  ^bb13(%202: !llvm.i64):  // 2 preds: ^bb11, ^bb17
    %203 = llvm.icmp "slt" %202, %58 : !llvm.i64
    llvm.cond_br %203, ^bb14, ^bb18
  ^bb14:  // pred: ^bb13
    %204 = llvm.mlir.constant(1 : index) : !llvm.i64
    %205 = llvm.mlir.constant(1 : index) : !llvm.i64
    %206 = llvm.mlir.null : !llvm.ptr<double>
    %207 = llvm.getelementptr %206[%204] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    %208 = llvm.ptrtoint %207 : !llvm.ptr<double> to !llvm.i64
    %209 = llvm.alloca %208 x !llvm.double : (!llvm.i64) -> !llvm.ptr<double>
    %210 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %211 = llvm.insertvalue %209, %210[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %212 = llvm.insertvalue %209, %211[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %213 = llvm.mlir.constant(0 : index) : !llvm.i64
    %214 = llvm.insertvalue %213, %212[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %215 = llvm.insertvalue %204, %214[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %216 = llvm.insertvalue %205, %215[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %217 = llvm.extractvalue %216[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %218 = llvm.extractvalue %216[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %219 = llvm.extractvalue %216[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %220 = llvm.extractvalue %216[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %221 = llvm.extractvalue %216[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %222 = llvm.extractvalue %37[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %223 = llvm.extractvalue %37[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %224 = llvm.extractvalue %37[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %225 = llvm.extractvalue %37[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %226 = llvm.extractvalue %37[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S6(%217, %218, %219, %220, %221, %222, %223, %224, %225, %226, %202) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.br ^bb15(%56 : !llvm.i64)
  ^bb15(%227: !llvm.i64):  // 2 preds: ^bb14, ^bb16
    %228 = llvm.icmp "slt" %227, %58 : !llvm.i64
    llvm.cond_br %228, ^bb16, ^bb17
  ^bb16:  // pred: ^bb15
    %229 = llvm.extractvalue %37[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %230 = llvm.extractvalue %37[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %231 = llvm.extractvalue %37[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %232 = llvm.extractvalue %37[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %233 = llvm.extractvalue %37[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %234 = llvm.extractvalue %43[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %235 = llvm.extractvalue %43[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %236 = llvm.extractvalue %43[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %237 = llvm.extractvalue %43[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %238 = llvm.extractvalue %43[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %239 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %240 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %241 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %242 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %243 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %244 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %245 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %246 = llvm.extractvalue %216[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %247 = llvm.extractvalue %216[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %248 = llvm.extractvalue %216[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %249 = llvm.extractvalue %216[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %250 = llvm.extractvalue %216[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S7(%229, %230, %231, %232, %233, %202, %234, %235, %236, %237, %238, %227, %arg1, %239, %240, %241, %242, %243, %244, %245, %246, %247, %248, %249, %250) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.double, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %251 = llvm.add %227, %57 : !llvm.i64
    llvm.br ^bb15(%251 : !llvm.i64)
  ^bb17:  // pred: ^bb15
    %252 = llvm.add %202, %57 : !llvm.i64
    llvm.br ^bb13(%252 : !llvm.i64)
  ^bb18:  // pred: ^bb13
    llvm.return
  }
  llvm.func @_mlir_ciface_kernel_gemver(%arg0: !llvm.i32, %arg1: !llvm.double, %arg2: !llvm.double, %arg3: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>>, %arg4: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>, %arg5: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>, %arg6: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>, %arg7: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>, %arg8: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>, %arg9: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>, %arg10: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>, %arg11: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>) {
    %0 = llvm.load %arg3 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.load %arg4 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>
    %9 = llvm.extractvalue %8[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %10 = llvm.extractvalue %8[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.extractvalue %8[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %12 = llvm.extractvalue %8[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %13 = llvm.extractvalue %8[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %14 = llvm.load %arg5 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>
    %15 = llvm.extractvalue %14[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %16 = llvm.extractvalue %14[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %17 = llvm.extractvalue %14[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %18 = llvm.extractvalue %14[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %19 = llvm.extractvalue %14[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %20 = llvm.load %arg6 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>
    %21 = llvm.extractvalue %20[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %22 = llvm.extractvalue %20[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %23 = llvm.extractvalue %20[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %24 = llvm.extractvalue %20[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %25 = llvm.extractvalue %20[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %26 = llvm.load %arg7 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>
    %27 = llvm.extractvalue %26[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %28 = llvm.extractvalue %26[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %29 = llvm.extractvalue %26[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %30 = llvm.extractvalue %26[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %31 = llvm.extractvalue %26[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %32 = llvm.load %arg8 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>
    %33 = llvm.extractvalue %32[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %34 = llvm.extractvalue %32[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %35 = llvm.extractvalue %32[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %36 = llvm.extractvalue %32[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %37 = llvm.extractvalue %32[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %38 = llvm.load %arg9 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>
    %39 = llvm.extractvalue %38[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %40 = llvm.extractvalue %38[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %41 = llvm.extractvalue %38[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %42 = llvm.extractvalue %38[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %43 = llvm.extractvalue %38[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %44 = llvm.load %arg10 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>
    %45 = llvm.extractvalue %44[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %46 = llvm.extractvalue %44[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %47 = llvm.extractvalue %44[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %48 = llvm.extractvalue %44[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %49 = llvm.extractvalue %44[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %50 = llvm.load %arg11 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>
    %51 = llvm.extractvalue %50[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %52 = llvm.extractvalue %50[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %53 = llvm.extractvalue %50[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %54 = llvm.extractvalue %50[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %55 = llvm.extractvalue %50[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @kernel_gemver(%arg0, %arg1, %arg2, %1, %2, %3, %4, %5, %6, %7, %9, %10, %11, %12, %13, %15, %16, %17, %18, %19, %21, %22, %23, %24, %25, %27, %28, %29, %30, %31, %33, %34, %35, %36, %37, %39, %40, %41, %42, %43, %45, %46, %47, %48, %49, %51, %52, %53, %54, %55) : (!llvm.i32, !llvm.double, !llvm.double, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.return
  }
  llvm.func @S0(%arg0: !llvm.ptr<double>, %arg1: !llvm.ptr<double>, %arg2: !llvm.i64, %arg3: !llvm.i64, %arg4: !llvm.i64, %arg5: !llvm.ptr<double>, %arg6: !llvm.ptr<double>, %arg7: !llvm.i64, %arg8: !llvm.i64, %arg9: !llvm.i64, %arg10: !llvm.i64) attributes {scop.stmt} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %5 = llvm.insertvalue %arg4, %4[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %6 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %7 = llvm.insertvalue %arg5, %6[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %8 = llvm.insertvalue %arg6, %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.insertvalue %arg7, %8[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %10 = llvm.insertvalue %arg8, %9[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.insertvalue %arg9, %10[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %12 = llvm.mlir.constant(0 : index) : !llvm.i64
    %13 = llvm.extractvalue %11[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %14 = llvm.getelementptr %13[%arg10] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    %15 = llvm.load %14 : !llvm.ptr<double>
    %16 = llvm.extractvalue %5[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %17 = llvm.getelementptr %16[%12] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    llvm.store %15, %17 : !llvm.ptr<double>
    llvm.return
  }
  llvm.func @_mlir_ciface_S0(%arg0: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>, %arg1: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>, %arg2: !llvm.i64) attributes {scop.stmt} {
    %0 = llvm.load %arg0 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %5 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %6 = llvm.load %arg1 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>
    %7 = llvm.extractvalue %6[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %8 = llvm.extractvalue %6[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.extractvalue %6[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %10 = llvm.extractvalue %6[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.extractvalue %6[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S0(%1, %2, %3, %4, %5, %7, %8, %9, %10, %11, %arg2) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.return
  }
  llvm.func @S1(%arg0: !llvm.ptr<double>, %arg1: !llvm.ptr<double>, %arg2: !llvm.i64, %arg3: !llvm.i64, %arg4: !llvm.i64, %arg5: !llvm.ptr<double>, %arg6: !llvm.ptr<double>, %arg7: !llvm.i64, %arg8: !llvm.i64, %arg9: !llvm.i64, %arg10: !llvm.i64) attributes {scop.stmt} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %5 = llvm.insertvalue %arg4, %4[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %6 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %7 = llvm.insertvalue %arg5, %6[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %8 = llvm.insertvalue %arg6, %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.insertvalue %arg7, %8[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %10 = llvm.insertvalue %arg8, %9[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.insertvalue %arg9, %10[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %12 = llvm.mlir.constant(0 : index) : !llvm.i64
    %13 = llvm.extractvalue %11[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %14 = llvm.getelementptr %13[%arg10] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    %15 = llvm.load %14 : !llvm.ptr<double>
    %16 = llvm.extractvalue %5[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %17 = llvm.getelementptr %16[%12] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    llvm.store %15, %17 : !llvm.ptr<double>
    llvm.return
  }
  llvm.func @_mlir_ciface_S1(%arg0: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>, %arg1: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>, %arg2: !llvm.i64) attributes {scop.stmt} {
    %0 = llvm.load %arg0 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %5 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %6 = llvm.load %arg1 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>
    %7 = llvm.extractvalue %6[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %8 = llvm.extractvalue %6[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.extractvalue %6[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %10 = llvm.extractvalue %6[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.extractvalue %6[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S1(%1, %2, %3, %4, %5, %7, %8, %9, %10, %11, %arg2) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.return
  }
  llvm.func @S2(%arg0: !llvm.ptr<double>, %arg1: !llvm.ptr<double>, %arg2: !llvm.i64, %arg3: !llvm.i64, %arg4: !llvm.i64, %arg5: !llvm.i64, %arg6: !llvm.i64, %arg7: !llvm.i64, %arg8: !llvm.i64, %arg9: !llvm.ptr<double>, %arg10: !llvm.ptr<double>, %arg11: !llvm.i64, %arg12: !llvm.i64, %arg13: !llvm.i64, %arg14: !llvm.ptr<double>, %arg15: !llvm.ptr<double>, %arg16: !llvm.i64, %arg17: !llvm.i64, %arg18: !llvm.i64, %arg19: !llvm.ptr<double>, %arg20: !llvm.ptr<double>, %arg21: !llvm.i64, %arg22: !llvm.i64, %arg23: !llvm.i64, %arg24: !llvm.ptr<double>, %arg25: !llvm.ptr<double>, %arg26: !llvm.i64, %arg27: !llvm.i64, %arg28: !llvm.i64) attributes {scop.stmt} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.insertvalue %arg9, %8[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %10 = llvm.insertvalue %arg10, %9[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.insertvalue %arg11, %10[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %12 = llvm.insertvalue %arg12, %11[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %13 = llvm.insertvalue %arg13, %12[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %14 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %15 = llvm.insertvalue %arg14, %14[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %16 = llvm.insertvalue %arg15, %15[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %17 = llvm.insertvalue %arg16, %16[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %18 = llvm.insertvalue %arg17, %17[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %19 = llvm.insertvalue %arg18, %18[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %20 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %21 = llvm.insertvalue %arg19, %20[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %22 = llvm.insertvalue %arg20, %21[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %23 = llvm.insertvalue %arg21, %22[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %24 = llvm.insertvalue %arg22, %23[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %25 = llvm.insertvalue %arg23, %24[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %26 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %27 = llvm.insertvalue %arg24, %26[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %28 = llvm.insertvalue %arg25, %27[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %29 = llvm.insertvalue %arg26, %28[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %30 = llvm.insertvalue %arg27, %29[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %31 = llvm.insertvalue %arg28, %30[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %32 = llvm.mlir.constant(0 : index) : !llvm.i64
    %33 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %34 = llvm.mlir.constant(2000 : index) : !llvm.i64
    %35 = llvm.mul %arg7, %34 : !llvm.i64
    %36 = llvm.add %35, %arg8 : !llvm.i64
    %37 = llvm.getelementptr %33[%36] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    %38 = llvm.load %37 : !llvm.ptr<double>
    %39 = llvm.extractvalue %31[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %40 = llvm.getelementptr %39[%32] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    %41 = llvm.load %40 : !llvm.ptr<double>
    %42 = llvm.extractvalue %25[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %43 = llvm.getelementptr %42[%arg8] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    %44 = llvm.load %43 : !llvm.ptr<double>
    %45 = llvm.fmul %41, %44 : !llvm.double
    %46 = llvm.fadd %38, %45 : !llvm.double
    %47 = llvm.extractvalue %19[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %48 = llvm.getelementptr %47[%32] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    %49 = llvm.load %48 : !llvm.ptr<double>
    %50 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %51 = llvm.getelementptr %50[%arg8] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    %52 = llvm.load %51 : !llvm.ptr<double>
    %53 = llvm.fmul %49, %52 : !llvm.double
    %54 = llvm.fadd %46, %53 : !llvm.double
    %55 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %56 = llvm.mlir.constant(2000 : index) : !llvm.i64
    %57 = llvm.mul %arg7, %56 : !llvm.i64
    %58 = llvm.add %57, %arg8 : !llvm.i64
    %59 = llvm.getelementptr %55[%58] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    llvm.store %54, %59 : !llvm.ptr<double>
    llvm.return
  }
  llvm.func @_mlir_ciface_S2(%arg0: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>>, %arg1: !llvm.i64, %arg2: !llvm.i64, %arg3: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>, %arg4: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>, %arg5: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>, %arg6: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>) attributes {scop.stmt} {
    %0 = llvm.load %arg0 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.load %arg3 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>
    %9 = llvm.extractvalue %8[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %10 = llvm.extractvalue %8[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.extractvalue %8[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %12 = llvm.extractvalue %8[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %13 = llvm.extractvalue %8[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %14 = llvm.load %arg4 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>
    %15 = llvm.extractvalue %14[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %16 = llvm.extractvalue %14[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %17 = llvm.extractvalue %14[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %18 = llvm.extractvalue %14[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %19 = llvm.extractvalue %14[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %20 = llvm.load %arg5 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>
    %21 = llvm.extractvalue %20[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %22 = llvm.extractvalue %20[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %23 = llvm.extractvalue %20[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %24 = llvm.extractvalue %20[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %25 = llvm.extractvalue %20[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %26 = llvm.load %arg6 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>
    %27 = llvm.extractvalue %26[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %28 = llvm.extractvalue %26[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %29 = llvm.extractvalue %26[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %30 = llvm.extractvalue %26[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %31 = llvm.extractvalue %26[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S2(%1, %2, %3, %4, %5, %6, %7, %arg1, %arg2, %9, %10, %11, %12, %13, %15, %16, %17, %18, %19, %21, %22, %23, %24, %25, %27, %28, %29, %30, %31) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.return
  }
  llvm.func @S3(%arg0: !llvm.ptr<double>, %arg1: !llvm.ptr<double>, %arg2: !llvm.i64, %arg3: !llvm.i64, %arg4: !llvm.i64, %arg5: !llvm.ptr<double>, %arg6: !llvm.ptr<double>, %arg7: !llvm.i64, %arg8: !llvm.i64, %arg9: !llvm.i64, %arg10: !llvm.i64) attributes {scop.stmt} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %5 = llvm.insertvalue %arg4, %4[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %6 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %7 = llvm.insertvalue %arg5, %6[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %8 = llvm.insertvalue %arg6, %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.insertvalue %arg7, %8[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %10 = llvm.insertvalue %arg8, %9[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.insertvalue %arg9, %10[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %12 = llvm.mlir.constant(0 : index) : !llvm.i64
    %13 = llvm.extractvalue %11[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %14 = llvm.getelementptr %13[%arg10] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    %15 = llvm.load %14 : !llvm.ptr<double>
    %16 = llvm.extractvalue %5[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %17 = llvm.getelementptr %16[%12] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    llvm.store %15, %17 : !llvm.ptr<double>
    llvm.return
  }
  llvm.func @_mlir_ciface_S3(%arg0: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>, %arg1: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>, %arg2: !llvm.i64) attributes {scop.stmt} {
    %0 = llvm.load %arg0 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %5 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %6 = llvm.load %arg1 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>
    %7 = llvm.extractvalue %6[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %8 = llvm.extractvalue %6[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.extractvalue %6[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %10 = llvm.extractvalue %6[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.extractvalue %6[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S3(%1, %2, %3, %4, %5, %7, %8, %9, %10, %11, %arg2) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.return
  }
  llvm.func @S4(%arg0: !llvm.ptr<double>, %arg1: !llvm.ptr<double>, %arg2: !llvm.i64, %arg3: !llvm.i64, %arg4: !llvm.i64, %arg5: !llvm.i64, %arg6: !llvm.ptr<double>, %arg7: !llvm.ptr<double>, %arg8: !llvm.i64, %arg9: !llvm.i64, %arg10: !llvm.i64, %arg11: !llvm.i64, %arg12: !llvm.double, %arg13: !llvm.ptr<double>, %arg14: !llvm.ptr<double>, %arg15: !llvm.i64, %arg16: !llvm.i64, %arg17: !llvm.i64, %arg18: !llvm.i64, %arg19: !llvm.i64, %arg20: !llvm.ptr<double>, %arg21: !llvm.ptr<double>, %arg22: !llvm.i64, %arg23: !llvm.i64, %arg24: !llvm.i64) attributes {scop.stmt} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %5 = llvm.insertvalue %arg4, %4[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %6 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %7 = llvm.insertvalue %arg6, %6[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %8 = llvm.insertvalue %arg7, %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.insertvalue %arg8, %8[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %10 = llvm.insertvalue %arg9, %9[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.insertvalue %arg10, %10[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %12 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %13 = llvm.insertvalue %arg13, %12[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %14 = llvm.insertvalue %arg14, %13[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %15 = llvm.insertvalue %arg15, %14[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %16 = llvm.insertvalue %arg16, %15[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %17 = llvm.insertvalue %arg18, %16[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %18 = llvm.insertvalue %arg17, %17[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %19 = llvm.insertvalue %arg19, %18[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %20 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %21 = llvm.insertvalue %arg20, %20[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %22 = llvm.insertvalue %arg21, %21[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %23 = llvm.insertvalue %arg22, %22[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %24 = llvm.insertvalue %arg23, %23[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %25 = llvm.insertvalue %arg24, %24[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %26 = llvm.mlir.constant(0 : index) : !llvm.i64
    %27 = llvm.extractvalue %25[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %28 = llvm.getelementptr %27[%26] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    %29 = llvm.load %28 : !llvm.ptr<double>
    %30 = llvm.extractvalue %19[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %31 = llvm.mlir.constant(2000 : index) : !llvm.i64
    %32 = llvm.mul %arg11, %31 : !llvm.i64
    %33 = llvm.add %32, %arg5 : !llvm.i64
    %34 = llvm.getelementptr %30[%33] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    %35 = llvm.load %34 : !llvm.ptr<double>
    %36 = llvm.fmul %arg12, %35 : !llvm.double
    %37 = llvm.extractvalue %11[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %38 = llvm.getelementptr %37[%arg11] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    %39 = llvm.load %38 : !llvm.ptr<double>
    %40 = llvm.fmul %36, %39 : !llvm.double
    %41 = llvm.fadd %29, %40 : !llvm.double
    %42 = llvm.extractvalue %5[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %43 = llvm.getelementptr %42[%arg5] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    llvm.store %41, %43 : !llvm.ptr<double>
    llvm.return
  }
  llvm.func @_mlir_ciface_S4(%arg0: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>, %arg1: !llvm.i64, %arg2: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>, %arg3: !llvm.i64, %arg4: !llvm.double, %arg5: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>>, %arg6: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>) attributes {scop.stmt} {
    %0 = llvm.load %arg0 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %5 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %6 = llvm.load %arg2 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>
    %7 = llvm.extractvalue %6[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %8 = llvm.extractvalue %6[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.extractvalue %6[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %10 = llvm.extractvalue %6[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.extractvalue %6[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %12 = llvm.load %arg5 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>>
    %13 = llvm.extractvalue %12[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %14 = llvm.extractvalue %12[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %15 = llvm.extractvalue %12[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %16 = llvm.extractvalue %12[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %17 = llvm.extractvalue %12[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %18 = llvm.extractvalue %12[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %19 = llvm.extractvalue %12[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %20 = llvm.load %arg6 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>
    %21 = llvm.extractvalue %20[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %22 = llvm.extractvalue %20[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %23 = llvm.extractvalue %20[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %24 = llvm.extractvalue %20[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %25 = llvm.extractvalue %20[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S4(%1, %2, %3, %4, %5, %arg1, %7, %8, %9, %10, %11, %arg3, %arg4, %13, %14, %15, %16, %17, %18, %19, %21, %22, %23, %24, %25) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.double, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.return
  }
  llvm.func @S5(%arg0: !llvm.ptr<double>, %arg1: !llvm.ptr<double>, %arg2: !llvm.i64, %arg3: !llvm.i64, %arg4: !llvm.i64, %arg5: !llvm.i64, %arg6: !llvm.ptr<double>, %arg7: !llvm.ptr<double>, %arg8: !llvm.i64, %arg9: !llvm.i64, %arg10: !llvm.i64) attributes {scop.stmt} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %5 = llvm.insertvalue %arg4, %4[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %6 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %7 = llvm.insertvalue %arg6, %6[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %8 = llvm.insertvalue %arg7, %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.insertvalue %arg8, %8[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %10 = llvm.insertvalue %arg9, %9[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.insertvalue %arg10, %10[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %12 = llvm.extractvalue %5[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %13 = llvm.getelementptr %12[%arg5] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    %14 = llvm.load %13 : !llvm.ptr<double>
    %15 = llvm.extractvalue %11[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %16 = llvm.getelementptr %15[%arg5] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    %17 = llvm.load %16 : !llvm.ptr<double>
    %18 = llvm.fadd %14, %17 : !llvm.double
    %19 = llvm.extractvalue %5[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %20 = llvm.getelementptr %19[%arg5] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    llvm.store %18, %20 : !llvm.ptr<double>
    llvm.return
  }
  llvm.func @_mlir_ciface_S5(%arg0: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>, %arg1: !llvm.i64, %arg2: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>) attributes {scop.stmt} {
    %0 = llvm.load %arg0 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %5 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %6 = llvm.load %arg2 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>
    %7 = llvm.extractvalue %6[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %8 = llvm.extractvalue %6[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.extractvalue %6[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %10 = llvm.extractvalue %6[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.extractvalue %6[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S5(%1, %2, %3, %4, %5, %arg1, %7, %8, %9, %10, %11) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.return
  }
  llvm.func @S6(%arg0: !llvm.ptr<double>, %arg1: !llvm.ptr<double>, %arg2: !llvm.i64, %arg3: !llvm.i64, %arg4: !llvm.i64, %arg5: !llvm.ptr<double>, %arg6: !llvm.ptr<double>, %arg7: !llvm.i64, %arg8: !llvm.i64, %arg9: !llvm.i64, %arg10: !llvm.i64) attributes {scop.stmt} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %5 = llvm.insertvalue %arg4, %4[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %6 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %7 = llvm.insertvalue %arg5, %6[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %8 = llvm.insertvalue %arg6, %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.insertvalue %arg7, %8[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %10 = llvm.insertvalue %arg8, %9[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.insertvalue %arg9, %10[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %12 = llvm.mlir.constant(0 : index) : !llvm.i64
    %13 = llvm.extractvalue %11[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %14 = llvm.getelementptr %13[%arg10] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    %15 = llvm.load %14 : !llvm.ptr<double>
    %16 = llvm.extractvalue %5[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %17 = llvm.getelementptr %16[%12] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    llvm.store %15, %17 : !llvm.ptr<double>
    llvm.return
  }
  llvm.func @_mlir_ciface_S6(%arg0: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>, %arg1: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>, %arg2: !llvm.i64) attributes {scop.stmt} {
    %0 = llvm.load %arg0 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %5 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %6 = llvm.load %arg1 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>
    %7 = llvm.extractvalue %6[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %8 = llvm.extractvalue %6[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.extractvalue %6[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %10 = llvm.extractvalue %6[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.extractvalue %6[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S6(%1, %2, %3, %4, %5, %7, %8, %9, %10, %11, %arg2) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.return
  }
  llvm.func @S7(%arg0: !llvm.ptr<double>, %arg1: !llvm.ptr<double>, %arg2: !llvm.i64, %arg3: !llvm.i64, %arg4: !llvm.i64, %arg5: !llvm.i64, %arg6: !llvm.ptr<double>, %arg7: !llvm.ptr<double>, %arg8: !llvm.i64, %arg9: !llvm.i64, %arg10: !llvm.i64, %arg11: !llvm.i64, %arg12: !llvm.double, %arg13: !llvm.ptr<double>, %arg14: !llvm.ptr<double>, %arg15: !llvm.i64, %arg16: !llvm.i64, %arg17: !llvm.i64, %arg18: !llvm.i64, %arg19: !llvm.i64, %arg20: !llvm.ptr<double>, %arg21: !llvm.ptr<double>, %arg22: !llvm.i64, %arg23: !llvm.i64, %arg24: !llvm.i64) attributes {scop.stmt} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %5 = llvm.insertvalue %arg4, %4[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %6 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %7 = llvm.insertvalue %arg6, %6[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %8 = llvm.insertvalue %arg7, %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.insertvalue %arg8, %8[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %10 = llvm.insertvalue %arg9, %9[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.insertvalue %arg10, %10[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %12 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %13 = llvm.insertvalue %arg13, %12[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %14 = llvm.insertvalue %arg14, %13[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %15 = llvm.insertvalue %arg15, %14[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %16 = llvm.insertvalue %arg16, %15[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %17 = llvm.insertvalue %arg18, %16[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %18 = llvm.insertvalue %arg17, %17[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %19 = llvm.insertvalue %arg19, %18[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %20 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %21 = llvm.insertvalue %arg20, %20[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %22 = llvm.insertvalue %arg21, %21[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %23 = llvm.insertvalue %arg22, %22[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %24 = llvm.insertvalue %arg23, %23[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %25 = llvm.insertvalue %arg24, %24[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %26 = llvm.mlir.constant(0 : index) : !llvm.i64
    %27 = llvm.extractvalue %25[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %28 = llvm.getelementptr %27[%26] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    %29 = llvm.load %28 : !llvm.ptr<double>
    %30 = llvm.extractvalue %19[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %31 = llvm.mlir.constant(2000 : index) : !llvm.i64
    %32 = llvm.mul %arg5, %31 : !llvm.i64
    %33 = llvm.add %32, %arg11 : !llvm.i64
    %34 = llvm.getelementptr %30[%33] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    %35 = llvm.load %34 : !llvm.ptr<double>
    %36 = llvm.fmul %arg12, %35 : !llvm.double
    %37 = llvm.extractvalue %11[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %38 = llvm.getelementptr %37[%arg11] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    %39 = llvm.load %38 : !llvm.ptr<double>
    %40 = llvm.fmul %36, %39 : !llvm.double
    %41 = llvm.fadd %29, %40 : !llvm.double
    %42 = llvm.extractvalue %5[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %43 = llvm.getelementptr %42[%arg5] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    llvm.store %41, %43 : !llvm.ptr<double>
    llvm.return
  }
  llvm.func @_mlir_ciface_S7(%arg0: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>, %arg1: !llvm.i64, %arg2: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>, %arg3: !llvm.i64, %arg4: !llvm.double, %arg5: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>>, %arg6: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>) attributes {scop.stmt} {
    %0 = llvm.load %arg0 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %5 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %6 = llvm.load %arg2 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>
    %7 = llvm.extractvalue %6[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %8 = llvm.extractvalue %6[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.extractvalue %6[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %10 = llvm.extractvalue %6[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.extractvalue %6[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %12 = llvm.load %arg5 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>>
    %13 = llvm.extractvalue %12[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %14 = llvm.extractvalue %12[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %15 = llvm.extractvalue %12[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %16 = llvm.extractvalue %12[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %17 = llvm.extractvalue %12[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %18 = llvm.extractvalue %12[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %19 = llvm.extractvalue %12[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %20 = llvm.load %arg6 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>
    %21 = llvm.extractvalue %20[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %22 = llvm.extractvalue %20[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %23 = llvm.extractvalue %20[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %24 = llvm.extractvalue %20[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %25 = llvm.extractvalue %20[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S7(%1, %2, %3, %4, %5, %arg1, %7, %8, %9, %10, %11, %arg3, %arg4, %13, %14, %15, %16, %17, %18, %19, %21, %22, %23, %24, %25) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.double, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.return
  }
  llvm.func @kernel_gemver_new(%arg0: !llvm.i32, %arg1: !llvm.double, %arg2: !llvm.double, %arg3: !llvm.ptr<double>, %arg4: !llvm.ptr<double>, %arg5: !llvm.i64, %arg6: !llvm.i64, %arg7: !llvm.i64, %arg8: !llvm.i64, %arg9: !llvm.i64, %arg10: !llvm.ptr<double>, %arg11: !llvm.ptr<double>, %arg12: !llvm.i64, %arg13: !llvm.i64, %arg14: !llvm.i64, %arg15: !llvm.ptr<double>, %arg16: !llvm.ptr<double>, %arg17: !llvm.i64, %arg18: !llvm.i64, %arg19: !llvm.i64, %arg20: !llvm.ptr<double>, %arg21: !llvm.ptr<double>, %arg22: !llvm.i64, %arg23: !llvm.i64, %arg24: !llvm.i64, %arg25: !llvm.ptr<double>, %arg26: !llvm.ptr<double>, %arg27: !llvm.i64, %arg28: !llvm.i64, %arg29: !llvm.i64, %arg30: !llvm.ptr<double>, %arg31: !llvm.ptr<double>, %arg32: !llvm.i64, %arg33: !llvm.i64, %arg34: !llvm.i64, %arg35: !llvm.ptr<double>, %arg36: !llvm.ptr<double>, %arg37: !llvm.i64, %arg38: !llvm.i64, %arg39: !llvm.i64, %arg40: !llvm.ptr<double>, %arg41: !llvm.ptr<double>, %arg42: !llvm.i64, %arg43: !llvm.i64, %arg44: !llvm.i64, %arg45: !llvm.ptr<double>, %arg46: !llvm.ptr<double>, %arg47: !llvm.i64, %arg48: !llvm.i64, %arg49: !llvm.i64) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg3, %0[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.insertvalue %arg4, %1[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.insertvalue %arg5, %2[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.insertvalue %arg6, %3[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.insertvalue %arg8, %4[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.insertvalue %arg7, %5[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.insertvalue %arg9, %6[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.insertvalue %arg10, %8[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %10 = llvm.insertvalue %arg11, %9[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.insertvalue %arg12, %10[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %12 = llvm.insertvalue %arg13, %11[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %13 = llvm.insertvalue %arg14, %12[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %14 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %15 = llvm.insertvalue %arg15, %14[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %16 = llvm.insertvalue %arg16, %15[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %17 = llvm.insertvalue %arg17, %16[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %18 = llvm.insertvalue %arg18, %17[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %19 = llvm.insertvalue %arg19, %18[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %20 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %21 = llvm.insertvalue %arg20, %20[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %22 = llvm.insertvalue %arg21, %21[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %23 = llvm.insertvalue %arg22, %22[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %24 = llvm.insertvalue %arg23, %23[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %25 = llvm.insertvalue %arg24, %24[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %26 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %27 = llvm.insertvalue %arg25, %26[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %28 = llvm.insertvalue %arg26, %27[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %29 = llvm.insertvalue %arg27, %28[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %30 = llvm.insertvalue %arg28, %29[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %31 = llvm.insertvalue %arg29, %30[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %32 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %33 = llvm.insertvalue %arg30, %32[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %34 = llvm.insertvalue %arg31, %33[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %35 = llvm.insertvalue %arg32, %34[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %36 = llvm.insertvalue %arg33, %35[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %37 = llvm.insertvalue %arg34, %36[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %38 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %39 = llvm.insertvalue %arg35, %38[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %40 = llvm.insertvalue %arg36, %39[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %41 = llvm.insertvalue %arg37, %40[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %42 = llvm.insertvalue %arg38, %41[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %43 = llvm.insertvalue %arg39, %42[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %44 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %45 = llvm.insertvalue %arg40, %44[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %46 = llvm.insertvalue %arg41, %45[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %47 = llvm.insertvalue %arg42, %46[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %48 = llvm.insertvalue %arg43, %47[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %49 = llvm.insertvalue %arg44, %48[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %50 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %51 = llvm.insertvalue %arg45, %50[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %52 = llvm.insertvalue %arg46, %51[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %53 = llvm.insertvalue %arg47, %52[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %54 = llvm.insertvalue %arg48, %53[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %55 = llvm.insertvalue %arg49, %54[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %56 = llvm.mlir.constant(-6 : index) : !llvm.i64
    %57 = llvm.mlir.constant(5 : index) : !llvm.i64
    %58 = llvm.mlir.constant(-7 : index) : !llvm.i64
    %59 = llvm.mlir.constant(6 : index) : !llvm.i64
    %60 = llvm.mlir.constant(7 : index) : !llvm.i64
    %61 = llvm.mlir.constant(-1 : index) : !llvm.i64
    %62 = llvm.mlir.constant(32 : index) : !llvm.i64
    %63 = llvm.mlir.constant(0 : index) : !llvm.i64
    %64 = llvm.mlir.constant(1 : index) : !llvm.i64
    %65 = llvm.mlir.constant(1 : index) : !llvm.i64
    %66 = llvm.mlir.constant(1 : index) : !llvm.i64
    %67 = llvm.mlir.null : !llvm.ptr<double>
    %68 = llvm.getelementptr %67[%65] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    %69 = llvm.ptrtoint %68 : !llvm.ptr<double> to !llvm.i64
    %70 = llvm.alloca %69 x !llvm.double : (!llvm.i64) -> !llvm.ptr<double>
    %71 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %72 = llvm.insertvalue %70, %71[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %73 = llvm.insertvalue %70, %72[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %74 = llvm.mlir.constant(0 : index) : !llvm.i64
    %75 = llvm.insertvalue %74, %73[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %76 = llvm.insertvalue %65, %75[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %77 = llvm.insertvalue %66, %76[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %78 = llvm.mlir.constant(1 : index) : !llvm.i64
    %79 = llvm.mlir.constant(1 : index) : !llvm.i64
    %80 = llvm.mlir.null : !llvm.ptr<double>
    %81 = llvm.getelementptr %80[%78] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    %82 = llvm.ptrtoint %81 : !llvm.ptr<double> to !llvm.i64
    %83 = llvm.alloca %82 x !llvm.double : (!llvm.i64) -> !llvm.ptr<double>
    %84 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %85 = llvm.insertvalue %83, %84[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %86 = llvm.insertvalue %83, %85[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %87 = llvm.mlir.constant(0 : index) : !llvm.i64
    %88 = llvm.insertvalue %87, %86[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %89 = llvm.insertvalue %78, %88[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %90 = llvm.insertvalue %79, %89[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %91 = llvm.mlir.constant(1 : index) : !llvm.i64
    %92 = llvm.mlir.constant(1 : index) : !llvm.i64
    %93 = llvm.mlir.null : !llvm.ptr<double>
    %94 = llvm.getelementptr %93[%91] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    %95 = llvm.ptrtoint %94 : !llvm.ptr<double> to !llvm.i64
    %96 = llvm.alloca %95 x !llvm.double : (!llvm.i64) -> !llvm.ptr<double>
    %97 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %98 = llvm.insertvalue %96, %97[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %99 = llvm.insertvalue %96, %98[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %100 = llvm.mlir.constant(0 : index) : !llvm.i64
    %101 = llvm.insertvalue %100, %99[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %102 = llvm.insertvalue %91, %101[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %103 = llvm.insertvalue %92, %102[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %104 = llvm.mlir.constant(1 : index) : !llvm.i64
    %105 = llvm.mlir.constant(1 : index) : !llvm.i64
    %106 = llvm.mlir.null : !llvm.ptr<double>
    %107 = llvm.getelementptr %106[%104] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    %108 = llvm.ptrtoint %107 : !llvm.ptr<double> to !llvm.i64
    %109 = llvm.alloca %108 x !llvm.double : (!llvm.i64) -> !llvm.ptr<double>
    %110 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %111 = llvm.insertvalue %109, %110[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %112 = llvm.insertvalue %109, %111[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %113 = llvm.mlir.constant(0 : index) : !llvm.i64
    %114 = llvm.insertvalue %113, %112[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %115 = llvm.insertvalue %104, %114[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %116 = llvm.insertvalue %105, %115[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %117 = llvm.sext %arg0 : !llvm.i32 to !llvm.i64
    llvm.br ^bb1(%63 : !llvm.i64)
  ^bb1(%118: !llvm.i64):  // 2 preds: ^bb0, ^bb22
    %119 = llvm.icmp "slt" %118, %117 : !llvm.i64
    llvm.cond_br %119, ^bb2, ^bb23(%63 : !llvm.i64)
  ^bb2:  // pred: ^bb1
    %120 = llvm.icmp "slt" %57, %117 : !llvm.i64
    %121 = llvm.select %120, %57, %117 : !llvm.i1, !llvm.i64
    llvm.br ^bb3(%63 : !llvm.i64)
  ^bb3(%122: !llvm.i64):  // 2 preds: ^bb2, ^bb4
    %123 = llvm.icmp "slt" %122, %121 : !llvm.i64
    llvm.cond_br %123, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %124 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %125 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %126 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %127 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %128 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %129 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %130 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %131 = llvm.extractvalue %31[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %132 = llvm.extractvalue %31[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %133 = llvm.extractvalue %31[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %134 = llvm.extractvalue %31[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %135 = llvm.extractvalue %31[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %136 = llvm.extractvalue %77[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %137 = llvm.extractvalue %77[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %138 = llvm.extractvalue %77[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %139 = llvm.extractvalue %77[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %140 = llvm.extractvalue %77[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %141 = llvm.extractvalue %19[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %142 = llvm.extractvalue %19[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %143 = llvm.extractvalue %19[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %144 = llvm.extractvalue %19[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %145 = llvm.extractvalue %19[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %146 = llvm.extractvalue %90[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %147 = llvm.extractvalue %90[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %148 = llvm.extractvalue %90[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %149 = llvm.extractvalue %90[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %150 = llvm.extractvalue %90[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S2(%124, %125, %126, %127, %128, %129, %130, %118, %122, %131, %132, %133, %134, %135, %136, %137, %138, %139, %140, %141, %142, %143, %144, %145, %146, %147, %148, %149, %150) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %151 = llvm.add %122, %64 : !llvm.i64
    llvm.br ^bb3(%151 : !llvm.i64)
  ^bb5:  // pred: ^bb3
    %152 = llvm.add %117, %56 : !llvm.i64
    %153 = llvm.icmp "sge" %152, %63 : !llvm.i64
    llvm.cond_br %153, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %154 = llvm.extractvalue %77[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %155 = llvm.extractvalue %77[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %156 = llvm.extractvalue %77[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %157 = llvm.extractvalue %77[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %158 = llvm.extractvalue %77[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %159 = llvm.extractvalue %25[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %160 = llvm.extractvalue %25[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %161 = llvm.extractvalue %25[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %162 = llvm.extractvalue %25[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %163 = llvm.extractvalue %25[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S1(%154, %155, %156, %157, %158, %159, %160, %161, %162, %163, %118) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %164 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %165 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %166 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %167 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %168 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %169 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %170 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %171 = llvm.extractvalue %31[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %172 = llvm.extractvalue %31[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %173 = llvm.extractvalue %31[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %174 = llvm.extractvalue %31[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %175 = llvm.extractvalue %31[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %176 = llvm.extractvalue %77[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %177 = llvm.extractvalue %77[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %178 = llvm.extractvalue %77[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %179 = llvm.extractvalue %77[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %180 = llvm.extractvalue %77[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %181 = llvm.extractvalue %19[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %182 = llvm.extractvalue %19[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %183 = llvm.extractvalue %19[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %184 = llvm.extractvalue %19[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %185 = llvm.extractvalue %19[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %186 = llvm.extractvalue %90[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %187 = llvm.extractvalue %90[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %188 = llvm.extractvalue %90[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %189 = llvm.extractvalue %90[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %190 = llvm.extractvalue %90[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S2(%164, %165, %166, %167, %168, %169, %170, %118, %57, %171, %172, %173, %174, %175, %176, %177, %178, %179, %180, %181, %182, %183, %184, %185, %186, %187, %188, %189, %190) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.br ^bb7
  ^bb7:  // 2 preds: ^bb5, ^bb6
    %191 = llvm.mul %117, %61 : !llvm.i64
    %192 = llvm.add %191, %57 : !llvm.i64
    %193 = llvm.icmp "sge" %192, %63 : !llvm.i64
    llvm.cond_br %193, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %194 = llvm.extractvalue %77[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %195 = llvm.extractvalue %77[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %196 = llvm.extractvalue %77[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %197 = llvm.extractvalue %77[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %198 = llvm.extractvalue %77[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %199 = llvm.extractvalue %25[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %200 = llvm.extractvalue %25[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %201 = llvm.extractvalue %25[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %202 = llvm.extractvalue %25[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %203 = llvm.extractvalue %25[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S1(%194, %195, %196, %197, %198, %199, %200, %201, %202, %203, %118) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.br ^bb9
  ^bb9:  // 2 preds: ^bb7, ^bb8
    %204 = llvm.add %117, %58 : !llvm.i64
    %205 = llvm.icmp "sge" %204, %63 : !llvm.i64
    llvm.cond_br %205, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    %206 = llvm.extractvalue %90[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %207 = llvm.extractvalue %90[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %208 = llvm.extractvalue %90[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %209 = llvm.extractvalue %90[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %210 = llvm.extractvalue %90[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %211 = llvm.extractvalue %13[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %212 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %213 = llvm.extractvalue %13[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %214 = llvm.extractvalue %13[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %215 = llvm.extractvalue %13[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S0(%206, %207, %208, %209, %210, %211, %212, %213, %214, %215, %118) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %216 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %217 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %218 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %219 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %220 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %221 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %222 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %223 = llvm.extractvalue %31[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %224 = llvm.extractvalue %31[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %225 = llvm.extractvalue %31[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %226 = llvm.extractvalue %31[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %227 = llvm.extractvalue %31[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %228 = llvm.extractvalue %77[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %229 = llvm.extractvalue %77[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %230 = llvm.extractvalue %77[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %231 = llvm.extractvalue %77[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %232 = llvm.extractvalue %77[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %233 = llvm.extractvalue %19[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %234 = llvm.extractvalue %19[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %235 = llvm.extractvalue %19[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %236 = llvm.extractvalue %19[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %237 = llvm.extractvalue %19[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %238 = llvm.extractvalue %90[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %239 = llvm.extractvalue %90[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %240 = llvm.extractvalue %90[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %241 = llvm.extractvalue %90[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %242 = llvm.extractvalue %90[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S2(%216, %217, %218, %219, %220, %221, %222, %118, %59, %223, %224, %225, %226, %227, %228, %229, %230, %231, %232, %233, %234, %235, %236, %237, %238, %239, %240, %241, %242) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.br ^bb11
  ^bb11:  // 2 preds: ^bb9, ^bb10
    %243 = llvm.mul %117, %61 : !llvm.i64
    %244 = llvm.add %243, %59 : !llvm.i64
    %245 = llvm.icmp "sge" %244, %63 : !llvm.i64
    llvm.cond_br %245, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    %246 = llvm.extractvalue %90[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %247 = llvm.extractvalue %90[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %248 = llvm.extractvalue %90[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %249 = llvm.extractvalue %90[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %250 = llvm.extractvalue %90[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %251 = llvm.extractvalue %13[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %252 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %253 = llvm.extractvalue %13[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %254 = llvm.extractvalue %13[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %255 = llvm.extractvalue %13[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S0(%246, %247, %248, %249, %250, %251, %252, %253, %254, %255, %118) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.br ^bb13
  ^bb13:  // 2 preds: ^bb11, ^bb12
    %256 = llvm.icmp "slt" %62, %117 : !llvm.i64
    %257 = llvm.select %256, %62, %117 : !llvm.i1, !llvm.i64
    llvm.br ^bb14(%60 : !llvm.i64)
  ^bb14(%258: !llvm.i64):  // 2 preds: ^bb13, ^bb15
    %259 = llvm.icmp "slt" %258, %257 : !llvm.i64
    llvm.cond_br %259, ^bb15, ^bb16
  ^bb15:  // pred: ^bb14
    %260 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %261 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %262 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %263 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %264 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %265 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %266 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %267 = llvm.extractvalue %31[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %268 = llvm.extractvalue %31[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %269 = llvm.extractvalue %31[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %270 = llvm.extractvalue %31[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %271 = llvm.extractvalue %31[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %272 = llvm.extractvalue %77[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %273 = llvm.extractvalue %77[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %274 = llvm.extractvalue %77[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %275 = llvm.extractvalue %77[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %276 = llvm.extractvalue %77[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %277 = llvm.extractvalue %19[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %278 = llvm.extractvalue %19[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %279 = llvm.extractvalue %19[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %280 = llvm.extractvalue %19[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %281 = llvm.extractvalue %19[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %282 = llvm.extractvalue %90[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %283 = llvm.extractvalue %90[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %284 = llvm.extractvalue %90[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %285 = llvm.extractvalue %90[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %286 = llvm.extractvalue %90[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S2(%260, %261, %262, %263, %264, %265, %266, %118, %258, %267, %268, %269, %270, %271, %272, %273, %274, %275, %276, %277, %278, %279, %280, %281, %282, %283, %284, %285, %286) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %287 = llvm.add %258, %64 : !llvm.i64
    llvm.br ^bb14(%287 : !llvm.i64)
  ^bb16:  // pred: ^bb14
    %288 = llvm.add %117, %61 : !llvm.i64
    %289 = llvm.icmp "slt" %288, %63 : !llvm.i64
    %290 = llvm.sub %61, %288 : !llvm.i64
    %291 = llvm.select %289, %290, %288 : !llvm.i1, !llvm.i64
    %292 = llvm.sdiv %291, %62 : !llvm.i64
    %293 = llvm.sub %61, %292 : !llvm.i64
    %294 = llvm.select %289, %293, %292 : !llvm.i1, !llvm.i64
    %295 = llvm.add %294, %64 : !llvm.i64
    llvm.br ^bb17(%64 : !llvm.i64)
  ^bb17(%296: !llvm.i64):  // 2 preds: ^bb16, ^bb21
    %297 = llvm.icmp "slt" %296, %295 : !llvm.i64
    llvm.cond_br %297, ^bb18, ^bb22
  ^bb18:  // pred: ^bb17
    %298 = llvm.mul %296, %62 : !llvm.i64
    %299 = llvm.mul %296, %62 : !llvm.i64
    %300 = llvm.add %299, %62 : !llvm.i64
    %301 = llvm.icmp "slt" %117, %300 : !llvm.i64
    %302 = llvm.select %301, %117, %300 : !llvm.i1, !llvm.i64
    llvm.br ^bb19(%298 : !llvm.i64)
  ^bb19(%303: !llvm.i64):  // 2 preds: ^bb18, ^bb20
    %304 = llvm.icmp "slt" %303, %302 : !llvm.i64
    llvm.cond_br %304, ^bb20, ^bb21
  ^bb20:  // pred: ^bb19
    %305 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %306 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %307 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %308 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %309 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %310 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %311 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %312 = llvm.extractvalue %31[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %313 = llvm.extractvalue %31[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %314 = llvm.extractvalue %31[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %315 = llvm.extractvalue %31[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %316 = llvm.extractvalue %31[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %317 = llvm.extractvalue %77[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %318 = llvm.extractvalue %77[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %319 = llvm.extractvalue %77[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %320 = llvm.extractvalue %77[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %321 = llvm.extractvalue %77[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %322 = llvm.extractvalue %19[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %323 = llvm.extractvalue %19[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %324 = llvm.extractvalue %19[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %325 = llvm.extractvalue %19[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %326 = llvm.extractvalue %19[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %327 = llvm.extractvalue %90[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %328 = llvm.extractvalue %90[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %329 = llvm.extractvalue %90[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %330 = llvm.extractvalue %90[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %331 = llvm.extractvalue %90[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S2(%305, %306, %307, %308, %309, %310, %311, %118, %303, %312, %313, %314, %315, %316, %317, %318, %319, %320, %321, %322, %323, %324, %325, %326, %327, %328, %329, %330, %331) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %332 = llvm.add %303, %64 : !llvm.i64
    llvm.br ^bb19(%332 : !llvm.i64)
  ^bb21:  // pred: ^bb19
    %333 = llvm.add %296, %64 : !llvm.i64
    llvm.br ^bb17(%333 : !llvm.i64)
  ^bb22:  // pred: ^bb17
    %334 = llvm.add %118, %64 : !llvm.i64
    llvm.br ^bb1(%334 : !llvm.i64)
  ^bb23(%335: !llvm.i64):  // 2 preds: ^bb1, ^bb27
    %336 = llvm.icmp "slt" %335, %117 : !llvm.i64
    llvm.cond_br %336, ^bb24, ^bb28
  ^bb24:  // pred: ^bb23
    %337 = llvm.extractvalue %103[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %338 = llvm.extractvalue %103[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %339 = llvm.extractvalue %103[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %340 = llvm.extractvalue %103[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %341 = llvm.extractvalue %103[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %342 = llvm.extractvalue %43[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %343 = llvm.extractvalue %43[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %344 = llvm.extractvalue %43[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %345 = llvm.extractvalue %43[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %346 = llvm.extractvalue %43[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S3(%337, %338, %339, %340, %341, %342, %343, %344, %345, %346, %335) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.br ^bb25(%63 : !llvm.i64)
  ^bb25(%347: !llvm.i64):  // 2 preds: ^bb24, ^bb26
    %348 = llvm.icmp "slt" %347, %117 : !llvm.i64
    llvm.cond_br %348, ^bb26, ^bb27
  ^bb26:  // pred: ^bb25
    %349 = llvm.extractvalue %43[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %350 = llvm.extractvalue %43[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %351 = llvm.extractvalue %43[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %352 = llvm.extractvalue %43[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %353 = llvm.extractvalue %43[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %354 = llvm.extractvalue %49[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %355 = llvm.extractvalue %49[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %356 = llvm.extractvalue %49[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %357 = llvm.extractvalue %49[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %358 = llvm.extractvalue %49[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %359 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %360 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %361 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %362 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %363 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %364 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %365 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %366 = llvm.extractvalue %103[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %367 = llvm.extractvalue %103[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %368 = llvm.extractvalue %103[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %369 = llvm.extractvalue %103[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %370 = llvm.extractvalue %103[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S4(%349, %350, %351, %352, %353, %335, %354, %355, %356, %357, %358, %347, %arg2, %359, %360, %361, %362, %363, %364, %365, %366, %367, %368, %369, %370) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.double, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %371 = llvm.add %347, %64 : !llvm.i64
    llvm.br ^bb25(%371 : !llvm.i64)
  ^bb27:  // pred: ^bb25
    %372 = llvm.add %335, %64 : !llvm.i64
    llvm.br ^bb23(%372 : !llvm.i64)
  ^bb28:  // pred: ^bb23
    %373 = llvm.add %117, %61 : !llvm.i64
    %374 = llvm.icmp "slt" %373, %63 : !llvm.i64
    %375 = llvm.sub %61, %373 : !llvm.i64
    %376 = llvm.select %374, %375, %373 : !llvm.i1, !llvm.i64
    %377 = llvm.sdiv %376, %62 : !llvm.i64
    %378 = llvm.sub %61, %377 : !llvm.i64
    %379 = llvm.select %374, %378, %377 : !llvm.i1, !llvm.i64
    %380 = llvm.add %379, %64 : !llvm.i64
    llvm.br ^bb29(%63 : !llvm.i64)
  ^bb29(%381: !llvm.i64):  // 2 preds: ^bb28, ^bb33
    %382 = llvm.icmp "slt" %381, %380 : !llvm.i64
    llvm.cond_br %382, ^bb30, ^bb34(%63 : !llvm.i64)
  ^bb30:  // pred: ^bb29
    %383 = llvm.mul %381, %62 : !llvm.i64
    %384 = llvm.mul %381, %62 : !llvm.i64
    %385 = llvm.add %384, %62 : !llvm.i64
    %386 = llvm.icmp "slt" %117, %385 : !llvm.i64
    %387 = llvm.select %386, %117, %385 : !llvm.i1, !llvm.i64
    llvm.br ^bb31(%383 : !llvm.i64)
  ^bb31(%388: !llvm.i64):  // 2 preds: ^bb30, ^bb32
    %389 = llvm.icmp "slt" %388, %387 : !llvm.i64
    llvm.cond_br %389, ^bb32, ^bb33
  ^bb32:  // pred: ^bb31
    %390 = llvm.extractvalue %43[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %391 = llvm.extractvalue %43[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %392 = llvm.extractvalue %43[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %393 = llvm.extractvalue %43[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %394 = llvm.extractvalue %43[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %395 = llvm.extractvalue %55[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %396 = llvm.extractvalue %55[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %397 = llvm.extractvalue %55[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %398 = llvm.extractvalue %55[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %399 = llvm.extractvalue %55[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S5(%390, %391, %392, %393, %394, %388, %395, %396, %397, %398, %399) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %400 = llvm.add %388, %64 : !llvm.i64
    llvm.br ^bb31(%400 : !llvm.i64)
  ^bb33:  // pred: ^bb31
    %401 = llvm.add %381, %64 : !llvm.i64
    llvm.br ^bb29(%401 : !llvm.i64)
  ^bb34(%402: !llvm.i64):  // 2 preds: ^bb29, ^bb38
    %403 = llvm.icmp "slt" %402, %117 : !llvm.i64
    llvm.cond_br %403, ^bb35, ^bb39
  ^bb35:  // pred: ^bb34
    %404 = llvm.extractvalue %116[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %405 = llvm.extractvalue %116[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %406 = llvm.extractvalue %116[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %407 = llvm.extractvalue %116[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %408 = llvm.extractvalue %116[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %409 = llvm.extractvalue %37[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %410 = llvm.extractvalue %37[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %411 = llvm.extractvalue %37[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %412 = llvm.extractvalue %37[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %413 = llvm.extractvalue %37[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S6(%404, %405, %406, %407, %408, %409, %410, %411, %412, %413, %402) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.br ^bb36(%63 : !llvm.i64)
  ^bb36(%414: !llvm.i64):  // 2 preds: ^bb35, ^bb37
    %415 = llvm.icmp "slt" %414, %117 : !llvm.i64
    llvm.cond_br %415, ^bb37, ^bb38
  ^bb37:  // pred: ^bb36
    %416 = llvm.extractvalue %37[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %417 = llvm.extractvalue %37[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %418 = llvm.extractvalue %37[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %419 = llvm.extractvalue %37[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %420 = llvm.extractvalue %37[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %421 = llvm.extractvalue %43[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %422 = llvm.extractvalue %43[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %423 = llvm.extractvalue %43[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %424 = llvm.extractvalue %43[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %425 = llvm.extractvalue %43[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %426 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %427 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %428 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %429 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %430 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %431 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %432 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %433 = llvm.extractvalue %116[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %434 = llvm.extractvalue %116[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %435 = llvm.extractvalue %116[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %436 = llvm.extractvalue %116[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %437 = llvm.extractvalue %116[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S7(%416, %417, %418, %419, %420, %402, %421, %422, %423, %424, %425, %414, %arg1, %426, %427, %428, %429, %430, %431, %432, %433, %434, %435, %436, %437) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.double, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %438 = llvm.add %414, %64 : !llvm.i64
    llvm.br ^bb36(%438 : !llvm.i64)
  ^bb38:  // pred: ^bb36
    %439 = llvm.add %402, %64 : !llvm.i64
    llvm.br ^bb34(%439 : !llvm.i64)
  ^bb39:  // pred: ^bb34
    llvm.return
  }
  llvm.func @_mlir_ciface_kernel_gemver_new(%arg0: !llvm.i32, %arg1: !llvm.double, %arg2: !llvm.double, %arg3: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>>, %arg4: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>, %arg5: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>, %arg6: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>, %arg7: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>, %arg8: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>, %arg9: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>, %arg10: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>, %arg11: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>) {
    %0 = llvm.load %arg3 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.load %arg4 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>
    %9 = llvm.extractvalue %8[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %10 = llvm.extractvalue %8[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.extractvalue %8[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %12 = llvm.extractvalue %8[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %13 = llvm.extractvalue %8[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %14 = llvm.load %arg5 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>
    %15 = llvm.extractvalue %14[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %16 = llvm.extractvalue %14[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %17 = llvm.extractvalue %14[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %18 = llvm.extractvalue %14[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %19 = llvm.extractvalue %14[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %20 = llvm.load %arg6 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>
    %21 = llvm.extractvalue %20[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %22 = llvm.extractvalue %20[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %23 = llvm.extractvalue %20[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %24 = llvm.extractvalue %20[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %25 = llvm.extractvalue %20[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %26 = llvm.load %arg7 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>
    %27 = llvm.extractvalue %26[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %28 = llvm.extractvalue %26[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %29 = llvm.extractvalue %26[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %30 = llvm.extractvalue %26[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %31 = llvm.extractvalue %26[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %32 = llvm.load %arg8 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>
    %33 = llvm.extractvalue %32[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %34 = llvm.extractvalue %32[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %35 = llvm.extractvalue %32[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %36 = llvm.extractvalue %32[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %37 = llvm.extractvalue %32[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %38 = llvm.load %arg9 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>
    %39 = llvm.extractvalue %38[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %40 = llvm.extractvalue %38[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %41 = llvm.extractvalue %38[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %42 = llvm.extractvalue %38[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %43 = llvm.extractvalue %38[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %44 = llvm.load %arg10 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>
    %45 = llvm.extractvalue %44[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %46 = llvm.extractvalue %44[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %47 = llvm.extractvalue %44[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %48 = llvm.extractvalue %44[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %49 = llvm.extractvalue %44[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %50 = llvm.load %arg11 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>
    %51 = llvm.extractvalue %50[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %52 = llvm.extractvalue %50[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %53 = llvm.extractvalue %50[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %54 = llvm.extractvalue %50[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %55 = llvm.extractvalue %50[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @kernel_gemver_new(%arg0, %arg1, %arg2, %1, %2, %3, %4, %5, %6, %7, %9, %10, %11, %12, %13, %15, %16, %17, %18, %19, %21, %22, %23, %24, %25, %27, %28, %29, %30, %31, %33, %34, %35, %36, %37, %39, %40, %41, %42, %43, %45, %46, %47, %48, %49, %51, %52, %53, %54, %55) : (!llvm.i32, !llvm.double, !llvm.double, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.return
  }
}

