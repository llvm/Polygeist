module attributes {llvm.data_layout = ""}  {
  llvm.func @kernel_lu(%arg0: !llvm.i32, %arg1: !llvm.ptr<double>, %arg2: !llvm.ptr<double>, %arg3: !llvm.i64, %arg4: !llvm.i64, %arg5: !llvm.i64, %arg6: !llvm.i64, %arg7: !llvm.i64) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg1, %0[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.insertvalue %arg2, %1[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.insertvalue %arg3, %2[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.insertvalue %arg4, %3[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.insertvalue %arg6, %4[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.insertvalue %arg5, %5[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.insertvalue %arg7, %6[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.mlir.constant(0 : index) : !llvm.i64
    %9 = llvm.mlir.constant(1 : index) : !llvm.i64
    %10 = llvm.sext %arg0 : !llvm.i32 to !llvm.i64
    llvm.br ^bb1(%8 : !llvm.i64)
  ^bb1(%11: !llvm.i64):  // 2 preds: ^bb0, ^bb12
    %12 = llvm.icmp "slt" %11, %10 : !llvm.i64
    llvm.cond_br %12, ^bb2(%8 : !llvm.i64), ^bb13
  ^bb2(%13: !llvm.i64):  // 2 preds: ^bb1, ^bb6
    %14 = llvm.icmp "slt" %13, %11 : !llvm.i64
    llvm.cond_br %14, ^bb3, ^bb7(%11 : !llvm.i64)
  ^bb3:  // pred: ^bb2
    %15 = llvm.mlir.constant(1 : index) : !llvm.i64
    %16 = llvm.mlir.constant(1 : index) : !llvm.i64
    %17 = llvm.mlir.null : !llvm.ptr<double>
    %18 = llvm.getelementptr %17[%15] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    %19 = llvm.ptrtoint %18 : !llvm.ptr<double> to !llvm.i64
    %20 = llvm.alloca %19 x !llvm.double : (!llvm.i64) -> !llvm.ptr<double>
    %21 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %22 = llvm.insertvalue %20, %21[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %23 = llvm.insertvalue %20, %22[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %24 = llvm.mlir.constant(0 : index) : !llvm.i64
    %25 = llvm.insertvalue %24, %23[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %26 = llvm.insertvalue %15, %25[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %27 = llvm.insertvalue %16, %26[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %28 = llvm.extractvalue %27[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %29 = llvm.extractvalue %27[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %30 = llvm.extractvalue %27[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %31 = llvm.extractvalue %27[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %32 = llvm.extractvalue %27[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %33 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %34 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %35 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %36 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %37 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %38 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %39 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S0(%28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %11, %13) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.br ^bb4(%8 : !llvm.i64)
  ^bb4(%40: !llvm.i64):  // 2 preds: ^bb3, ^bb5
    %41 = llvm.icmp "slt" %40, %13 : !llvm.i64
    llvm.cond_br %41, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %42 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %43 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %44 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %45 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %46 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %47 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %48 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %49 = llvm.extractvalue %27[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %50 = llvm.extractvalue %27[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %51 = llvm.extractvalue %27[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %52 = llvm.extractvalue %27[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %53 = llvm.extractvalue %27[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S1(%42, %43, %44, %45, %46, %47, %48, %11, %13, %40, %49, %50, %51, %52, %53) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %54 = llvm.add %40, %9 : !llvm.i64
    llvm.br ^bb4(%54 : !llvm.i64)
  ^bb6:  // pred: ^bb4
    %55 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %56 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %57 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %58 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %59 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %60 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %61 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S2(%55, %56, %57, %58, %59, %60, %61, %11, %13) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %62 = llvm.add %13, %9 : !llvm.i64
    llvm.br ^bb2(%62 : !llvm.i64)
  ^bb7(%63: !llvm.i64):  // 2 preds: ^bb2, ^bb11
    %64 = llvm.icmp "slt" %63, %10 : !llvm.i64
    llvm.cond_br %64, ^bb8, ^bb12
  ^bb8:  // pred: ^bb7
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
    %78 = llvm.extractvalue %77[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %79 = llvm.extractvalue %77[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %80 = llvm.extractvalue %77[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %81 = llvm.extractvalue %77[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %82 = llvm.extractvalue %77[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %83 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %84 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %85 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %86 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %87 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %88 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %89 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S3(%78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %11, %63) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.br ^bb9(%8 : !llvm.i64)
  ^bb9(%90: !llvm.i64):  // 2 preds: ^bb8, ^bb10
    %91 = llvm.icmp "slt" %90, %11 : !llvm.i64
    llvm.cond_br %91, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    %92 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %93 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %94 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %95 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %96 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %97 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %98 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %99 = llvm.extractvalue %77[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %100 = llvm.extractvalue %77[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %101 = llvm.extractvalue %77[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %102 = llvm.extractvalue %77[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %103 = llvm.extractvalue %77[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S4(%92, %93, %94, %95, %96, %97, %98, %11, %63, %90, %99, %100, %101, %102, %103) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %104 = llvm.add %90, %9 : !llvm.i64
    llvm.br ^bb9(%104 : !llvm.i64)
  ^bb11:  // pred: ^bb9
    %105 = llvm.add %63, %9 : !llvm.i64
    llvm.br ^bb7(%105 : !llvm.i64)
  ^bb12:  // pred: ^bb7
    %106 = llvm.add %11, %9 : !llvm.i64
    llvm.br ^bb1(%106 : !llvm.i64)
  ^bb13:  // pred: ^bb1
    llvm.return
  }
  llvm.func @_mlir_ciface_kernel_lu(%arg0: !llvm.i32, %arg1: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>>) {
    %0 = llvm.load %arg1 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @kernel_lu(%arg0, %1, %2, %3, %4, %5, %6, %7) : (!llvm.i32, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.return
  }
  llvm.func @S0(%arg0: !llvm.ptr<double>, %arg1: !llvm.ptr<double>, %arg2: !llvm.i64, %arg3: !llvm.i64, %arg4: !llvm.i64, %arg5: !llvm.ptr<double>, %arg6: !llvm.ptr<double>, %arg7: !llvm.i64, %arg8: !llvm.i64, %arg9: !llvm.i64, %arg10: !llvm.i64, %arg11: !llvm.i64, %arg12: !llvm.i64, %arg13: !llvm.i64) attributes {scop.stmt} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %5 = llvm.insertvalue %arg4, %4[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %6 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.insertvalue %arg5, %6[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.insertvalue %arg6, %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.insertvalue %arg7, %8[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %10 = llvm.insertvalue %arg8, %9[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %11 = llvm.insertvalue %arg10, %10[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %12 = llvm.insertvalue %arg9, %11[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %13 = llvm.insertvalue %arg11, %12[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %14 = llvm.mlir.constant(0 : index) : !llvm.i64
    %15 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %16 = llvm.mlir.constant(2000 : index) : !llvm.i64
    %17 = llvm.mul %arg12, %16 : !llvm.i64
    %18 = llvm.add %17, %arg13 : !llvm.i64
    %19 = llvm.getelementptr %15[%18] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    %20 = llvm.load %19 : !llvm.ptr<double>
    %21 = llvm.extractvalue %5[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %22 = llvm.getelementptr %21[%14] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    llvm.store %20, %22 : !llvm.ptr<double>
    llvm.return
  }
  llvm.func @_mlir_ciface_S0(%arg0: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>, %arg1: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>>, %arg2: !llvm.i64, %arg3: !llvm.i64) attributes {scop.stmt} {
    %0 = llvm.load %arg0 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %5 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %6 = llvm.load %arg1 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>>
    %7 = llvm.extractvalue %6[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.extractvalue %6[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.extractvalue %6[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %10 = llvm.extractvalue %6[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %11 = llvm.extractvalue %6[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %12 = llvm.extractvalue %6[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %13 = llvm.extractvalue %6[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S0(%1, %2, %3, %4, %5, %7, %8, %9, %10, %11, %12, %13, %arg2, %arg3) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.return
  }
  llvm.func @S1(%arg0: !llvm.ptr<double>, %arg1: !llvm.ptr<double>, %arg2: !llvm.i64, %arg3: !llvm.i64, %arg4: !llvm.i64, %arg5: !llvm.i64, %arg6: !llvm.i64, %arg7: !llvm.i64, %arg8: !llvm.i64, %arg9: !llvm.i64, %arg10: !llvm.ptr<double>, %arg11: !llvm.ptr<double>, %arg12: !llvm.i64, %arg13: !llvm.i64, %arg14: !llvm.i64) attributes {scop.stmt} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.insertvalue %arg10, %8[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %10 = llvm.insertvalue %arg11, %9[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.insertvalue %arg12, %10[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %12 = llvm.insertvalue %arg13, %11[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %13 = llvm.insertvalue %arg14, %12[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %14 = llvm.mlir.constant(0 : index) : !llvm.i64
    %15 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %16 = llvm.getelementptr %15[%14] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    %17 = llvm.load %16 : !llvm.ptr<double>
    %18 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %19 = llvm.mlir.constant(2000 : index) : !llvm.i64
    %20 = llvm.mul %arg7, %19 : !llvm.i64
    %21 = llvm.add %20, %arg9 : !llvm.i64
    %22 = llvm.getelementptr %18[%21] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    %23 = llvm.load %22 : !llvm.ptr<double>
    %24 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %25 = llvm.mlir.constant(2000 : index) : !llvm.i64
    %26 = llvm.mul %arg9, %25 : !llvm.i64
    %27 = llvm.add %26, %arg8 : !llvm.i64
    %28 = llvm.getelementptr %24[%27] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    %29 = llvm.load %28 : !llvm.ptr<double>
    %30 = llvm.fmul %23, %29 : !llvm.double
    %31 = llvm.fsub %17, %30 : !llvm.double
    %32 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %33 = llvm.mlir.constant(2000 : index) : !llvm.i64
    %34 = llvm.mul %arg7, %33 : !llvm.i64
    %35 = llvm.add %34, %arg8 : !llvm.i64
    %36 = llvm.getelementptr %32[%35] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    llvm.store %31, %36 : !llvm.ptr<double>
    llvm.return
  }
  llvm.func @_mlir_ciface_S1(%arg0: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>>, %arg1: !llvm.i64, %arg2: !llvm.i64, %arg3: !llvm.i64, %arg4: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>) attributes {scop.stmt} {
    %0 = llvm.load %arg0 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>>
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
    llvm.call @S1(%1, %2, %3, %4, %5, %6, %7, %arg1, %arg2, %arg3, %9, %10, %11, %12, %13) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.return
  }
  llvm.func @S2(%arg0: !llvm.ptr<double>, %arg1: !llvm.ptr<double>, %arg2: !llvm.i64, %arg3: !llvm.i64, %arg4: !llvm.i64, %arg5: !llvm.i64, %arg6: !llvm.i64, %arg7: !llvm.i64, %arg8: !llvm.i64) attributes {scop.stmt} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.mlir.constant(2000 : index) : !llvm.i64
    %10 = llvm.mul %arg7, %9 : !llvm.i64
    %11 = llvm.add %10, %arg8 : !llvm.i64
    %12 = llvm.getelementptr %8[%11] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    %13 = llvm.load %12 : !llvm.ptr<double>
    %14 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %15 = llvm.mlir.constant(2000 : index) : !llvm.i64
    %16 = llvm.mul %arg8, %15 : !llvm.i64
    %17 = llvm.add %16, %arg8 : !llvm.i64
    %18 = llvm.getelementptr %14[%17] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    %19 = llvm.load %18 : !llvm.ptr<double>
    %20 = llvm.fdiv %13, %19 : !llvm.double
    %21 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %22 = llvm.mlir.constant(2000 : index) : !llvm.i64
    %23 = llvm.mul %arg7, %22 : !llvm.i64
    %24 = llvm.add %23, %arg8 : !llvm.i64
    %25 = llvm.getelementptr %21[%24] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    llvm.store %20, %25 : !llvm.ptr<double>
    llvm.return
  }
  llvm.func @_mlir_ciface_S2(%arg0: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>>, %arg1: !llvm.i64, %arg2: !llvm.i64) attributes {scop.stmt} {
    %0 = llvm.load %arg0 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S2(%1, %2, %3, %4, %5, %6, %7, %arg1, %arg2) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.return
  }
  llvm.func @S3(%arg0: !llvm.ptr<double>, %arg1: !llvm.ptr<double>, %arg2: !llvm.i64, %arg3: !llvm.i64, %arg4: !llvm.i64, %arg5: !llvm.ptr<double>, %arg6: !llvm.ptr<double>, %arg7: !llvm.i64, %arg8: !llvm.i64, %arg9: !llvm.i64, %arg10: !llvm.i64, %arg11: !llvm.i64, %arg12: !llvm.i64, %arg13: !llvm.i64) attributes {scop.stmt} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %5 = llvm.insertvalue %arg4, %4[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %6 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.insertvalue %arg5, %6[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.insertvalue %arg6, %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.insertvalue %arg7, %8[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %10 = llvm.insertvalue %arg8, %9[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %11 = llvm.insertvalue %arg10, %10[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %12 = llvm.insertvalue %arg9, %11[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %13 = llvm.insertvalue %arg11, %12[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %14 = llvm.mlir.constant(0 : index) : !llvm.i64
    %15 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %16 = llvm.mlir.constant(2000 : index) : !llvm.i64
    %17 = llvm.mul %arg12, %16 : !llvm.i64
    %18 = llvm.add %17, %arg13 : !llvm.i64
    %19 = llvm.getelementptr %15[%18] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    %20 = llvm.load %19 : !llvm.ptr<double>
    %21 = llvm.extractvalue %5[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %22 = llvm.getelementptr %21[%14] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    llvm.store %20, %22 : !llvm.ptr<double>
    llvm.return
  }
  llvm.func @_mlir_ciface_S3(%arg0: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>, %arg1: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>>, %arg2: !llvm.i64, %arg3: !llvm.i64) attributes {scop.stmt} {
    %0 = llvm.load %arg0 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %5 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %6 = llvm.load %arg1 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>>
    %7 = llvm.extractvalue %6[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.extractvalue %6[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.extractvalue %6[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %10 = llvm.extractvalue %6[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %11 = llvm.extractvalue %6[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %12 = llvm.extractvalue %6[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %13 = llvm.extractvalue %6[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S3(%1, %2, %3, %4, %5, %7, %8, %9, %10, %11, %12, %13, %arg2, %arg3) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.return
  }
  llvm.func @S4(%arg0: !llvm.ptr<double>, %arg1: !llvm.ptr<double>, %arg2: !llvm.i64, %arg3: !llvm.i64, %arg4: !llvm.i64, %arg5: !llvm.i64, %arg6: !llvm.i64, %arg7: !llvm.i64, %arg8: !llvm.i64, %arg9: !llvm.i64, %arg10: !llvm.ptr<double>, %arg11: !llvm.ptr<double>, %arg12: !llvm.i64, %arg13: !llvm.i64, %arg14: !llvm.i64) attributes {scop.stmt} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.insertvalue %arg10, %8[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %10 = llvm.insertvalue %arg11, %9[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.insertvalue %arg12, %10[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %12 = llvm.insertvalue %arg13, %11[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %13 = llvm.insertvalue %arg14, %12[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %14 = llvm.mlir.constant(0 : index) : !llvm.i64
    %15 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %16 = llvm.getelementptr %15[%14] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    %17 = llvm.load %16 : !llvm.ptr<double>
    %18 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %19 = llvm.mlir.constant(2000 : index) : !llvm.i64
    %20 = llvm.mul %arg7, %19 : !llvm.i64
    %21 = llvm.add %20, %arg9 : !llvm.i64
    %22 = llvm.getelementptr %18[%21] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    %23 = llvm.load %22 : !llvm.ptr<double>
    %24 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %25 = llvm.mlir.constant(2000 : index) : !llvm.i64
    %26 = llvm.mul %arg9, %25 : !llvm.i64
    %27 = llvm.add %26, %arg8 : !llvm.i64
    %28 = llvm.getelementptr %24[%27] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    %29 = llvm.load %28 : !llvm.ptr<double>
    %30 = llvm.fmul %23, %29 : !llvm.double
    %31 = llvm.fsub %17, %30 : !llvm.double
    %32 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %33 = llvm.mlir.constant(2000 : index) : !llvm.i64
    %34 = llvm.mul %arg7, %33 : !llvm.i64
    %35 = llvm.add %34, %arg8 : !llvm.i64
    %36 = llvm.getelementptr %32[%35] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    llvm.store %31, %36 : !llvm.ptr<double>
    llvm.return
  }
  llvm.func @_mlir_ciface_S4(%arg0: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>>, %arg1: !llvm.i64, %arg2: !llvm.i64, %arg3: !llvm.i64, %arg4: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>>) attributes {scop.stmt} {
    %0 = llvm.load %arg0 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>>
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
    llvm.call @S4(%1, %2, %3, %4, %5, %6, %7, %arg1, %arg2, %arg3, %9, %10, %11, %12, %13) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.return
  }
  llvm.func @kernel_lu_new(%arg0: !llvm.i32, %arg1: !llvm.ptr<double>, %arg2: !llvm.ptr<double>, %arg3: !llvm.i64, %arg4: !llvm.i64, %arg5: !llvm.i64, %arg6: !llvm.i64, %arg7: !llvm.i64) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg1, %0[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.insertvalue %arg2, %1[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.insertvalue %arg3, %2[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.insertvalue %arg4, %3[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.insertvalue %arg6, %4[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.insertvalue %arg5, %5[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.insertvalue %arg7, %6[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.mlir.constant(2 : index) : !llvm.i64
    %9 = llvm.mlir.constant(0 : index) : !llvm.i64
    %10 = llvm.mlir.constant(1 : index) : !llvm.i64
    %11 = llvm.mlir.constant(1 : index) : !llvm.i64
    %12 = llvm.mlir.constant(1 : index) : !llvm.i64
    %13 = llvm.mlir.null : !llvm.ptr<double>
    %14 = llvm.getelementptr %13[%11] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    %15 = llvm.ptrtoint %14 : !llvm.ptr<double> to !llvm.i64
    %16 = llvm.alloca %15 x !llvm.double : (!llvm.i64) -> !llvm.ptr<double>
    %17 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %18 = llvm.insertvalue %16, %17[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %19 = llvm.insertvalue %16, %18[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %20 = llvm.mlir.constant(0 : index) : !llvm.i64
    %21 = llvm.insertvalue %20, %19[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %22 = llvm.insertvalue %11, %21[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %23 = llvm.insertvalue %12, %22[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %24 = llvm.mlir.constant(1 : index) : !llvm.i64
    %25 = llvm.mlir.constant(1 : index) : !llvm.i64
    %26 = llvm.mlir.null : !llvm.ptr<double>
    %27 = llvm.getelementptr %26[%24] : (!llvm.ptr<double>, !llvm.i64) -> !llvm.ptr<double>
    %28 = llvm.ptrtoint %27 : !llvm.ptr<double> to !llvm.i64
    %29 = llvm.alloca %28 x !llvm.double : (!llvm.i64) -> !llvm.ptr<double>
    %30 = llvm.mlir.undef : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %31 = llvm.insertvalue %29, %30[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %32 = llvm.insertvalue %29, %31[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %33 = llvm.mlir.constant(0 : index) : !llvm.i64
    %34 = llvm.insertvalue %33, %32[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %35 = llvm.insertvalue %24, %34[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %36 = llvm.insertvalue %25, %35[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %37 = llvm.sext %arg0 : !llvm.i32 to !llvm.i64
    llvm.br ^bb1(%9 : !llvm.i64)
  ^bb1(%38: !llvm.i64):  // 2 preds: ^bb0, ^bb2
    %39 = llvm.icmp "slt" %38, %37 : !llvm.i64
    llvm.cond_br %39, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %40 = llvm.extractvalue %23[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %41 = llvm.extractvalue %23[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %42 = llvm.extractvalue %23[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %43 = llvm.extractvalue %23[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %44 = llvm.extractvalue %23[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %45 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %46 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %47 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %48 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %49 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %50 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %51 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S3(%40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %9, %38) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %52 = llvm.add %38, %10 : !llvm.i64
    llvm.br ^bb1(%52 : !llvm.i64)
  ^bb3:  // pred: ^bb1
    %53 = llvm.extractvalue %36[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %54 = llvm.extractvalue %36[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %55 = llvm.extractvalue %36[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %56 = llvm.extractvalue %36[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %57 = llvm.extractvalue %36[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %58 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %59 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %60 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %61 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %62 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %63 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %64 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S0(%53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %10, %9) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %65 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %66 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %67 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %68 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %69 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %70 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %71 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S2(%65, %66, %67, %68, %69, %70, %71, %10, %9) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.br ^bb4(%10 : !llvm.i64)
  ^bb4(%72: !llvm.i64):  // 2 preds: ^bb3, ^bb5
    %73 = llvm.icmp "slt" %72, %37 : !llvm.i64
    llvm.cond_br %73, ^bb5, ^bb6(%8 : !llvm.i64)
  ^bb5:  // pred: ^bb4
    %74 = llvm.extractvalue %23[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %75 = llvm.extractvalue %23[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %76 = llvm.extractvalue %23[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %77 = llvm.extractvalue %23[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %78 = llvm.extractvalue %23[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %79 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %80 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %81 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %82 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %83 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %84 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %85 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S3(%74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %10, %72) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %86 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %87 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %88 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %89 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %90 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %91 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %92 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %93 = llvm.extractvalue %23[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %94 = llvm.extractvalue %23[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %95 = llvm.extractvalue %23[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %96 = llvm.extractvalue %23[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %97 = llvm.extractvalue %23[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S4(%86, %87, %88, %89, %90, %91, %92, %10, %72, %9, %93, %94, %95, %96, %97) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %98 = llvm.add %72, %10 : !llvm.i64
    llvm.br ^bb4(%98 : !llvm.i64)
  ^bb6(%99: !llvm.i64):  // 2 preds: ^bb4, ^bb18
    %100 = llvm.icmp "slt" %99, %37 : !llvm.i64
    llvm.cond_br %100, ^bb7, ^bb19
  ^bb7:  // pred: ^bb6
    %101 = llvm.extractvalue %36[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %102 = llvm.extractvalue %36[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %103 = llvm.extractvalue %36[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %104 = llvm.extractvalue %36[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %105 = llvm.extractvalue %36[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %106 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %107 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %108 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %109 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %110 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %111 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %112 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S0(%101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %99, %9) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %113 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %114 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %115 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %116 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %117 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %118 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %119 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S2(%113, %114, %115, %116, %117, %118, %119, %99, %9) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.br ^bb8(%10 : !llvm.i64)
  ^bb8(%120: !llvm.i64):  // 2 preds: ^bb7, ^bb12
    %121 = llvm.icmp "slt" %120, %99 : !llvm.i64
    llvm.cond_br %121, ^bb9, ^bb13(%99 : !llvm.i64)
  ^bb9:  // pred: ^bb8
    %122 = llvm.extractvalue %36[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %123 = llvm.extractvalue %36[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %124 = llvm.extractvalue %36[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %125 = llvm.extractvalue %36[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %126 = llvm.extractvalue %36[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %127 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %128 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %129 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %130 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %131 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %132 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %133 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S0(%122, %123, %124, %125, %126, %127, %128, %129, %130, %131, %132, %133, %99, %120) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.br ^bb10(%9 : !llvm.i64)
  ^bb10(%134: !llvm.i64):  // 2 preds: ^bb9, ^bb11
    %135 = llvm.icmp "slt" %134, %120 : !llvm.i64
    llvm.cond_br %135, ^bb11, ^bb12
  ^bb11:  // pred: ^bb10
    %136 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %137 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %138 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %139 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %140 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %141 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %142 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %143 = llvm.extractvalue %36[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %144 = llvm.extractvalue %36[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %145 = llvm.extractvalue %36[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %146 = llvm.extractvalue %36[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %147 = llvm.extractvalue %36[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S1(%136, %137, %138, %139, %140, %141, %142, %99, %120, %134, %143, %144, %145, %146, %147) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %148 = llvm.add %134, %10 : !llvm.i64
    llvm.br ^bb10(%148 : !llvm.i64)
  ^bb12:  // pred: ^bb10
    %149 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %150 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %151 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %152 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %153 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %154 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %155 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S2(%149, %150, %151, %152, %153, %154, %155, %99, %120) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %156 = llvm.add %120, %10 : !llvm.i64
    llvm.br ^bb8(%156 : !llvm.i64)
  ^bb13(%157: !llvm.i64):  // 2 preds: ^bb8, ^bb17
    %158 = llvm.icmp "slt" %157, %37 : !llvm.i64
    llvm.cond_br %158, ^bb14, ^bb18
  ^bb14:  // pred: ^bb13
    %159 = llvm.extractvalue %23[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %160 = llvm.extractvalue %23[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %161 = llvm.extractvalue %23[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %162 = llvm.extractvalue %23[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %163 = llvm.extractvalue %23[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %164 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %165 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %166 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %167 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %168 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %169 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %170 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @S3(%159, %160, %161, %162, %163, %164, %165, %166, %167, %168, %169, %170, %99, %157) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.br ^bb15(%9 : !llvm.i64)
  ^bb15(%171: !llvm.i64):  // 2 preds: ^bb14, ^bb16
    %172 = llvm.icmp "slt" %171, %99 : !llvm.i64
    llvm.cond_br %172, ^bb16, ^bb17
  ^bb16:  // pred: ^bb15
    %173 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %174 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %175 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %176 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %177 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %178 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %179 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %180 = llvm.extractvalue %23[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %181 = llvm.extractvalue %23[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %182 = llvm.extractvalue %23[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %183 = llvm.extractvalue %23[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    %184 = llvm.extractvalue %23[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @S4(%173, %174, %175, %176, %177, %178, %179, %99, %157, %171, %180, %181, %182, %183, %184) : (!llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %185 = llvm.add %171, %10 : !llvm.i64
    llvm.br ^bb15(%185 : !llvm.i64)
  ^bb17:  // pred: ^bb15
    %186 = llvm.add %157, %10 : !llvm.i64
    llvm.br ^bb13(%186 : !llvm.i64)
  ^bb18:  // pred: ^bb13
    %187 = llvm.add %99, %10 : !llvm.i64
    llvm.br ^bb6(%187 : !llvm.i64)
  ^bb19:  // pred: ^bb6
    llvm.return
  }
  llvm.func @_mlir_ciface_kernel_lu_new(%arg0: !llvm.i32, %arg1: !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>>) {
    %0 = llvm.load %arg1 : !llvm.ptr<struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr<double>, ptr<double>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @kernel_lu_new(%arg0, %1, %2, %3, %4, %5, %6, %7) : (!llvm.i32, !llvm.ptr<double>, !llvm.ptr<double>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.return
  }
}

