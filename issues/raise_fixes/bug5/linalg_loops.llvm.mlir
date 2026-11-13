module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.func @kernel_jacobi_2d(%arg0: i32, %arg1: i32, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: !llvm.ptr, %arg10: !llvm.ptr, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: i64) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg2, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.insertvalue %arg3, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.insertvalue %arg4, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.insertvalue %arg5, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.insertvalue %arg7, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.insertvalue %arg6, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.insertvalue %arg8, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.insertvalue %arg9, %8[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %10 = llvm.insertvalue %arg10, %9[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.insertvalue %arg11, %10[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %12 = llvm.insertvalue %arg12, %11[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.insertvalue %arg14, %12[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.insertvalue %arg13, %13[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.insertvalue %arg15, %14[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.mlir.constant(2 : index) : i64
    %17 = llvm.mlir.constant(-1 : index) : i64
    %18 = llvm.mlir.constant(0 : index) : i64
    %19 = llvm.mlir.constant(1 : index) : i64
    %20 = llvm.mlir.constant(2.000000e-01 : f64) : f64
    %21 = llvm.sext %arg1 : i32 to i64
    %22 = llvm.sext %arg0 : i32 to i64
    %23 = llvm.add %21, %17  : i64
    %24 = llvm.add %21, %17  : i64
    %25 = llvm.sub %24, %19  : i64
    %26 = llvm.add %21, %17  : i64
    %27 = llvm.sub %26, %19  : i64
    %28 = llvm.sub %23, %19  : i64
    %29 = llvm.add %21, %17  : i64
    %30 = llvm.sub %29, %19  : i64
    llvm.br ^bb1(%18 : i64)
  ^bb1(%31: i64):  // 2 preds: ^bb0, ^bb14
    %32 = llvm.icmp "slt" %31, %22 : i64
    llvm.cond_br %32, ^bb2, ^bb15
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%18 : i64)
  ^bb3(%33: i64):  // 2 preds: ^bb2, ^bb7
    %34 = llvm.icmp "slt" %33, %27 : i64
    llvm.cond_br %34, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%18 : i64)
  ^bb5(%35: i64):  // 2 preds: ^bb4, ^bb6
    %36 = llvm.icmp "slt" %35, %25 : i64
    llvm.cond_br %36, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %37 = llvm.add %33, %19  : i64
    %38 = llvm.add %35, %19  : i64
    %39 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %40 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %41 = llvm.mul %37, %40  : i64
    %42 = llvm.add %41, %38  : i64
    %43 = llvm.getelementptr %39[%42] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %44 = llvm.load %43 : !llvm.ptr -> f64
    %45 = llvm.add %33, %19  : i64
    %46 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %47 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %48 = llvm.mul %45, %47  : i64
    %49 = llvm.add %48, %35  : i64
    %50 = llvm.getelementptr %46[%49] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %51 = llvm.load %50 : !llvm.ptr -> f64
    %52 = llvm.add %33, %19  : i64
    %53 = llvm.add %35, %16  : i64
    %54 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %55 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %56 = llvm.mul %52, %55  : i64
    %57 = llvm.add %56, %53  : i64
    %58 = llvm.getelementptr %54[%57] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %59 = llvm.load %58 : !llvm.ptr -> f64
    %60 = llvm.add %33, %16  : i64
    %61 = llvm.add %35, %19  : i64
    %62 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %63 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %64 = llvm.mul %60, %63  : i64
    %65 = llvm.add %64, %61  : i64
    %66 = llvm.getelementptr %62[%65] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %67 = llvm.load %66 : !llvm.ptr -> f64
    %68 = llvm.add %35, %19  : i64
    %69 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %70 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %71 = llvm.mul %33, %70  : i64
    %72 = llvm.add %71, %68  : i64
    %73 = llvm.getelementptr %69[%72] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %74 = llvm.load %73 : !llvm.ptr -> f64
    %75 = llvm.fadd %44, %51  : f64
    %76 = llvm.fadd %75, %59  : f64
    %77 = llvm.fadd %76, %67  : f64
    %78 = llvm.fadd %77, %74  : f64
    %79 = llvm.fmul %78, %20  : f64
    %80 = llvm.add %33, %19  : i64
    %81 = llvm.add %35, %19  : i64
    %82 = llvm.extractvalue %15[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %83 = llvm.extractvalue %15[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %84 = llvm.mul %80, %83  : i64
    %85 = llvm.add %84, %81  : i64
    %86 = llvm.getelementptr %82[%85] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %79, %86 : f64, !llvm.ptr
    %87 = llvm.add %35, %19  : i64
    llvm.br ^bb5(%87 : i64)
  ^bb7:  // pred: ^bb5
    %88 = llvm.add %33, %19  : i64
    llvm.br ^bb3(%88 : i64)
  ^bb8:  // pred: ^bb3
    llvm.br ^bb9(%18 : i64)
  ^bb9(%89: i64):  // 2 preds: ^bb8, ^bb13
    %90 = llvm.icmp "slt" %89, %30 : i64
    llvm.cond_br %90, ^bb10, ^bb14
  ^bb10:  // pred: ^bb9
    llvm.br ^bb11(%18 : i64)
  ^bb11(%91: i64):  // 2 preds: ^bb10, ^bb12
    %92 = llvm.icmp "slt" %91, %28 : i64
    llvm.cond_br %92, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    %93 = llvm.add %89, %19  : i64
    %94 = llvm.add %91, %19  : i64
    %95 = llvm.extractvalue %15[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %96 = llvm.extractvalue %15[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %97 = llvm.mul %93, %96  : i64
    %98 = llvm.add %97, %94  : i64
    %99 = llvm.getelementptr %95[%98] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %100 = llvm.load %99 : !llvm.ptr -> f64
    %101 = llvm.add %89, %19  : i64
    %102 = llvm.extractvalue %15[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %103 = llvm.extractvalue %15[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %104 = llvm.mul %101, %103  : i64
    %105 = llvm.add %104, %91  : i64
    %106 = llvm.getelementptr %102[%105] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %107 = llvm.load %106 : !llvm.ptr -> f64
    %108 = llvm.add %89, %19  : i64
    %109 = llvm.add %91, %16  : i64
    %110 = llvm.extractvalue %15[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %111 = llvm.extractvalue %15[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %112 = llvm.mul %108, %111  : i64
    %113 = llvm.add %112, %109  : i64
    %114 = llvm.getelementptr %110[%113] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %115 = llvm.load %114 : !llvm.ptr -> f64
    %116 = llvm.add %89, %16  : i64
    %117 = llvm.add %91, %19  : i64
    %118 = llvm.extractvalue %15[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %119 = llvm.extractvalue %15[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %120 = llvm.mul %116, %119  : i64
    %121 = llvm.add %120, %117  : i64
    %122 = llvm.getelementptr %118[%121] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %123 = llvm.load %122 : !llvm.ptr -> f64
    %124 = llvm.add %91, %19  : i64
    %125 = llvm.extractvalue %15[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %126 = llvm.extractvalue %15[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %127 = llvm.mul %89, %126  : i64
    %128 = llvm.add %127, %124  : i64
    %129 = llvm.getelementptr %125[%128] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %130 = llvm.load %129 : !llvm.ptr -> f64
    %131 = llvm.fadd %100, %107  : f64
    %132 = llvm.fadd %131, %115  : f64
    %133 = llvm.fadd %132, %123  : f64
    %134 = llvm.fadd %133, %130  : f64
    %135 = llvm.fmul %134, %20  : f64
    %136 = llvm.add %89, %19  : i64
    %137 = llvm.add %91, %19  : i64
    %138 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %139 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %140 = llvm.mul %136, %139  : i64
    %141 = llvm.add %140, %137  : i64
    %142 = llvm.getelementptr %138[%141] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %135, %142 : f64, !llvm.ptr
    %143 = llvm.add %91, %19  : i64
    llvm.br ^bb11(%143 : i64)
  ^bb13:  // pred: ^bb11
    %144 = llvm.add %89, %19  : i64
    llvm.br ^bb9(%144 : i64)
  ^bb14:  // pred: ^bb9
    %145 = llvm.add %31, %19  : i64
    llvm.br ^bb1(%145 : i64)
  ^bb15:  // pred: ^bb1
    llvm.return
  }
}

