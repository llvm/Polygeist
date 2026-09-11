module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @kernel_gesummv(%arg0: i32, %arg1: f64, %arg2: f64, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: !llvm.ptr, %arg11: !llvm.ptr, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: !llvm.ptr, %arg18: !llvm.ptr, %arg19: i64, %arg20: i64, %arg21: i64, %arg22: !llvm.ptr, %arg23: !llvm.ptr, %arg24: i64, %arg25: i64, %arg26: i64, %arg27: !llvm.ptr, %arg28: !llvm.ptr, %arg29: i64, %arg30: i64, %arg31: i64) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg3, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.insertvalue %arg4, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.insertvalue %arg5, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.insertvalue %arg6, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.insertvalue %arg8, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.insertvalue %arg7, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.insertvalue %arg9, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.insertvalue %arg10, %8[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %10 = llvm.insertvalue %arg11, %9[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.insertvalue %arg12, %10[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %12 = llvm.insertvalue %arg13, %11[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.insertvalue %arg15, %12[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.insertvalue %arg14, %13[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.insertvalue %arg16, %14[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %17 = llvm.insertvalue %arg17, %16[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %18 = llvm.insertvalue %arg18, %17[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %19 = llvm.insertvalue %arg19, %18[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %20 = llvm.insertvalue %arg20, %19[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %21 = llvm.insertvalue %arg21, %20[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %22 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %23 = llvm.insertvalue %arg22, %22[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %24 = llvm.insertvalue %arg23, %23[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %25 = llvm.insertvalue %arg24, %24[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %26 = llvm.insertvalue %arg25, %25[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %27 = llvm.insertvalue %arg26, %26[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %28 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %29 = llvm.insertvalue %arg27, %28[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %30 = llvm.insertvalue %arg28, %29[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %31 = llvm.insertvalue %arg29, %30[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %32 = llvm.insertvalue %arg30, %31[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %33 = llvm.insertvalue %arg31, %32[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %34 = llvm.mlir.constant(0 : index) : i64
    %35 = llvm.mlir.constant(1 : index) : i64
    %36 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %37 = llvm.sext %arg0 : i32 to i64
    %38 = llvm.mlir.constant(1 : index) : i64
    %39 = llvm.extractvalue %21[3] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %40 = llvm.alloca %38 x !llvm.array<1 x i64> : (i64) -> !llvm.ptr
    llvm.store %39, %40 : !llvm.array<1 x i64>, !llvm.ptr
    %41 = llvm.getelementptr %40[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x i64>
    %42 = llvm.load %41 : !llvm.ptr -> i64
    %43 = llvm.mlir.constant(1 : index) : i64
    %44 = llvm.mlir.zero : !llvm.ptr
    %45 = llvm.getelementptr %44[%42] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %46 = llvm.ptrtoint %45 : !llvm.ptr to i64
    %47 = llvm.mlir.constant(64 : index) : i64
    %48 = llvm.add %46, %47  : i64
    %49 = llvm.call @malloc(%48) : (i64) -> !llvm.ptr
    %50 = llvm.ptrtoint %49 : !llvm.ptr to i64
    %51 = llvm.mlir.constant(1 : index) : i64
    %52 = llvm.sub %47, %51  : i64
    %53 = llvm.add %50, %52  : i64
    %54 = llvm.urem %53, %47  : i64
    %55 = llvm.sub %53, %54  : i64
    %56 = llvm.inttoptr %55 : i64 to !llvm.ptr
    %57 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %58 = llvm.insertvalue %49, %57[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %59 = llvm.insertvalue %56, %58[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %60 = llvm.mlir.constant(0 : index) : i64
    %61 = llvm.insertvalue %60, %59[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %62 = llvm.insertvalue %42, %61[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %63 = llvm.insertvalue %43, %62[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb1(%34 : i64)
  ^bb1(%64: i64):  // 2 preds: ^bb0, ^bb2
    %65 = llvm.icmp "slt" %64, %42 : i64
    llvm.cond_br %65, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %66 = llvm.getelementptr %56[%64] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %36, %66 : f64, !llvm.ptr
    %67 = llvm.add %64, %35  : i64
    llvm.br ^bb1(%67 : i64)
  ^bb3:  // pred: ^bb1
    %68 = llvm.mlir.constant(1 : index) : i64
    %69 = llvm.extractvalue %33[3] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %70 = llvm.alloca %68 x !llvm.array<1 x i64> : (i64) -> !llvm.ptr
    llvm.store %69, %70 : !llvm.array<1 x i64>, !llvm.ptr
    %71 = llvm.getelementptr %70[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x i64>
    %72 = llvm.load %71 : !llvm.ptr -> i64
    %73 = llvm.mlir.constant(1 : index) : i64
    %74 = llvm.mlir.zero : !llvm.ptr
    %75 = llvm.getelementptr %74[%72] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %76 = llvm.ptrtoint %75 : !llvm.ptr to i64
    %77 = llvm.mlir.constant(64 : index) : i64
    %78 = llvm.add %76, %77  : i64
    %79 = llvm.call @malloc(%78) : (i64) -> !llvm.ptr
    %80 = llvm.ptrtoint %79 : !llvm.ptr to i64
    %81 = llvm.mlir.constant(1 : index) : i64
    %82 = llvm.sub %77, %81  : i64
    %83 = llvm.add %80, %82  : i64
    %84 = llvm.urem %83, %77  : i64
    %85 = llvm.sub %83, %84  : i64
    %86 = llvm.inttoptr %85 : i64 to !llvm.ptr
    %87 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %88 = llvm.insertvalue %79, %87[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %89 = llvm.insertvalue %86, %88[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %90 = llvm.mlir.constant(0 : index) : i64
    %91 = llvm.insertvalue %90, %89[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %92 = llvm.insertvalue %72, %91[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %93 = llvm.insertvalue %73, %92[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb4(%34 : i64)
  ^bb4(%94: i64):  // 2 preds: ^bb3, ^bb5
    %95 = llvm.icmp "slt" %94, %72 : i64
    llvm.cond_br %95, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %96 = llvm.getelementptr %86[%94] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %36, %96 : f64, !llvm.ptr
    %97 = llvm.add %94, %35  : i64
    llvm.br ^bb4(%97 : i64)
  ^bb6:  // pred: ^bb4
    %98 = llvm.extractvalue %7[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %99 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %100 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %101 = llvm.insertvalue %98, %100[0] : !llvm.struct<(ptr, ptr, i64)> 
    %102 = llvm.insertvalue %99, %101[1] : !llvm.struct<(ptr, ptr, i64)> 
    %103 = llvm.mlir.constant(0 : index) : i64
    %104 = llvm.insertvalue %103, %102[2] : !llvm.struct<(ptr, ptr, i64)> 
    %105 = llvm.extractvalue %7[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %106 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %107 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %108 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %109 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %110 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %111 = llvm.insertvalue %98, %110[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %112 = llvm.insertvalue %99, %111[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %113 = llvm.mlir.constant(0 : index) : i64
    %114 = llvm.insertvalue %113, %112[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %115 = llvm.insertvalue %37, %114[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %116 = llvm.insertvalue %108, %115[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %117 = llvm.insertvalue %37, %116[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %118 = llvm.mlir.constant(1 : index) : i64
    %119 = llvm.insertvalue %118, %117[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %120 = llvm.extractvalue %27[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %121 = llvm.extractvalue %27[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %122 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %123 = llvm.insertvalue %120, %122[0] : !llvm.struct<(ptr, ptr, i64)> 
    %124 = llvm.insertvalue %121, %123[1] : !llvm.struct<(ptr, ptr, i64)> 
    %125 = llvm.mlir.constant(0 : index) : i64
    %126 = llvm.insertvalue %125, %124[2] : !llvm.struct<(ptr, ptr, i64)> 
    %127 = llvm.extractvalue %27[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %128 = llvm.extractvalue %27[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %129 = llvm.extractvalue %27[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %130 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %131 = llvm.insertvalue %120, %130[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %132 = llvm.insertvalue %121, %131[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %133 = llvm.mlir.constant(0 : index) : i64
    %134 = llvm.insertvalue %133, %132[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %135 = llvm.insertvalue %37, %134[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %136 = llvm.mlir.constant(1 : index) : i64
    %137 = llvm.insertvalue %136, %135[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %138 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %139 = llvm.insertvalue %49, %138[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %140 = llvm.insertvalue %56, %139[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %141 = llvm.mlir.constant(0 : index) : i64
    %142 = llvm.insertvalue %141, %140[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %143 = llvm.insertvalue %37, %142[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %144 = llvm.mlir.constant(1 : index) : i64
    %145 = llvm.insertvalue %144, %143[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb7(%34 : i64)
  ^bb7(%146: i64):  // 2 preds: ^bb6, ^bb11
    %147 = llvm.icmp "slt" %146, %37 : i64
    llvm.cond_br %147, ^bb8, ^bb12
  ^bb8:  // pred: ^bb7
    llvm.br ^bb9(%34 : i64)
  ^bb9(%148: i64):  // 2 preds: ^bb8, ^bb10
    %149 = llvm.icmp "slt" %148, %37 : i64
    llvm.cond_br %149, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    %150 = llvm.getelementptr %99[%113] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %151 = llvm.mul %146, %108  : i64
    %152 = llvm.add %151, %148  : i64
    %153 = llvm.getelementptr %150[%152] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %154 = llvm.load %153 : !llvm.ptr -> f64
    %155 = llvm.getelementptr %121[%148] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %156 = llvm.load %155 : !llvm.ptr -> f64
    %157 = llvm.getelementptr %56[%146] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %158 = llvm.load %157 : !llvm.ptr -> f64
    %159 = llvm.fmul %154, %156  : f64
    %160 = llvm.fadd %159, %158  : f64
    %161 = llvm.getelementptr %56[%146] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %160, %161 : f64, !llvm.ptr
    %162 = llvm.add %148, %35  : i64
    llvm.br ^bb9(%162 : i64)
  ^bb11:  // pred: ^bb9
    %163 = llvm.add %146, %35  : i64
    llvm.br ^bb7(%163 : i64)
  ^bb12:  // pred: ^bb7
    %164 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %165 = llvm.insertvalue %49, %164[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %166 = llvm.insertvalue %56, %165[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %167 = llvm.mlir.constant(0 : index) : i64
    %168 = llvm.insertvalue %167, %166[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %169 = llvm.insertvalue %37, %168[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %170 = llvm.mlir.constant(1 : index) : i64
    %171 = llvm.insertvalue %170, %169[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %172 = llvm.mlir.constant(1 : index) : i64
    %173 = llvm.mul %37, %172  : i64
    %174 = llvm.mlir.zero : !llvm.ptr
    %175 = llvm.getelementptr %174[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %176 = llvm.ptrtoint %175 : !llvm.ptr to i64
    %177 = llvm.mul %173, %176  : i64
    %178 = llvm.getelementptr %56[%141] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %179 = llvm.getelementptr %56[%167] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%179, %178, %177) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %180 = llvm.mlir.constant(1 : index) : i64
    %181 = llvm.mlir.zero : !llvm.ptr
    %182 = llvm.getelementptr %181[%42] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %183 = llvm.ptrtoint %182 : !llvm.ptr to i64
    %184 = llvm.mlir.constant(64 : index) : i64
    %185 = llvm.add %183, %184  : i64
    %186 = llvm.call @malloc(%185) : (i64) -> !llvm.ptr
    %187 = llvm.ptrtoint %186 : !llvm.ptr to i64
    %188 = llvm.mlir.constant(1 : index) : i64
    %189 = llvm.sub %184, %188  : i64
    %190 = llvm.add %187, %189  : i64
    %191 = llvm.urem %190, %184  : i64
    %192 = llvm.sub %190, %191  : i64
    %193 = llvm.inttoptr %192 : i64 to !llvm.ptr
    %194 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %195 = llvm.insertvalue %186, %194[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %196 = llvm.insertvalue %193, %195[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %197 = llvm.mlir.constant(0 : index) : i64
    %198 = llvm.insertvalue %197, %196[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %199 = llvm.insertvalue %42, %198[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %200 = llvm.insertvalue %180, %199[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %201 = llvm.mlir.constant(1 : index) : i64
    %202 = llvm.mul %42, %201  : i64
    %203 = llvm.mlir.zero : !llvm.ptr
    %204 = llvm.getelementptr %203[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %205 = llvm.ptrtoint %204 : !llvm.ptr to i64
    %206 = llvm.mul %202, %205  : i64
    %207 = llvm.getelementptr %56[%60] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %208 = llvm.getelementptr %193[%197] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%208, %207, %206) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %209 = llvm.mlir.constant(1 : index) : i64
    %210 = llvm.mul %42, %209  : i64
    %211 = llvm.mlir.zero : !llvm.ptr
    %212 = llvm.getelementptr %211[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %213 = llvm.ptrtoint %212 : !llvm.ptr to i64
    %214 = llvm.mul %210, %213  : i64
    %215 = llvm.getelementptr %193[%197] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %216 = llvm.extractvalue %21[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %217 = llvm.extractvalue %21[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %218 = llvm.getelementptr %216[%217] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%218, %215, %214) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %219 = llvm.extractvalue %15[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %220 = llvm.extractvalue %15[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %221 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %222 = llvm.insertvalue %219, %221[0] : !llvm.struct<(ptr, ptr, i64)> 
    %223 = llvm.insertvalue %220, %222[1] : !llvm.struct<(ptr, ptr, i64)> 
    %224 = llvm.mlir.constant(0 : index) : i64
    %225 = llvm.insertvalue %224, %223[2] : !llvm.struct<(ptr, ptr, i64)> 
    %226 = llvm.extractvalue %15[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %227 = llvm.extractvalue %15[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %228 = llvm.extractvalue %15[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %229 = llvm.extractvalue %15[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %230 = llvm.extractvalue %15[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %231 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %232 = llvm.insertvalue %219, %231[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %233 = llvm.insertvalue %220, %232[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %234 = llvm.mlir.constant(0 : index) : i64
    %235 = llvm.insertvalue %234, %233[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %236 = llvm.insertvalue %37, %235[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %237 = llvm.insertvalue %229, %236[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %238 = llvm.insertvalue %37, %237[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %239 = llvm.mlir.constant(1 : index) : i64
    %240 = llvm.insertvalue %239, %238[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %241 = llvm.extractvalue %27[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %242 = llvm.extractvalue %27[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %243 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %244 = llvm.insertvalue %241, %243[0] : !llvm.struct<(ptr, ptr, i64)> 
    %245 = llvm.insertvalue %242, %244[1] : !llvm.struct<(ptr, ptr, i64)> 
    %246 = llvm.mlir.constant(0 : index) : i64
    %247 = llvm.insertvalue %246, %245[2] : !llvm.struct<(ptr, ptr, i64)> 
    %248 = llvm.extractvalue %27[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %249 = llvm.extractvalue %27[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %250 = llvm.extractvalue %27[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %251 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %252 = llvm.insertvalue %241, %251[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %253 = llvm.insertvalue %242, %252[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %254 = llvm.mlir.constant(0 : index) : i64
    %255 = llvm.insertvalue %254, %253[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %256 = llvm.insertvalue %37, %255[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %257 = llvm.mlir.constant(1 : index) : i64
    %258 = llvm.insertvalue %257, %256[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %259 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %260 = llvm.insertvalue %79, %259[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %261 = llvm.insertvalue %86, %260[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %262 = llvm.mlir.constant(0 : index) : i64
    %263 = llvm.insertvalue %262, %261[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %264 = llvm.insertvalue %37, %263[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %265 = llvm.mlir.constant(1 : index) : i64
    %266 = llvm.insertvalue %265, %264[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb13(%34 : i64)
  ^bb13(%267: i64):  // 2 preds: ^bb12, ^bb17
    %268 = llvm.icmp "slt" %267, %37 : i64
    llvm.cond_br %268, ^bb14, ^bb18
  ^bb14:  // pred: ^bb13
    llvm.br ^bb15(%34 : i64)
  ^bb15(%269: i64):  // 2 preds: ^bb14, ^bb16
    %270 = llvm.icmp "slt" %269, %37 : i64
    llvm.cond_br %270, ^bb16, ^bb17
  ^bb16:  // pred: ^bb15
    %271 = llvm.getelementptr %220[%234] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %272 = llvm.mul %267, %229  : i64
    %273 = llvm.add %272, %269  : i64
    %274 = llvm.getelementptr %271[%273] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %275 = llvm.load %274 : !llvm.ptr -> f64
    %276 = llvm.getelementptr %242[%269] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %277 = llvm.load %276 : !llvm.ptr -> f64
    %278 = llvm.getelementptr %86[%267] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %279 = llvm.load %278 : !llvm.ptr -> f64
    %280 = llvm.fmul %275, %277  : f64
    %281 = llvm.fadd %280, %279  : f64
    %282 = llvm.getelementptr %86[%267] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %281, %282 : f64, !llvm.ptr
    %283 = llvm.add %269, %35  : i64
    llvm.br ^bb15(%283 : i64)
  ^bb17:  // pred: ^bb15
    %284 = llvm.add %267, %35  : i64
    llvm.br ^bb13(%284 : i64)
  ^bb18:  // pred: ^bb13
    %285 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %286 = llvm.insertvalue %79, %285[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %287 = llvm.insertvalue %86, %286[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %288 = llvm.mlir.constant(0 : index) : i64
    %289 = llvm.insertvalue %288, %287[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %290 = llvm.insertvalue %37, %289[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %291 = llvm.mlir.constant(1 : index) : i64
    %292 = llvm.insertvalue %291, %290[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %293 = llvm.mlir.constant(1 : index) : i64
    %294 = llvm.mul %37, %293  : i64
    %295 = llvm.mlir.zero : !llvm.ptr
    %296 = llvm.getelementptr %295[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %297 = llvm.ptrtoint %296 : !llvm.ptr to i64
    %298 = llvm.mul %294, %297  : i64
    %299 = llvm.getelementptr %86[%262] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %300 = llvm.getelementptr %86[%288] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%300, %299, %298) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb19(%34 : i64)
  ^bb19(%301: i64):  // 2 preds: ^bb18, ^bb20
    %302 = llvm.icmp "slt" %301, %42 : i64
    llvm.cond_br %302, ^bb20, ^bb21
  ^bb20:  // pred: ^bb19
    %303 = llvm.getelementptr %56[%301] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %304 = llvm.load %303 : !llvm.ptr -> f64
    %305 = llvm.getelementptr %86[%301] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %306 = llvm.load %305 : !llvm.ptr -> f64
    %307 = llvm.fmul %arg1, %304  : f64
    %308 = llvm.fmul %arg2, %306  : f64
    %309 = llvm.fadd %307, %308  : f64
    %310 = llvm.getelementptr %86[%301] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %309, %310 : f64, !llvm.ptr
    %311 = llvm.add %301, %35  : i64
    llvm.br ^bb19(%311 : i64)
  ^bb21:  // pred: ^bb19
    %312 = llvm.mlir.constant(1 : index) : i64
    %313 = llvm.mul %72, %312  : i64
    %314 = llvm.mlir.zero : !llvm.ptr
    %315 = llvm.getelementptr %314[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %316 = llvm.ptrtoint %315 : !llvm.ptr to i64
    %317 = llvm.mul %313, %316  : i64
    %318 = llvm.getelementptr %86[%90] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %319 = llvm.extractvalue %33[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %320 = llvm.extractvalue %33[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %321 = llvm.getelementptr %319[%320] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%321, %318, %317) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.return
  }
}

