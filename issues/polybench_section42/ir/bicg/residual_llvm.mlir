module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @kernel_bicg(%arg0: i32, %arg1: i32, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: !llvm.ptr, %arg10: !llvm.ptr, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: !llvm.ptr, %arg15: !llvm.ptr, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: !llvm.ptr, %arg20: !llvm.ptr, %arg21: i64, %arg22: i64, %arg23: i64, %arg24: !llvm.ptr, %arg25: !llvm.ptr, %arg26: i64, %arg27: i64, %arg28: i64) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg2, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.insertvalue %arg3, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.insertvalue %arg4, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.insertvalue %arg5, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.insertvalue %arg7, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.insertvalue %arg6, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.insertvalue %arg8, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.insertvalue %arg9, %8[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.insertvalue %arg10, %9[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.insertvalue %arg11, %10[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.insertvalue %arg12, %11[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = llvm.insertvalue %arg13, %12[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %15 = llvm.insertvalue %arg14, %14[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %16 = llvm.insertvalue %arg15, %15[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %17 = llvm.insertvalue %arg16, %16[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %18 = llvm.insertvalue %arg17, %17[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %19 = llvm.insertvalue %arg18, %18[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %20 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %21 = llvm.insertvalue %arg19, %20[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %22 = llvm.insertvalue %arg20, %21[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %23 = llvm.insertvalue %arg21, %22[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %24 = llvm.insertvalue %arg22, %23[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %25 = llvm.insertvalue %arg23, %24[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %26 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %27 = llvm.insertvalue %arg24, %26[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %28 = llvm.insertvalue %arg25, %27[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %29 = llvm.insertvalue %arg26, %28[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %30 = llvm.insertvalue %arg27, %29[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %31 = llvm.insertvalue %arg28, %30[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %32 = llvm.mlir.constant(0 : index) : i64
    %33 = llvm.mlir.constant(1 : index) : i64
    %34 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %35 = llvm.sext %arg0 : i32 to i64
    %36 = llvm.mlir.constant(1 : index) : i64
    %37 = llvm.extractvalue %13[3] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %38 = llvm.alloca %36 x !llvm.array<1 x i64> : (i64) -> !llvm.ptr
    llvm.store %37, %38 : !llvm.array<1 x i64>, !llvm.ptr
    %39 = llvm.getelementptr %38[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x i64>
    %40 = llvm.load %39 : !llvm.ptr -> i64
    %41 = llvm.mlir.constant(1 : index) : i64
    %42 = llvm.mlir.zero : !llvm.ptr
    %43 = llvm.getelementptr %42[%40] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %44 = llvm.ptrtoint %43 : !llvm.ptr to i64
    %45 = llvm.mlir.constant(64 : index) : i64
    %46 = llvm.add %44, %45  : i64
    %47 = llvm.call @malloc(%46) : (i64) -> !llvm.ptr
    %48 = llvm.ptrtoint %47 : !llvm.ptr to i64
    %49 = llvm.mlir.constant(1 : index) : i64
    %50 = llvm.sub %45, %49  : i64
    %51 = llvm.add %48, %50  : i64
    %52 = llvm.urem %51, %45  : i64
    %53 = llvm.sub %51, %52  : i64
    %54 = llvm.inttoptr %53 : i64 to !llvm.ptr
    %55 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %56 = llvm.insertvalue %47, %55[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %57 = llvm.insertvalue %54, %56[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %58 = llvm.mlir.constant(0 : index) : i64
    %59 = llvm.insertvalue %58, %57[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %60 = llvm.insertvalue %40, %59[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %61 = llvm.insertvalue %41, %60[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb1(%32 : i64)
  ^bb1(%62: i64):  // 2 preds: ^bb0, ^bb2
    %63 = llvm.icmp "slt" %62, %40 : i64
    llvm.cond_br %63, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %64 = llvm.getelementptr %54[%62] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %34, %64 : f64, !llvm.ptr
    %65 = llvm.add %62, %33  : i64
    llvm.br ^bb1(%65 : i64)
  ^bb3:  // pred: ^bb1
    %66 = llvm.sext %arg1 : i32 to i64
    %67 = llvm.mlir.constant(1 : index) : i64
    %68 = llvm.extractvalue %19[3] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %69 = llvm.alloca %67 x !llvm.array<1 x i64> : (i64) -> !llvm.ptr
    llvm.store %68, %69 : !llvm.array<1 x i64>, !llvm.ptr
    %70 = llvm.getelementptr %69[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x i64>
    %71 = llvm.load %70 : !llvm.ptr -> i64
    %72 = llvm.mlir.constant(1 : index) : i64
    %73 = llvm.mlir.zero : !llvm.ptr
    %74 = llvm.getelementptr %73[%71] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %75 = llvm.ptrtoint %74 : !llvm.ptr to i64
    %76 = llvm.mlir.constant(64 : index) : i64
    %77 = llvm.add %75, %76  : i64
    %78 = llvm.call @malloc(%77) : (i64) -> !llvm.ptr
    %79 = llvm.ptrtoint %78 : !llvm.ptr to i64
    %80 = llvm.mlir.constant(1 : index) : i64
    %81 = llvm.sub %76, %80  : i64
    %82 = llvm.add %79, %81  : i64
    %83 = llvm.urem %82, %76  : i64
    %84 = llvm.sub %82, %83  : i64
    %85 = llvm.inttoptr %84 : i64 to !llvm.ptr
    %86 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %87 = llvm.insertvalue %78, %86[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %88 = llvm.insertvalue %85, %87[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %89 = llvm.mlir.constant(0 : index) : i64
    %90 = llvm.insertvalue %89, %88[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %91 = llvm.insertvalue %71, %90[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %92 = llvm.insertvalue %72, %91[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb4(%32 : i64)
  ^bb4(%93: i64):  // 2 preds: ^bb3, ^bb5
    %94 = llvm.icmp "slt" %93, %71 : i64
    llvm.cond_br %94, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %95 = llvm.getelementptr %85[%93] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %34, %95 : f64, !llvm.ptr
    %96 = llvm.add %93, %33  : i64
    llvm.br ^bb4(%96 : i64)
  ^bb6:  // pred: ^bb4
    %97 = llvm.extractvalue %31[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %98 = llvm.extractvalue %31[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %99 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %100 = llvm.insertvalue %97, %99[0] : !llvm.struct<(ptr, ptr, i64)> 
    %101 = llvm.insertvalue %98, %100[1] : !llvm.struct<(ptr, ptr, i64)> 
    %102 = llvm.mlir.constant(0 : index) : i64
    %103 = llvm.insertvalue %102, %101[2] : !llvm.struct<(ptr, ptr, i64)> 
    %104 = llvm.extractvalue %31[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %105 = llvm.extractvalue %31[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %106 = llvm.extractvalue %31[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %107 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %108 = llvm.insertvalue %97, %107[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %109 = llvm.insertvalue %98, %108[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %110 = llvm.mlir.constant(0 : index) : i64
    %111 = llvm.insertvalue %110, %109[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %112 = llvm.insertvalue %66, %111[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %113 = llvm.mlir.constant(1 : index) : i64
    %114 = llvm.insertvalue %113, %112[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %115 = llvm.extractvalue %7[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %116 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %117 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %118 = llvm.insertvalue %115, %117[0] : !llvm.struct<(ptr, ptr, i64)> 
    %119 = llvm.insertvalue %116, %118[1] : !llvm.struct<(ptr, ptr, i64)> 
    %120 = llvm.mlir.constant(0 : index) : i64
    %121 = llvm.insertvalue %120, %119[2] : !llvm.struct<(ptr, ptr, i64)> 
    %122 = llvm.extractvalue %7[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %123 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %124 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %125 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %126 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %127 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %128 = llvm.insertvalue %115, %127[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %129 = llvm.insertvalue %116, %128[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %130 = llvm.mlir.constant(0 : index) : i64
    %131 = llvm.insertvalue %130, %129[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %132 = llvm.insertvalue %66, %131[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %133 = llvm.insertvalue %125, %132[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %134 = llvm.insertvalue %35, %133[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %135 = llvm.mlir.constant(1 : index) : i64
    %136 = llvm.insertvalue %135, %134[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %137 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %138 = llvm.insertvalue %47, %137[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %139 = llvm.insertvalue %54, %138[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %140 = llvm.mlir.constant(0 : index) : i64
    %141 = llvm.insertvalue %140, %139[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %142 = llvm.insertvalue %35, %141[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %143 = llvm.mlir.constant(1 : index) : i64
    %144 = llvm.insertvalue %143, %142[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb7(%32 : i64)
  ^bb7(%145: i64):  // 2 preds: ^bb6, ^bb11
    %146 = llvm.icmp "slt" %145, %66 : i64
    llvm.cond_br %146, ^bb8, ^bb12
  ^bb8:  // pred: ^bb7
    llvm.br ^bb9(%32 : i64)
  ^bb9(%147: i64):  // 2 preds: ^bb8, ^bb10
    %148 = llvm.icmp "slt" %147, %35 : i64
    llvm.cond_br %148, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    %149 = llvm.getelementptr %98[%145] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %150 = llvm.load %149 : !llvm.ptr -> f64
    %151 = llvm.getelementptr %116[%130] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %152 = llvm.mul %145, %125  : i64
    %153 = llvm.add %152, %147  : i64
    %154 = llvm.getelementptr %151[%153] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %155 = llvm.load %154 : !llvm.ptr -> f64
    %156 = llvm.getelementptr %54[%147] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %157 = llvm.load %156 : !llvm.ptr -> f64
    %158 = llvm.fmul %150, %155  : f64
    %159 = llvm.fadd %157, %158  : f64
    %160 = llvm.getelementptr %54[%147] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %159, %160 : f64, !llvm.ptr
    %161 = llvm.add %147, %33  : i64
    llvm.br ^bb9(%161 : i64)
  ^bb11:  // pred: ^bb9
    %162 = llvm.add %145, %33  : i64
    llvm.br ^bb7(%162 : i64)
  ^bb12:  // pred: ^bb7
    %163 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %164 = llvm.insertvalue %47, %163[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %165 = llvm.insertvalue %54, %164[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %166 = llvm.mlir.constant(0 : index) : i64
    %167 = llvm.insertvalue %166, %165[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %168 = llvm.insertvalue %35, %167[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %169 = llvm.mlir.constant(1 : index) : i64
    %170 = llvm.insertvalue %169, %168[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %171 = llvm.mlir.constant(1 : index) : i64
    %172 = llvm.mul %35, %171  : i64
    %173 = llvm.mlir.zero : !llvm.ptr
    %174 = llvm.getelementptr %173[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %175 = llvm.ptrtoint %174 : !llvm.ptr to i64
    %176 = llvm.mul %172, %175  : i64
    %177 = llvm.getelementptr %54[%140] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %178 = llvm.getelementptr %54[%166] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%178, %177, %176) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %179 = llvm.mlir.constant(1 : index) : i64
    %180 = llvm.mul %40, %179  : i64
    %181 = llvm.mlir.zero : !llvm.ptr
    %182 = llvm.getelementptr %181[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %183 = llvm.ptrtoint %182 : !llvm.ptr to i64
    %184 = llvm.mul %180, %183  : i64
    %185 = llvm.getelementptr %54[%58] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %186 = llvm.extractvalue %13[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %187 = llvm.extractvalue %13[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %188 = llvm.getelementptr %186[%187] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%188, %185, %184) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %189 = llvm.extractvalue %7[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %190 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %191 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %192 = llvm.insertvalue %189, %191[0] : !llvm.struct<(ptr, ptr, i64)> 
    %193 = llvm.insertvalue %190, %192[1] : !llvm.struct<(ptr, ptr, i64)> 
    %194 = llvm.mlir.constant(0 : index) : i64
    %195 = llvm.insertvalue %194, %193[2] : !llvm.struct<(ptr, ptr, i64)> 
    %196 = llvm.extractvalue %7[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %197 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %198 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %199 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %200 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %201 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %202 = llvm.insertvalue %189, %201[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %203 = llvm.insertvalue %190, %202[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %204 = llvm.mlir.constant(0 : index) : i64
    %205 = llvm.insertvalue %204, %203[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %206 = llvm.insertvalue %66, %205[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %207 = llvm.insertvalue %199, %206[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %208 = llvm.insertvalue %35, %207[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %209 = llvm.mlir.constant(1 : index) : i64
    %210 = llvm.insertvalue %209, %208[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %211 = llvm.extractvalue %25[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %212 = llvm.extractvalue %25[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %213 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %214 = llvm.insertvalue %211, %213[0] : !llvm.struct<(ptr, ptr, i64)> 
    %215 = llvm.insertvalue %212, %214[1] : !llvm.struct<(ptr, ptr, i64)> 
    %216 = llvm.mlir.constant(0 : index) : i64
    %217 = llvm.insertvalue %216, %215[2] : !llvm.struct<(ptr, ptr, i64)> 
    %218 = llvm.extractvalue %25[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %219 = llvm.extractvalue %25[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %220 = llvm.extractvalue %25[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %221 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %222 = llvm.insertvalue %211, %221[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %223 = llvm.insertvalue %212, %222[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %224 = llvm.mlir.constant(0 : index) : i64
    %225 = llvm.insertvalue %224, %223[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %226 = llvm.insertvalue %35, %225[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %227 = llvm.mlir.constant(1 : index) : i64
    %228 = llvm.insertvalue %227, %226[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %229 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %230 = llvm.insertvalue %78, %229[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %231 = llvm.insertvalue %85, %230[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %232 = llvm.mlir.constant(0 : index) : i64
    %233 = llvm.insertvalue %232, %231[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %234 = llvm.insertvalue %66, %233[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %235 = llvm.mlir.constant(1 : index) : i64
    %236 = llvm.insertvalue %235, %234[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb13(%32 : i64)
  ^bb13(%237: i64):  // 2 preds: ^bb12, ^bb17
    %238 = llvm.icmp "slt" %237, %66 : i64
    llvm.cond_br %238, ^bb14, ^bb18
  ^bb14:  // pred: ^bb13
    llvm.br ^bb15(%32 : i64)
  ^bb15(%239: i64):  // 2 preds: ^bb14, ^bb16
    %240 = llvm.icmp "slt" %239, %35 : i64
    llvm.cond_br %240, ^bb16, ^bb17
  ^bb16:  // pred: ^bb15
    %241 = llvm.getelementptr %190[%204] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %242 = llvm.mul %237, %199  : i64
    %243 = llvm.add %242, %239  : i64
    %244 = llvm.getelementptr %241[%243] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %245 = llvm.load %244 : !llvm.ptr -> f64
    %246 = llvm.getelementptr %212[%239] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %247 = llvm.load %246 : !llvm.ptr -> f64
    %248 = llvm.getelementptr %85[%237] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %249 = llvm.load %248 : !llvm.ptr -> f64
    %250 = llvm.fmul %245, %247  : f64
    %251 = llvm.fadd %249, %250  : f64
    %252 = llvm.getelementptr %85[%237] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %251, %252 : f64, !llvm.ptr
    %253 = llvm.add %239, %33  : i64
    llvm.br ^bb15(%253 : i64)
  ^bb17:  // pred: ^bb15
    %254 = llvm.add %237, %33  : i64
    llvm.br ^bb13(%254 : i64)
  ^bb18:  // pred: ^bb13
    %255 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %256 = llvm.insertvalue %78, %255[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %257 = llvm.insertvalue %85, %256[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %258 = llvm.mlir.constant(0 : index) : i64
    %259 = llvm.insertvalue %258, %257[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %260 = llvm.insertvalue %66, %259[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %261 = llvm.mlir.constant(1 : index) : i64
    %262 = llvm.insertvalue %261, %260[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %263 = llvm.mlir.constant(1 : index) : i64
    %264 = llvm.mul %66, %263  : i64
    %265 = llvm.mlir.zero : !llvm.ptr
    %266 = llvm.getelementptr %265[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %267 = llvm.ptrtoint %266 : !llvm.ptr to i64
    %268 = llvm.mul %264, %267  : i64
    %269 = llvm.getelementptr %85[%232] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %270 = llvm.getelementptr %85[%258] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%270, %269, %268) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %271 = llvm.mlir.constant(1 : index) : i64
    %272 = llvm.mul %71, %271  : i64
    %273 = llvm.mlir.zero : !llvm.ptr
    %274 = llvm.getelementptr %273[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %275 = llvm.ptrtoint %274 : !llvm.ptr to i64
    %276 = llvm.mul %272, %275  : i64
    %277 = llvm.getelementptr %85[%89] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %278 = llvm.extractvalue %19[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %279 = llvm.extractvalue %19[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %280 = llvm.getelementptr %278[%279] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%280, %277, %276) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.return
  }
}

