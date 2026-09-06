module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.func @memrefCopy(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @kernel_deriche(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: !llvm.ptr, %arg11: !llvm.ptr, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: !llvm.ptr, %arg18: !llvm.ptr, %arg19: i64, %arg20: i64, %arg21: i64, %arg22: i64, %arg23: i64, %arg24: !llvm.ptr, %arg25: !llvm.ptr, %arg26: i64, %arg27: i64, %arg28: i64, %arg29: i64, %arg30: i64) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg3, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.insertvalue %arg4, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.insertvalue %arg5, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.insertvalue %arg6, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.insertvalue %arg8, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.insertvalue %arg7, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.insertvalue %arg9, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.insertvalue %arg17, %8[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %10 = llvm.insertvalue %arg18, %9[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.insertvalue %arg19, %10[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %12 = llvm.insertvalue %arg20, %11[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.insertvalue %arg22, %12[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.insertvalue %arg21, %13[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.insertvalue %arg23, %14[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.mlir.constant(0 : index) : i64
    %17 = llvm.mlir.constant(1 : index) : i64
    %18 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %19 = llvm.mlir.constant(-2.000000e+00 : f64) : f64
    %20 = llvm.mlir.constant(2.000000e+00 : f64) : f64
    %21 = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %22 = llvm.sext %arg1 : i32 to i64
    %23 = llvm.fneg %arg2  : f64
    %24 = llvm.intr.exp(%23)  : (f64) -> f64
    %25 = llvm.fsub %21, %24  : f64
    %26 = llvm.fmul %25, %25  : f64
    %27 = llvm.fmul %arg2, %20  : f64
    %28 = llvm.fmul %27, %24  : f64
    %29 = llvm.fadd %28, %21  : f64
    %30 = llvm.intr.exp(%27)  : (f64) -> f64
    %31 = llvm.fsub %29, %30  : f64
    %32 = llvm.fdiv %26, %31  : f64
    %33 = llvm.fmul %32, %24  : f64
    %34 = llvm.fsub %arg2, %21  : f64
    %35 = llvm.fmul %33, %34  : f64
    %36 = llvm.intr.pow(%20, %23)  : (f64, f64) -> f64
    %37 = llvm.fmul %arg2, %19  : f64
    %38 = llvm.intr.exp(%37)  : (f64) -> f64
    %39 = llvm.fneg %38  : f64
    %40 = llvm.sext %arg0 : i32 to i64
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
    %62 = llvm.mlir.constant(1 : index) : i64
    %63 = llvm.mlir.zero : !llvm.ptr
    %64 = llvm.getelementptr %63[%40] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %65 = llvm.ptrtoint %64 : !llvm.ptr to i64
    %66 = llvm.mlir.constant(64 : index) : i64
    %67 = llvm.add %65, %66  : i64
    %68 = llvm.call @malloc(%67) : (i64) -> !llvm.ptr
    %69 = llvm.ptrtoint %68 : !llvm.ptr to i64
    %70 = llvm.mlir.constant(1 : index) : i64
    %71 = llvm.sub %66, %70  : i64
    %72 = llvm.add %69, %71  : i64
    %73 = llvm.urem %72, %66  : i64
    %74 = llvm.sub %72, %73  : i64
    %75 = llvm.inttoptr %74 : i64 to !llvm.ptr
    %76 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %77 = llvm.insertvalue %68, %76[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %78 = llvm.insertvalue %75, %77[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %79 = llvm.mlir.constant(0 : index) : i64
    %80 = llvm.insertvalue %79, %78[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %81 = llvm.insertvalue %40, %80[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %82 = llvm.insertvalue %62, %81[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %83 = llvm.mlir.constant(1 : index) : i64
    %84 = llvm.mlir.zero : !llvm.ptr
    %85 = llvm.getelementptr %84[%40] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %86 = llvm.ptrtoint %85 : !llvm.ptr to i64
    %87 = llvm.mlir.constant(64 : index) : i64
    %88 = llvm.add %86, %87  : i64
    %89 = llvm.call @malloc(%88) : (i64) -> !llvm.ptr
    %90 = llvm.ptrtoint %89 : !llvm.ptr to i64
    %91 = llvm.mlir.constant(1 : index) : i64
    %92 = llvm.sub %87, %91  : i64
    %93 = llvm.add %90, %92  : i64
    %94 = llvm.urem %93, %87  : i64
    %95 = llvm.sub %93, %94  : i64
    %96 = llvm.inttoptr %95 : i64 to !llvm.ptr
    %97 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %98 = llvm.insertvalue %89, %97[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %99 = llvm.insertvalue %96, %98[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %100 = llvm.mlir.constant(0 : index) : i64
    %101 = llvm.insertvalue %100, %99[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %102 = llvm.insertvalue %40, %101[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %103 = llvm.insertvalue %83, %102[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb1(%16 : i64)
  ^bb1(%104: i64):  // 2 preds: ^bb0, ^bb2
    %105 = llvm.icmp "slt" %104, %40 : i64
    llvm.cond_br %105, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %106 = llvm.getelementptr %54[%104] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %18, %106 : f64, !llvm.ptr
    %107 = llvm.add %104, %17  : i64
    llvm.br ^bb1(%107 : i64)
  ^bb3:  // pred: ^bb1
    llvm.br ^bb4(%16 : i64)
  ^bb4(%108: i64):  // 2 preds: ^bb3, ^bb5
    %109 = llvm.icmp "slt" %108, %40 : i64
    llvm.cond_br %109, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %110 = llvm.getelementptr %75[%108] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %18, %110 : f64, !llvm.ptr
    %111 = llvm.add %108, %17  : i64
    llvm.br ^bb4(%111 : i64)
  ^bb6:  // pred: ^bb4
    llvm.br ^bb7(%16 : i64)
  ^bb7(%112: i64):  // 2 preds: ^bb6, ^bb8
    %113 = llvm.icmp "slt" %112, %40 : i64
    llvm.cond_br %113, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %114 = llvm.getelementptr %96[%112] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %18, %114 : f64, !llvm.ptr
    %115 = llvm.add %112, %17  : i64
    llvm.br ^bb7(%115 : i64)
  ^bb9:  // pred: ^bb7
    %116 = llvm.extractvalue %7[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %117 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %118 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %119 = llvm.insertvalue %116, %118[0] : !llvm.struct<(ptr, ptr, i64)> 
    %120 = llvm.insertvalue %117, %119[1] : !llvm.struct<(ptr, ptr, i64)> 
    %121 = llvm.mlir.constant(0 : index) : i64
    %122 = llvm.insertvalue %121, %120[2] : !llvm.struct<(ptr, ptr, i64)> 
    %123 = llvm.extractvalue %7[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %124 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %125 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %126 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %127 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %128 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %129 = llvm.insertvalue %116, %128[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %130 = llvm.insertvalue %117, %129[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %131 = llvm.mlir.constant(0 : index) : i64
    %132 = llvm.insertvalue %131, %130[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %133 = llvm.insertvalue %40, %132[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %134 = llvm.insertvalue %126, %133[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %135 = llvm.insertvalue %22, %134[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %136 = llvm.mlir.constant(1 : index) : i64
    %137 = llvm.insertvalue %136, %135[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %138 = llvm.extractvalue %7[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %139 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %140 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %141 = llvm.insertvalue %138, %140[0] : !llvm.struct<(ptr, ptr, i64)> 
    %142 = llvm.insertvalue %139, %141[1] : !llvm.struct<(ptr, ptr, i64)> 
    %143 = llvm.mlir.constant(0 : index) : i64
    %144 = llvm.insertvalue %143, %142[2] : !llvm.struct<(ptr, ptr, i64)> 
    %145 = llvm.extractvalue %7[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %146 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %147 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %148 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %149 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %150 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %151 = llvm.insertvalue %138, %150[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %152 = llvm.insertvalue %139, %151[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %153 = llvm.mlir.constant(0 : index) : i64
    %154 = llvm.insertvalue %153, %152[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %155 = llvm.insertvalue %40, %154[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %156 = llvm.insertvalue %148, %155[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %157 = llvm.insertvalue %22, %156[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %158 = llvm.mlir.constant(1 : index) : i64
    %159 = llvm.insertvalue %158, %157[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %160 = llvm.mlir.constant(1 : index) : i64
    %161 = llvm.mul %22, %40  : i64
    %162 = llvm.mlir.zero : !llvm.ptr
    %163 = llvm.getelementptr %162[%161] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %164 = llvm.ptrtoint %163 : !llvm.ptr to i64
    %165 = llvm.mlir.constant(64 : index) : i64
    %166 = llvm.add %164, %165  : i64
    %167 = llvm.call @malloc(%166) : (i64) -> !llvm.ptr
    %168 = llvm.ptrtoint %167 : !llvm.ptr to i64
    %169 = llvm.mlir.constant(1 : index) : i64
    %170 = llvm.sub %165, %169  : i64
    %171 = llvm.add %168, %170  : i64
    %172 = llvm.urem %171, %165  : i64
    %173 = llvm.sub %171, %172  : i64
    %174 = llvm.inttoptr %173 : i64 to !llvm.ptr
    %175 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %176 = llvm.insertvalue %167, %175[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %177 = llvm.insertvalue %174, %176[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %178 = llvm.mlir.constant(0 : index) : i64
    %179 = llvm.insertvalue %178, %177[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %180 = llvm.insertvalue %40, %179[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %181 = llvm.insertvalue %22, %180[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %182 = llvm.insertvalue %22, %181[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %183 = llvm.insertvalue %160, %182[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %184 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %185 = llvm.insertvalue %89, %184[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %186 = llvm.insertvalue %96, %185[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %187 = llvm.mlir.constant(0 : index) : i64
    %188 = llvm.insertvalue %187, %186[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %189 = llvm.insertvalue %40, %188[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %190 = llvm.mlir.constant(1 : index) : i64
    %191 = llvm.insertvalue %190, %189[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %192 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %193 = llvm.insertvalue %68, %192[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %194 = llvm.insertvalue %75, %193[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %195 = llvm.mlir.constant(0 : index) : i64
    %196 = llvm.insertvalue %195, %194[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %197 = llvm.insertvalue %40, %196[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %198 = llvm.mlir.constant(1 : index) : i64
    %199 = llvm.insertvalue %198, %197[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %200 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %201 = llvm.insertvalue %47, %200[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %202 = llvm.insertvalue %54, %201[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %203 = llvm.mlir.constant(0 : index) : i64
    %204 = llvm.insertvalue %203, %202[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %205 = llvm.insertvalue %40, %204[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %206 = llvm.mlir.constant(1 : index) : i64
    %207 = llvm.insertvalue %206, %205[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb10(%16 : i64)
  ^bb10(%208: i64):  // 2 preds: ^bb9, ^bb14
    %209 = llvm.icmp "slt" %208, %40 : i64
    llvm.cond_br %209, ^bb11, ^bb15
  ^bb11:  // pred: ^bb10
    llvm.br ^bb12(%16 : i64)
  ^bb12(%210: i64):  // 2 preds: ^bb11, ^bb13
    %211 = llvm.icmp "slt" %210, %22 : i64
    llvm.cond_br %211, ^bb13, ^bb14
  ^bb13:  // pred: ^bb12
    %212 = llvm.getelementptr %117[%131] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %213 = llvm.mul %208, %126  : i64
    %214 = llvm.add %213, %210  : i64
    %215 = llvm.getelementptr %212[%214] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %216 = llvm.load %215 : !llvm.ptr -> f64
    %217 = llvm.getelementptr %139[%153] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %218 = llvm.mul %208, %148  : i64
    %219 = llvm.add %218, %210  : i64
    %220 = llvm.getelementptr %217[%219] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %221 = llvm.load %220 : !llvm.ptr -> f64
    %222 = llvm.getelementptr %96[%208] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %223 = llvm.load %222 : !llvm.ptr -> f64
    %224 = llvm.getelementptr %75[%208] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %225 = llvm.load %224 : !llvm.ptr -> f64
    %226 = llvm.getelementptr %54[%208] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %227 = llvm.load %226 : !llvm.ptr -> f64
    %228 = llvm.fmul %32, %216  : f64
    %229 = llvm.fmul %35, %223  : f64
    %230 = llvm.fadd %228, %229  : f64
    %231 = llvm.fmul %36, %227  : f64
    %232 = llvm.fadd %230, %231  : f64
    %233 = llvm.fmul %39, %225  : f64
    %234 = llvm.fadd %232, %233  : f64
    %235 = llvm.mul %208, %22  : i64
    %236 = llvm.add %235, %210  : i64
    %237 = llvm.getelementptr %174[%236] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %234, %237 : f64, !llvm.ptr
    %238 = llvm.getelementptr %96[%208] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %221, %238 : f64, !llvm.ptr
    %239 = llvm.getelementptr %75[%208] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %227, %239 : f64, !llvm.ptr
    %240 = llvm.getelementptr %54[%208] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %234, %240 : f64, !llvm.ptr
    %241 = llvm.add %210, %17  : i64
    llvm.br ^bb12(%241 : i64)
  ^bb14:  // pred: ^bb12
    %242 = llvm.add %208, %17  : i64
    llvm.br ^bb10(%242 : i64)
  ^bb15:  // pred: ^bb10
    %243 = llvm.mlir.constant(1 : index) : i64
    %244 = llvm.extractvalue %15[3] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %245 = llvm.alloca %243 x !llvm.array<2 x i64> : (i64) -> !llvm.ptr
    llvm.store %244, %245 : !llvm.array<2 x i64>, !llvm.ptr
    %246 = llvm.getelementptr %245[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
    %247 = llvm.load %246 : !llvm.ptr -> i64
    %248 = llvm.mlir.constant(1 : index) : i64
    %249 = llvm.extractvalue %15[3] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %250 = llvm.alloca %248 x !llvm.array<2 x i64> : (i64) -> !llvm.ptr
    llvm.store %249, %250 : !llvm.array<2 x i64>, !llvm.ptr
    %251 = llvm.getelementptr %250[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
    %252 = llvm.load %251 : !llvm.ptr -> i64
    %253 = llvm.mlir.constant(1 : index) : i64
    %254 = llvm.mul %252, %247  : i64
    %255 = llvm.mlir.zero : !llvm.ptr
    %256 = llvm.getelementptr %255[%254] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %257 = llvm.ptrtoint %256 : !llvm.ptr to i64
    %258 = llvm.mlir.constant(64 : index) : i64
    %259 = llvm.add %257, %258  : i64
    %260 = llvm.call @malloc(%259) : (i64) -> !llvm.ptr
    %261 = llvm.ptrtoint %260 : !llvm.ptr to i64
    %262 = llvm.mlir.constant(1 : index) : i64
    %263 = llvm.sub %258, %262  : i64
    %264 = llvm.add %261, %263  : i64
    %265 = llvm.urem %264, %258  : i64
    %266 = llvm.sub %264, %265  : i64
    %267 = llvm.inttoptr %266 : i64 to !llvm.ptr
    %268 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %269 = llvm.insertvalue %260, %268[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %270 = llvm.insertvalue %267, %269[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %271 = llvm.mlir.constant(0 : index) : i64
    %272 = llvm.insertvalue %271, %270[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %273 = llvm.insertvalue %247, %272[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %274 = llvm.insertvalue %252, %273[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %275 = llvm.insertvalue %252, %274[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %276 = llvm.insertvalue %253, %275[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %277 = llvm.mlir.constant(1 : index) : i64
    %278 = llvm.extractvalue %15[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %279 = llvm.mul %278, %277  : i64
    %280 = llvm.extractvalue %15[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %281 = llvm.mul %279, %280  : i64
    %282 = llvm.mlir.zero : !llvm.ptr
    %283 = llvm.getelementptr %282[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %284 = llvm.ptrtoint %283 : !llvm.ptr to i64
    %285 = llvm.mul %281, %284  : i64
    %286 = llvm.extractvalue %15[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %287 = llvm.extractvalue %15[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %288 = llvm.getelementptr %286[%287] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %289 = llvm.getelementptr %267[%271] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%289, %288, %285) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %290 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %291 = llvm.insertvalue %260, %290[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %292 = llvm.insertvalue %267, %291[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %293 = llvm.mlir.constant(0 : index) : i64
    %294 = llvm.insertvalue %293, %292[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %295 = llvm.insertvalue %40, %294[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %296 = llvm.insertvalue %252, %295[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %297 = llvm.insertvalue %22, %296[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %298 = llvm.mlir.constant(1 : index) : i64
    %299 = llvm.insertvalue %298, %297[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %300 = llvm.intr.stacksave : !llvm.ptr
    %301 = llvm.mlir.constant(2 : i64) : i64
    %302 = llvm.mlir.constant(1 : index) : i64
    %303 = llvm.alloca %302 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %183, %303 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %304 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %305 = llvm.insertvalue %301, %304[0] : !llvm.struct<(i64, ptr)> 
    %306 = llvm.insertvalue %303, %305[1] : !llvm.struct<(i64, ptr)> 
    %307 = llvm.mlir.constant(2 : i64) : i64
    %308 = llvm.mlir.constant(1 : index) : i64
    %309 = llvm.alloca %308 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %299, %309 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %310 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %311 = llvm.insertvalue %307, %310[0] : !llvm.struct<(i64, ptr)> 
    %312 = llvm.insertvalue %309, %311[1] : !llvm.struct<(i64, ptr)> 
    %313 = llvm.mlir.constant(1 : index) : i64
    %314 = llvm.alloca %313 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %306, %314 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %315 = llvm.alloca %313 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %312, %315 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %316 = llvm.mlir.zero : !llvm.ptr
    %317 = llvm.getelementptr %316[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %318 = llvm.ptrtoint %317 : !llvm.ptr to i64
    llvm.call @memrefCopy(%318, %314, %315) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %300 : !llvm.ptr
    %319 = llvm.mlir.constant(1 : index) : i64
    %320 = llvm.mul %247, %319  : i64
    %321 = llvm.mul %320, %252  : i64
    %322 = llvm.mlir.zero : !llvm.ptr
    %323 = llvm.getelementptr %322[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %324 = llvm.ptrtoint %323 : !llvm.ptr to i64
    %325 = llvm.mul %321, %324  : i64
    %326 = llvm.getelementptr %267[%271] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %327 = llvm.extractvalue %15[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %328 = llvm.extractvalue %15[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %329 = llvm.getelementptr %327[%328] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%329, %326, %325) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.return
  }
}

