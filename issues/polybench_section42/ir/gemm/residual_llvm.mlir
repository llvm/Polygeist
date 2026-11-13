module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.func @memrefCopy(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @kernel_gemm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: f64, %arg4: f64, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: !llvm.ptr, %arg13: !llvm.ptr, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: !llvm.ptr, %arg20: !llvm.ptr, %arg21: i64, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i64) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg5, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.insertvalue %arg6, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.insertvalue %arg7, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.insertvalue %arg8, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.insertvalue %arg10, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.insertvalue %arg9, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.insertvalue %arg11, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.insertvalue %arg12, %8[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %10 = llvm.insertvalue %arg13, %9[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.insertvalue %arg14, %10[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %12 = llvm.insertvalue %arg15, %11[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.insertvalue %arg17, %12[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.insertvalue %arg16, %13[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.insertvalue %arg18, %14[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %17 = llvm.insertvalue %arg19, %16[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.insertvalue %arg20, %17[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %19 = llvm.insertvalue %arg21, %18[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.insertvalue %arg22, %19[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.insertvalue %arg24, %20[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.insertvalue %arg23, %21[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.insertvalue %arg25, %22[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.mlir.constant(0 : index) : i64
    %25 = llvm.mlir.constant(1 : index) : i64
    %26 = llvm.sext %arg1 : i32 to i64
    %27 = llvm.sext %arg2 : i32 to i64
    %28 = llvm.sext %arg0 : i32 to i64
    %29 = llvm.mlir.constant(1 : index) : i64
    %30 = llvm.extractvalue %7[3] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %31 = llvm.alloca %29 x !llvm.array<2 x i64> : (i64) -> !llvm.ptr
    llvm.store %30, %31 : !llvm.array<2 x i64>, !llvm.ptr
    %32 = llvm.getelementptr %31[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
    %33 = llvm.load %32 : !llvm.ptr -> i64
    %34 = llvm.mlir.constant(1 : index) : i64
    %35 = llvm.extractvalue %7[3] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %36 = llvm.alloca %34 x !llvm.array<2 x i64> : (i64) -> !llvm.ptr
    llvm.store %35, %36 : !llvm.array<2 x i64>, !llvm.ptr
    %37 = llvm.getelementptr %36[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
    %38 = llvm.load %37 : !llvm.ptr -> i64
    %39 = llvm.mlir.constant(1 : index) : i64
    %40 = llvm.mul %38, %33  : i64
    %41 = llvm.mlir.zero : !llvm.ptr
    %42 = llvm.getelementptr %41[%40] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %43 = llvm.ptrtoint %42 : !llvm.ptr to i64
    %44 = llvm.mlir.constant(64 : index) : i64
    %45 = llvm.add %43, %44  : i64
    %46 = llvm.call @malloc(%45) : (i64) -> !llvm.ptr
    %47 = llvm.ptrtoint %46 : !llvm.ptr to i64
    %48 = llvm.mlir.constant(1 : index) : i64
    %49 = llvm.sub %44, %48  : i64
    %50 = llvm.add %47, %49  : i64
    %51 = llvm.urem %50, %44  : i64
    %52 = llvm.sub %50, %51  : i64
    %53 = llvm.inttoptr %52 : i64 to !llvm.ptr
    %54 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %55 = llvm.insertvalue %46, %54[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %56 = llvm.insertvalue %53, %55[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %57 = llvm.mlir.constant(0 : index) : i64
    %58 = llvm.insertvalue %57, %56[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %59 = llvm.insertvalue %33, %58[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %60 = llvm.insertvalue %38, %59[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %61 = llvm.insertvalue %38, %60[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %62 = llvm.insertvalue %39, %61[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %63 = llvm.mlir.constant(1 : index) : i64
    %64 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %65 = llvm.mul %64, %63  : i64
    %66 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %67 = llvm.mul %65, %66  : i64
    %68 = llvm.mlir.zero : !llvm.ptr
    %69 = llvm.getelementptr %68[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %70 = llvm.ptrtoint %69 : !llvm.ptr to i64
    %71 = llvm.mul %67, %70  : i64
    %72 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %73 = llvm.extractvalue %7[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %74 = llvm.getelementptr %72[%73] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %75 = llvm.getelementptr %53[%57] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%75, %74, %71) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb1(%24 : i64)
  ^bb1(%76: i64):  // 2 preds: ^bb0, ^bb5
    %77 = llvm.icmp "slt" %76, %33 : i64
    llvm.cond_br %77, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%24 : i64)
  ^bb3(%78: i64):  // 2 preds: ^bb2, ^bb4
    %79 = llvm.icmp "slt" %78, %38 : i64
    llvm.cond_br %79, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %80 = llvm.mul %76, %38  : i64
    %81 = llvm.add %80, %78  : i64
    %82 = llvm.getelementptr %53[%81] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %83 = llvm.load %82 : !llvm.ptr -> f64
    %84 = llvm.fmul %83, %arg4  : f64
    %85 = llvm.mul %76, %38  : i64
    %86 = llvm.add %85, %78  : i64
    %87 = llvm.getelementptr %53[%86] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %84, %87 : f64, !llvm.ptr
    %88 = llvm.add %78, %25  : i64
    llvm.br ^bb3(%88 : i64)
  ^bb5:  // pred: ^bb3
    %89 = llvm.add %76, %25  : i64
    llvm.br ^bb1(%89 : i64)
  ^bb6:  // pred: ^bb1
    %90 = llvm.extractvalue %15[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %91 = llvm.extractvalue %15[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %92 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %93 = llvm.insertvalue %90, %92[0] : !llvm.struct<(ptr, ptr, i64)> 
    %94 = llvm.insertvalue %91, %93[1] : !llvm.struct<(ptr, ptr, i64)> 
    %95 = llvm.mlir.constant(0 : index) : i64
    %96 = llvm.insertvalue %95, %94[2] : !llvm.struct<(ptr, ptr, i64)> 
    %97 = llvm.extractvalue %15[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %98 = llvm.extractvalue %15[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %99 = llvm.extractvalue %15[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %100 = llvm.extractvalue %15[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %101 = llvm.extractvalue %15[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %102 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %103 = llvm.insertvalue %90, %102[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %104 = llvm.insertvalue %91, %103[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %105 = llvm.mlir.constant(0 : index) : i64
    %106 = llvm.insertvalue %105, %104[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %107 = llvm.insertvalue %28, %106[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %108 = llvm.insertvalue %100, %107[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %109 = llvm.insertvalue %27, %108[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %110 = llvm.mlir.constant(1 : index) : i64
    %111 = llvm.insertvalue %110, %109[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %112 = llvm.extractvalue %23[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %113 = llvm.extractvalue %23[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %114 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %115 = llvm.insertvalue %112, %114[0] : !llvm.struct<(ptr, ptr, i64)> 
    %116 = llvm.insertvalue %113, %115[1] : !llvm.struct<(ptr, ptr, i64)> 
    %117 = llvm.mlir.constant(0 : index) : i64
    %118 = llvm.insertvalue %117, %116[2] : !llvm.struct<(ptr, ptr, i64)> 
    %119 = llvm.extractvalue %23[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %120 = llvm.extractvalue %23[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %121 = llvm.extractvalue %23[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %122 = llvm.extractvalue %23[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %123 = llvm.extractvalue %23[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %124 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %125 = llvm.insertvalue %112, %124[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %126 = llvm.insertvalue %113, %125[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %127 = llvm.mlir.constant(0 : index) : i64
    %128 = llvm.insertvalue %127, %126[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %129 = llvm.insertvalue %27, %128[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %130 = llvm.insertvalue %122, %129[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %131 = llvm.insertvalue %26, %130[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %132 = llvm.mlir.constant(1 : index) : i64
    %133 = llvm.insertvalue %132, %131[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %134 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %135 = llvm.insertvalue %46, %134[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %136 = llvm.insertvalue %53, %135[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %137 = llvm.mlir.constant(0 : index) : i64
    %138 = llvm.insertvalue %137, %136[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %139 = llvm.insertvalue %28, %138[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %140 = llvm.insertvalue %38, %139[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %141 = llvm.insertvalue %26, %140[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %142 = llvm.mlir.constant(1 : index) : i64
    %143 = llvm.insertvalue %142, %141[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb7(%24 : i64)
  ^bb7(%144: i64):  // 2 preds: ^bb6, ^bb14
    %145 = llvm.icmp "slt" %144, %28 : i64
    llvm.cond_br %145, ^bb8, ^bb15
  ^bb8:  // pred: ^bb7
    llvm.br ^bb9(%24 : i64)
  ^bb9(%146: i64):  // 2 preds: ^bb8, ^bb13
    %147 = llvm.icmp "slt" %146, %27 : i64
    llvm.cond_br %147, ^bb10, ^bb14
  ^bb10:  // pred: ^bb9
    llvm.br ^bb11(%24 : i64)
  ^bb11(%148: i64):  // 2 preds: ^bb10, ^bb12
    %149 = llvm.icmp "slt" %148, %26 : i64
    llvm.cond_br %149, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    %150 = llvm.getelementptr %91[%105] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %151 = llvm.mul %144, %100  : i64
    %152 = llvm.add %151, %146  : i64
    %153 = llvm.getelementptr %150[%152] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %154 = llvm.load %153 : !llvm.ptr -> f64
    %155 = llvm.getelementptr %113[%127] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %156 = llvm.mul %146, %122  : i64
    %157 = llvm.add %156, %148  : i64
    %158 = llvm.getelementptr %155[%157] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %159 = llvm.load %158 : !llvm.ptr -> f64
    %160 = llvm.getelementptr %53[%137] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %161 = llvm.mul %144, %38  : i64
    %162 = llvm.add %161, %148  : i64
    %163 = llvm.getelementptr %160[%162] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %164 = llvm.load %163 : !llvm.ptr -> f64
    %165 = llvm.fmul %arg3, %154  : f64
    %166 = llvm.fmul %165, %159  : f64
    %167 = llvm.fadd %164, %166  : f64
    %168 = llvm.getelementptr %53[%137] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %169 = llvm.mul %144, %38  : i64
    %170 = llvm.add %169, %148  : i64
    %171 = llvm.getelementptr %168[%170] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %167, %171 : f64, !llvm.ptr
    %172 = llvm.add %148, %25  : i64
    llvm.br ^bb11(%172 : i64)
  ^bb13:  // pred: ^bb11
    %173 = llvm.add %146, %25  : i64
    llvm.br ^bb9(%173 : i64)
  ^bb14:  // pred: ^bb9
    %174 = llvm.add %144, %25  : i64
    llvm.br ^bb7(%174 : i64)
  ^bb15:  // pred: ^bb7
    %175 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %176 = llvm.insertvalue %46, %175[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %177 = llvm.insertvalue %53, %176[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %178 = llvm.mlir.constant(0 : index) : i64
    %179 = llvm.insertvalue %178, %177[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %180 = llvm.insertvalue %28, %179[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %181 = llvm.insertvalue %38, %180[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %182 = llvm.insertvalue %26, %181[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %183 = llvm.mlir.constant(1 : index) : i64
    %184 = llvm.insertvalue %183, %182[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %185 = llvm.intr.stacksave : !llvm.ptr
    %186 = llvm.mlir.constant(2 : i64) : i64
    %187 = llvm.mlir.constant(1 : index) : i64
    %188 = llvm.alloca %187 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %143, %188 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %189 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %190 = llvm.insertvalue %186, %189[0] : !llvm.struct<(i64, ptr)> 
    %191 = llvm.insertvalue %188, %190[1] : !llvm.struct<(i64, ptr)> 
    %192 = llvm.mlir.constant(2 : i64) : i64
    %193 = llvm.mlir.constant(1 : index) : i64
    %194 = llvm.alloca %193 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %184, %194 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %195 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %196 = llvm.insertvalue %192, %195[0] : !llvm.struct<(i64, ptr)> 
    %197 = llvm.insertvalue %194, %196[1] : !llvm.struct<(i64, ptr)> 
    %198 = llvm.mlir.constant(1 : index) : i64
    %199 = llvm.alloca %198 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %191, %199 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %200 = llvm.alloca %198 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %197, %200 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %201 = llvm.mlir.zero : !llvm.ptr
    %202 = llvm.getelementptr %201[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %203 = llvm.ptrtoint %202 : !llvm.ptr to i64
    llvm.call @memrefCopy(%203, %199, %200) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %185 : !llvm.ptr
    %204 = llvm.mlir.constant(1 : index) : i64
    %205 = llvm.mul %33, %204  : i64
    %206 = llvm.mul %205, %38  : i64
    %207 = llvm.mlir.zero : !llvm.ptr
    %208 = llvm.getelementptr %207[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %209 = llvm.ptrtoint %208 : !llvm.ptr to i64
    %210 = llvm.mul %206, %209  : i64
    %211 = llvm.getelementptr %53[%57] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %212 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %213 = llvm.extractvalue %7[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %214 = llvm.getelementptr %212[%213] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%214, %211, %210) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.return
  }
}

