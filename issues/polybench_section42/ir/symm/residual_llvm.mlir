module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.func @memrefCopy(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @kernel_symm(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: f64, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: !llvm.ptr, %arg12: !llvm.ptr, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: !llvm.ptr, %arg19: !llvm.ptr, %arg20: i64, %arg21: i64, %arg22: i64, %arg23: i64, %arg24: i64) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg4, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.insertvalue %arg5, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.insertvalue %arg6, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.insertvalue %arg7, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.insertvalue %arg9, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.insertvalue %arg8, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.insertvalue %arg10, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.insertvalue %arg11, %8[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %10 = llvm.insertvalue %arg12, %9[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.insertvalue %arg13, %10[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %12 = llvm.insertvalue %arg14, %11[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.insertvalue %arg16, %12[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.insertvalue %arg15, %13[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.insertvalue %arg17, %14[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %17 = llvm.insertvalue %arg18, %16[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.insertvalue %arg19, %17[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %19 = llvm.insertvalue %arg20, %18[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.insertvalue %arg21, %19[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.insertvalue %arg23, %20[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.insertvalue %arg22, %21[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.insertvalue %arg24, %22[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.mlir.constant(0 : index) : i64
    %25 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %26 = llvm.mlir.constant(1 : index) : i64
    %27 = llvm.sext %arg1 : i32 to i64
    %28 = llvm.mlir.constant(1 : index) : i64
    %29 = llvm.mlir.zero : !llvm.ptr
    %30 = llvm.getelementptr %29[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %31 = llvm.ptrtoint %30 : !llvm.ptr to i64
    %32 = llvm.mlir.constant(64 : index) : i64
    %33 = llvm.add %31, %32  : i64
    %34 = llvm.call @malloc(%33) : (i64) -> !llvm.ptr
    %35 = llvm.ptrtoint %34 : !llvm.ptr to i64
    %36 = llvm.mlir.constant(1 : index) : i64
    %37 = llvm.sub %32, %36  : i64
    %38 = llvm.add %35, %37  : i64
    %39 = llvm.urem %38, %32  : i64
    %40 = llvm.sub %38, %39  : i64
    %41 = llvm.inttoptr %40 : i64 to !llvm.ptr
    %42 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %43 = llvm.insertvalue %34, %42[0] : !llvm.struct<(ptr, ptr, i64)> 
    %44 = llvm.insertvalue %41, %43[1] : !llvm.struct<(ptr, ptr, i64)> 
    %45 = llvm.mlir.constant(0 : index) : i64
    %46 = llvm.insertvalue %45, %44[2] : !llvm.struct<(ptr, ptr, i64)> 
    %47 = llvm.mlir.undef : f64
    llvm.store %47, %41 : f64, !llvm.ptr
    %48 = llvm.sext %arg0 : i32 to i64
    %49 = llvm.sub %48, %26  : i64
    %50 = llvm.sub %48, %26  : i64
    %51 = llvm.mlir.constant(1 : index) : i64
    %52 = llvm.extractvalue %7[3] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %53 = llvm.alloca %51 x !llvm.array<2 x i64> : (i64) -> !llvm.ptr
    llvm.store %52, %53 : !llvm.array<2 x i64>, !llvm.ptr
    %54 = llvm.getelementptr %53[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
    %55 = llvm.load %54 : !llvm.ptr -> i64
    %56 = llvm.mlir.constant(1 : index) : i64
    %57 = llvm.extractvalue %7[3] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %58 = llvm.alloca %56 x !llvm.array<2 x i64> : (i64) -> !llvm.ptr
    llvm.store %57, %58 : !llvm.array<2 x i64>, !llvm.ptr
    %59 = llvm.getelementptr %58[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
    %60 = llvm.load %59 : !llvm.ptr -> i64
    %61 = llvm.mlir.constant(1 : index) : i64
    %62 = llvm.mul %60, %55  : i64
    %63 = llvm.mlir.zero : !llvm.ptr
    %64 = llvm.getelementptr %63[%62] : (!llvm.ptr, i64) -> !llvm.ptr, f64
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
    %76 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %77 = llvm.insertvalue %68, %76[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %78 = llvm.insertvalue %75, %77[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %79 = llvm.mlir.constant(0 : index) : i64
    %80 = llvm.insertvalue %79, %78[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %81 = llvm.insertvalue %55, %80[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %82 = llvm.insertvalue %60, %81[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %83 = llvm.insertvalue %60, %82[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %84 = llvm.insertvalue %61, %83[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %85 = llvm.mlir.constant(1 : index) : i64
    %86 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %87 = llvm.mul %86, %85  : i64
    %88 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %89 = llvm.mul %87, %88  : i64
    %90 = llvm.mlir.zero : !llvm.ptr
    %91 = llvm.getelementptr %90[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %92 = llvm.ptrtoint %91 : !llvm.ptr to i64
    %93 = llvm.mul %89, %92  : i64
    %94 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %95 = llvm.extractvalue %7[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %96 = llvm.getelementptr %94[%95] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %97 = llvm.getelementptr %75[%79] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%97, %96, %93) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb1(%24, %46, %84 : i64, !llvm.struct<(ptr, ptr, i64)>, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb1(%98: i64, %99: !llvm.struct<(ptr, ptr, i64)>, %100: !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>):  // 2 preds: ^bb0, ^bb11
    %101 = llvm.icmp "slt" %98, %48 : i64
    llvm.cond_br %101, ^bb2, ^bb12
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%24, %99, %100 : i64, !llvm.struct<(ptr, ptr, i64)>, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb3(%102: i64, %103: !llvm.struct<(ptr, ptr, i64)>, %104: !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>):  // 2 preds: ^bb2, ^bb10
    %105 = llvm.icmp "slt" %102, %27 : i64
    llvm.cond_br %105, ^bb4, ^bb11
  ^bb4:  // pred: ^bb3
    %106 = llvm.extractvalue %103[1] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %25, %106 : f64, !llvm.ptr
    %107 = llvm.extractvalue %23[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %108 = llvm.extractvalue %23[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %109 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %110 = llvm.insertvalue %107, %109[0] : !llvm.struct<(ptr, ptr, i64)> 
    %111 = llvm.insertvalue %108, %110[1] : !llvm.struct<(ptr, ptr, i64)> 
    %112 = llvm.mlir.constant(0 : index) : i64
    %113 = llvm.insertvalue %112, %111[2] : !llvm.struct<(ptr, ptr, i64)> 
    %114 = llvm.extractvalue %23[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %115 = llvm.extractvalue %23[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %116 = llvm.extractvalue %23[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %117 = llvm.extractvalue %23[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %118 = llvm.extractvalue %23[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %119 = llvm.mul %98, %117  : i64
    %120 = llvm.add %119, %102  : i64
    %121 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %122 = llvm.insertvalue %107, %121[0] : !llvm.struct<(ptr, ptr, i64)> 
    %123 = llvm.insertvalue %108, %122[1] : !llvm.struct<(ptr, ptr, i64)> 
    %124 = llvm.insertvalue %120, %123[2] : !llvm.struct<(ptr, ptr, i64)> 
    %125 = llvm.extractvalue %15[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %126 = llvm.extractvalue %15[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %127 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %128 = llvm.insertvalue %125, %127[0] : !llvm.struct<(ptr, ptr, i64)> 
    %129 = llvm.insertvalue %126, %128[1] : !llvm.struct<(ptr, ptr, i64)> 
    %130 = llvm.mlir.constant(0 : index) : i64
    %131 = llvm.insertvalue %130, %129[2] : !llvm.struct<(ptr, ptr, i64)> 
    %132 = llvm.extractvalue %15[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %133 = llvm.extractvalue %15[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %134 = llvm.extractvalue %15[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %135 = llvm.extractvalue %15[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %136 = llvm.extractvalue %15[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %137 = llvm.mul %98, %135  : i64
    %138 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %139 = llvm.insertvalue %125, %138[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %140 = llvm.insertvalue %126, %139[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %141 = llvm.insertvalue %137, %140[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %142 = llvm.insertvalue %49, %141[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %143 = llvm.mlir.constant(1 : index) : i64
    %144 = llvm.insertvalue %143, %142[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %145 = llvm.extractvalue %104[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %146 = llvm.extractvalue %104[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %147 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %148 = llvm.insertvalue %145, %147[0] : !llvm.struct<(ptr, ptr, i64)> 
    %149 = llvm.insertvalue %146, %148[1] : !llvm.struct<(ptr, ptr, i64)> 
    %150 = llvm.mlir.constant(0 : index) : i64
    %151 = llvm.insertvalue %150, %149[2] : !llvm.struct<(ptr, ptr, i64)> 
    %152 = llvm.extractvalue %104[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %153 = llvm.extractvalue %104[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %154 = llvm.extractvalue %104[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %155 = llvm.extractvalue %104[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %156 = llvm.extractvalue %104[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %157 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %158 = llvm.insertvalue %145, %157[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %159 = llvm.insertvalue %146, %158[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %160 = llvm.insertvalue %102, %159[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %161 = llvm.insertvalue %49, %160[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %162 = llvm.insertvalue %155, %161[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb5(%24 : i64)
  ^bb5(%163: i64):  // 2 preds: ^bb4, ^bb6
    %164 = llvm.icmp "slt" %163, %49 : i64
    llvm.cond_br %164, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %165 = llvm.getelementptr %108[%120] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %166 = llvm.load %165 : !llvm.ptr -> f64
    %167 = llvm.getelementptr %126[%137] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %168 = llvm.getelementptr %167[%163] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %169 = llvm.load %168 : !llvm.ptr -> f64
    %170 = llvm.getelementptr %146[%102] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %171 = llvm.mul %163, %155  : i64
    %172 = llvm.getelementptr %170[%171] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %173 = llvm.load %172 : !llvm.ptr -> f64
    %174 = llvm.fmul %arg2, %166  : f64
    %175 = llvm.fmul %174, %169  : f64
    %176 = llvm.fadd %173, %175  : f64
    %177 = llvm.icmp "slt" %163, %98 : i64
    %178 = llvm.select %177, %176, %173 : i1, f64
    %179 = llvm.getelementptr %146[%102] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %180 = llvm.mul %163, %155  : i64
    %181 = llvm.getelementptr %179[%180] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %178, %181 : f64, !llvm.ptr
    %182 = llvm.add %163, %26  : i64
    llvm.br ^bb5(%182 : i64)
  ^bb7:  // pred: ^bb5
    %183 = llvm.extractvalue %104[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %184 = llvm.extractvalue %104[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %185 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %186 = llvm.insertvalue %183, %185[0] : !llvm.struct<(ptr, ptr, i64)> 
    %187 = llvm.insertvalue %184, %186[1] : !llvm.struct<(ptr, ptr, i64)> 
    %188 = llvm.mlir.constant(0 : index) : i64
    %189 = llvm.insertvalue %188, %187[2] : !llvm.struct<(ptr, ptr, i64)> 
    %190 = llvm.extractvalue %104[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %191 = llvm.extractvalue %104[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %192 = llvm.extractvalue %104[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %193 = llvm.extractvalue %104[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %194 = llvm.extractvalue %104[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %195 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %196 = llvm.insertvalue %183, %195[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %197 = llvm.insertvalue %184, %196[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %198 = llvm.insertvalue %102, %197[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %199 = llvm.insertvalue %49, %198[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %200 = llvm.insertvalue %193, %199[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %201 = llvm.intr.stacksave : !llvm.ptr
    %202 = llvm.mlir.constant(1 : i64) : i64
    %203 = llvm.mlir.constant(1 : index) : i64
    %204 = llvm.alloca %203 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %162, %204 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %205 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %206 = llvm.insertvalue %202, %205[0] : !llvm.struct<(i64, ptr)> 
    %207 = llvm.insertvalue %204, %206[1] : !llvm.struct<(i64, ptr)> 
    %208 = llvm.mlir.constant(1 : i64) : i64
    %209 = llvm.mlir.constant(1 : index) : i64
    %210 = llvm.alloca %209 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %200, %210 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %211 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %212 = llvm.insertvalue %208, %211[0] : !llvm.struct<(i64, ptr)> 
    %213 = llvm.insertvalue %210, %212[1] : !llvm.struct<(i64, ptr)> 
    %214 = llvm.mlir.constant(1 : index) : i64
    %215 = llvm.alloca %214 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %207, %215 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %216 = llvm.alloca %214 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %213, %216 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %217 = llvm.mlir.zero : !llvm.ptr
    %218 = llvm.getelementptr %217[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %219 = llvm.ptrtoint %218 : !llvm.ptr to i64
    llvm.call @memrefCopy(%219, %215, %216) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %201 : !llvm.ptr
    %220 = llvm.extractvalue %23[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %221 = llvm.extractvalue %23[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %222 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %223 = llvm.insertvalue %220, %222[0] : !llvm.struct<(ptr, ptr, i64)> 
    %224 = llvm.insertvalue %221, %223[1] : !llvm.struct<(ptr, ptr, i64)> 
    %225 = llvm.mlir.constant(0 : index) : i64
    %226 = llvm.insertvalue %225, %224[2] : !llvm.struct<(ptr, ptr, i64)> 
    %227 = llvm.extractvalue %23[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %228 = llvm.extractvalue %23[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %229 = llvm.extractvalue %23[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %230 = llvm.extractvalue %23[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %231 = llvm.extractvalue %23[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %232 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %233 = llvm.insertvalue %220, %232[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %234 = llvm.insertvalue %221, %233[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %235 = llvm.insertvalue %102, %234[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %236 = llvm.insertvalue %50, %235[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %237 = llvm.insertvalue %230, %236[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %238 = llvm.extractvalue %15[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %239 = llvm.extractvalue %15[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %240 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %241 = llvm.insertvalue %238, %240[0] : !llvm.struct<(ptr, ptr, i64)> 
    %242 = llvm.insertvalue %239, %241[1] : !llvm.struct<(ptr, ptr, i64)> 
    %243 = llvm.mlir.constant(0 : index) : i64
    %244 = llvm.insertvalue %243, %242[2] : !llvm.struct<(ptr, ptr, i64)> 
    %245 = llvm.extractvalue %15[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %246 = llvm.extractvalue %15[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %247 = llvm.extractvalue %15[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %248 = llvm.extractvalue %15[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %249 = llvm.extractvalue %15[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %250 = llvm.mul %98, %248  : i64
    %251 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %252 = llvm.insertvalue %238, %251[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %253 = llvm.insertvalue %239, %252[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %254 = llvm.insertvalue %250, %253[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %255 = llvm.insertvalue %50, %254[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %256 = llvm.mlir.constant(1 : index) : i64
    %257 = llvm.insertvalue %256, %255[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb8(%24 : i64)
  ^bb8(%258: i64):  // 2 preds: ^bb7, ^bb9
    %259 = llvm.icmp "slt" %258, %50 : i64
    llvm.cond_br %259, ^bb9, ^bb10
  ^bb9:  // pred: ^bb8
    %260 = llvm.getelementptr %221[%102] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %261 = llvm.mul %258, %230  : i64
    %262 = llvm.getelementptr %260[%261] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %263 = llvm.load %262 : !llvm.ptr -> f64
    %264 = llvm.getelementptr %239[%250] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %265 = llvm.getelementptr %264[%258] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %266 = llvm.load %265 : !llvm.ptr -> f64
    %267 = llvm.extractvalue %103[1] : !llvm.struct<(ptr, ptr, i64)> 
    %268 = llvm.load %267 : !llvm.ptr -> f64
    %269 = llvm.fmul %263, %266  : f64
    %270 = llvm.fadd %268, %269  : f64
    %271 = llvm.icmp "slt" %258, %98 : i64
    %272 = llvm.select %271, %270, %268 : i1, f64
    %273 = llvm.extractvalue %103[1] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %272, %273 : f64, !llvm.ptr
    %274 = llvm.add %258, %26  : i64
    llvm.br ^bb8(%274 : i64)
  ^bb10:  // pred: ^bb8
    %275 = llvm.extractvalue %104[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %276 = llvm.extractvalue %104[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %277 = llvm.mul %98, %276  : i64
    %278 = llvm.add %277, %102  : i64
    %279 = llvm.getelementptr %275[%278] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %280 = llvm.load %279 : !llvm.ptr -> f64
    %281 = llvm.fmul %arg3, %280  : f64
    %282 = llvm.extractvalue %23[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %283 = llvm.extractvalue %23[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %284 = llvm.mul %98, %283  : i64
    %285 = llvm.add %284, %102  : i64
    %286 = llvm.getelementptr %282[%285] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %287 = llvm.load %286 : !llvm.ptr -> f64
    %288 = llvm.fmul %arg2, %287  : f64
    %289 = llvm.extractvalue %15[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %290 = llvm.extractvalue %15[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %291 = llvm.mul %98, %290  : i64
    %292 = llvm.add %291, %98  : i64
    %293 = llvm.getelementptr %289[%292] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %294 = llvm.load %293 : !llvm.ptr -> f64
    %295 = llvm.fmul %288, %294  : f64
    %296 = llvm.fadd %281, %295  : f64
    %297 = llvm.extractvalue %103[1] : !llvm.struct<(ptr, ptr, i64)> 
    %298 = llvm.load %297 : !llvm.ptr -> f64
    %299 = llvm.fmul %arg2, %298  : f64
    %300 = llvm.fadd %296, %299  : f64
    %301 = llvm.extractvalue %104[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %302 = llvm.extractvalue %104[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %303 = llvm.mul %98, %302  : i64
    %304 = llvm.add %303, %102  : i64
    %305 = llvm.getelementptr %301[%304] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %300, %305 : f64, !llvm.ptr
    %306 = llvm.add %102, %26  : i64
    llvm.br ^bb3(%306, %103, %104 : i64, !llvm.struct<(ptr, ptr, i64)>, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb11:  // pred: ^bb3
    %307 = llvm.add %98, %26  : i64
    llvm.br ^bb1(%307, %103, %104 : i64, !llvm.struct<(ptr, ptr, i64)>, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb12:  // pred: ^bb1
    %308 = llvm.mlir.constant(1 : index) : i64
    %309 = llvm.extractvalue %100[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %310 = llvm.mul %309, %308  : i64
    %311 = llvm.extractvalue %100[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %312 = llvm.mul %310, %311  : i64
    %313 = llvm.mlir.zero : !llvm.ptr
    %314 = llvm.getelementptr %313[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %315 = llvm.ptrtoint %314 : !llvm.ptr to i64
    %316 = llvm.mul %312, %315  : i64
    %317 = llvm.extractvalue %100[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %318 = llvm.extractvalue %100[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %319 = llvm.getelementptr %317[%318] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %320 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %321 = llvm.extractvalue %7[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %322 = llvm.getelementptr %320[%321] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%322, %319, %316) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.return
  }
}

