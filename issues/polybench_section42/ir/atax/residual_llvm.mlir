module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @kernel_atax(%arg0: i32, %arg1: i32, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: !llvm.ptr, %arg10: !llvm.ptr, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: !llvm.ptr, %arg15: !llvm.ptr, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: !llvm.ptr, %arg20: !llvm.ptr, %arg21: i64, %arg22: i64, %arg23: i64) {
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
    %26 = llvm.mlir.constant(0 : index) : i64
    %27 = llvm.mlir.constant(1 : index) : i64
    %28 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %29 = llvm.sext %arg1 : i32 to i64
    %30 = llvm.mlir.constant(1 : index) : i64
    %31 = llvm.extractvalue %19[3] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %32 = llvm.alloca %30 x !llvm.array<1 x i64> : (i64) -> !llvm.ptr
    llvm.store %31, %32 : !llvm.array<1 x i64>, !llvm.ptr
    %33 = llvm.getelementptr %32[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x i64>
    %34 = llvm.load %33 : !llvm.ptr -> i64
    %35 = llvm.mlir.constant(1 : index) : i64
    %36 = llvm.mlir.zero : !llvm.ptr
    %37 = llvm.getelementptr %36[%34] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %38 = llvm.ptrtoint %37 : !llvm.ptr to i64
    %39 = llvm.mlir.constant(64 : index) : i64
    %40 = llvm.add %38, %39  : i64
    %41 = llvm.call @malloc(%40) : (i64) -> !llvm.ptr
    %42 = llvm.ptrtoint %41 : !llvm.ptr to i64
    %43 = llvm.mlir.constant(1 : index) : i64
    %44 = llvm.sub %39, %43  : i64
    %45 = llvm.add %42, %44  : i64
    %46 = llvm.urem %45, %39  : i64
    %47 = llvm.sub %45, %46  : i64
    %48 = llvm.inttoptr %47 : i64 to !llvm.ptr
    %49 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %50 = llvm.insertvalue %41, %49[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %51 = llvm.insertvalue %48, %50[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %52 = llvm.mlir.constant(0 : index) : i64
    %53 = llvm.insertvalue %52, %51[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %54 = llvm.insertvalue %34, %53[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %55 = llvm.insertvalue %35, %54[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb1(%26 : i64)
  ^bb1(%56: i64):  // 2 preds: ^bb0, ^bb2
    %57 = llvm.icmp "slt" %56, %34 : i64
    llvm.cond_br %57, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %58 = llvm.getelementptr %48[%56] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %28, %58 : f64, !llvm.ptr
    %59 = llvm.add %56, %27  : i64
    llvm.br ^bb1(%59 : i64)
  ^bb3:  // pred: ^bb1
    %60 = llvm.sext %arg0 : i32 to i64
    %61 = llvm.mlir.constant(1 : index) : i64
    %62 = llvm.extractvalue %25[3] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %63 = llvm.alloca %61 x !llvm.array<1 x i64> : (i64) -> !llvm.ptr
    llvm.store %62, %63 : !llvm.array<1 x i64>, !llvm.ptr
    %64 = llvm.getelementptr %63[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x i64>
    %65 = llvm.load %64 : !llvm.ptr -> i64
    %66 = llvm.mlir.constant(1 : index) : i64
    %67 = llvm.mlir.zero : !llvm.ptr
    %68 = llvm.getelementptr %67[%65] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %69 = llvm.ptrtoint %68 : !llvm.ptr to i64
    %70 = llvm.mlir.constant(64 : index) : i64
    %71 = llvm.add %69, %70  : i64
    %72 = llvm.call @malloc(%71) : (i64) -> !llvm.ptr
    %73 = llvm.ptrtoint %72 : !llvm.ptr to i64
    %74 = llvm.mlir.constant(1 : index) : i64
    %75 = llvm.sub %70, %74  : i64
    %76 = llvm.add %73, %75  : i64
    %77 = llvm.urem %76, %70  : i64
    %78 = llvm.sub %76, %77  : i64
    %79 = llvm.inttoptr %78 : i64 to !llvm.ptr
    %80 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %81 = llvm.insertvalue %72, %80[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %82 = llvm.insertvalue %79, %81[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %83 = llvm.mlir.constant(0 : index) : i64
    %84 = llvm.insertvalue %83, %82[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %85 = llvm.insertvalue %65, %84[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %86 = llvm.insertvalue %66, %85[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb4(%26 : i64)
  ^bb4(%87: i64):  // 2 preds: ^bb3, ^bb5
    %88 = llvm.icmp "slt" %87, %65 : i64
    llvm.cond_br %88, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %89 = llvm.getelementptr %79[%87] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %28, %89 : f64, !llvm.ptr
    %90 = llvm.add %87, %27  : i64
    llvm.br ^bb4(%90 : i64)
  ^bb6:  // pred: ^bb4
    %91 = llvm.extractvalue %7[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %92 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %93 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %94 = llvm.insertvalue %91, %93[0] : !llvm.struct<(ptr, ptr, i64)> 
    %95 = llvm.insertvalue %92, %94[1] : !llvm.struct<(ptr, ptr, i64)> 
    %96 = llvm.mlir.constant(0 : index) : i64
    %97 = llvm.insertvalue %96, %95[2] : !llvm.struct<(ptr, ptr, i64)> 
    %98 = llvm.extractvalue %7[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %99 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %100 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %101 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %102 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %103 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %104 = llvm.insertvalue %91, %103[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %105 = llvm.insertvalue %92, %104[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %106 = llvm.mlir.constant(0 : index) : i64
    %107 = llvm.insertvalue %106, %105[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %108 = llvm.insertvalue %60, %107[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %109 = llvm.insertvalue %101, %108[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %110 = llvm.insertvalue %29, %109[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %111 = llvm.mlir.constant(1 : index) : i64
    %112 = llvm.insertvalue %111, %110[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %113 = llvm.extractvalue %13[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %114 = llvm.extractvalue %13[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %115 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %116 = llvm.insertvalue %113, %115[0] : !llvm.struct<(ptr, ptr, i64)> 
    %117 = llvm.insertvalue %114, %116[1] : !llvm.struct<(ptr, ptr, i64)> 
    %118 = llvm.mlir.constant(0 : index) : i64
    %119 = llvm.insertvalue %118, %117[2] : !llvm.struct<(ptr, ptr, i64)> 
    %120 = llvm.extractvalue %13[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %121 = llvm.extractvalue %13[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %122 = llvm.extractvalue %13[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %123 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %124 = llvm.insertvalue %113, %123[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %125 = llvm.insertvalue %114, %124[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %126 = llvm.mlir.constant(0 : index) : i64
    %127 = llvm.insertvalue %126, %125[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %128 = llvm.insertvalue %29, %127[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %129 = llvm.mlir.constant(1 : index) : i64
    %130 = llvm.insertvalue %129, %128[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %131 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %132 = llvm.insertvalue %72, %131[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %133 = llvm.insertvalue %79, %132[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %134 = llvm.mlir.constant(0 : index) : i64
    %135 = llvm.insertvalue %134, %133[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %136 = llvm.insertvalue %60, %135[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %137 = llvm.mlir.constant(1 : index) : i64
    %138 = llvm.insertvalue %137, %136[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %139 = llvm.mlir.constant(1 : index) : i64
    %140 = llvm.mlir.zero : !llvm.ptr
    %141 = llvm.getelementptr %140[%60] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %142 = llvm.ptrtoint %141 : !llvm.ptr to i64
    %143 = llvm.mlir.constant(64 : index) : i64
    %144 = llvm.add %142, %143  : i64
    %145 = llvm.call @malloc(%144) : (i64) -> !llvm.ptr
    %146 = llvm.ptrtoint %145 : !llvm.ptr to i64
    %147 = llvm.mlir.constant(1 : index) : i64
    %148 = llvm.sub %143, %147  : i64
    %149 = llvm.add %146, %148  : i64
    %150 = llvm.urem %149, %143  : i64
    %151 = llvm.sub %149, %150  : i64
    %152 = llvm.inttoptr %151 : i64 to !llvm.ptr
    %153 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %154 = llvm.insertvalue %145, %153[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %155 = llvm.insertvalue %152, %154[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %156 = llvm.mlir.constant(0 : index) : i64
    %157 = llvm.insertvalue %156, %155[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %158 = llvm.insertvalue %60, %157[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %159 = llvm.insertvalue %139, %158[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %160 = llvm.mlir.constant(1 : index) : i64
    %161 = llvm.mul %60, %160  : i64
    %162 = llvm.mlir.zero : !llvm.ptr
    %163 = llvm.getelementptr %162[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %164 = llvm.ptrtoint %163 : !llvm.ptr to i64
    %165 = llvm.mul %161, %164  : i64
    %166 = llvm.getelementptr %79[%134] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %167 = llvm.getelementptr %152[%156] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%167, %166, %165) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb7(%26 : i64)
  ^bb7(%168: i64):  // 2 preds: ^bb6, ^bb11
    %169 = llvm.icmp "slt" %168, %60 : i64
    llvm.cond_br %169, ^bb8, ^bb12
  ^bb8:  // pred: ^bb7
    llvm.br ^bb9(%26 : i64)
  ^bb9(%170: i64):  // 2 preds: ^bb8, ^bb10
    %171 = llvm.icmp "slt" %170, %29 : i64
    llvm.cond_br %171, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    %172 = llvm.getelementptr %92[%106] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %173 = llvm.mul %168, %101  : i64
    %174 = llvm.add %173, %170  : i64
    %175 = llvm.getelementptr %172[%174] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %176 = llvm.load %175 : !llvm.ptr -> f64
    %177 = llvm.getelementptr %114[%170] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %178 = llvm.load %177 : !llvm.ptr -> f64
    %179 = llvm.getelementptr %152[%168] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %180 = llvm.load %179 : !llvm.ptr -> f64
    %181 = llvm.fmul %176, %178  : f64
    %182 = llvm.fadd %180, %181  : f64
    %183 = llvm.getelementptr %152[%168] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %182, %183 : f64, !llvm.ptr
    %184 = llvm.add %170, %27  : i64
    llvm.br ^bb9(%184 : i64)
  ^bb11:  // pred: ^bb9
    %185 = llvm.add %168, %27  : i64
    llvm.br ^bb7(%185 : i64)
  ^bb12:  // pred: ^bb7
    %186 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %187 = llvm.insertvalue %72, %186[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %188 = llvm.insertvalue %79, %187[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %189 = llvm.mlir.constant(0 : index) : i64
    %190 = llvm.insertvalue %189, %188[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %191 = llvm.insertvalue %60, %190[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %192 = llvm.mlir.constant(1 : index) : i64
    %193 = llvm.insertvalue %192, %191[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %194 = llvm.mlir.constant(1 : index) : i64
    %195 = llvm.mul %60, %194  : i64
    %196 = llvm.mlir.zero : !llvm.ptr
    %197 = llvm.getelementptr %196[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %198 = llvm.ptrtoint %197 : !llvm.ptr to i64
    %199 = llvm.mul %195, %198  : i64
    %200 = llvm.getelementptr %152[%156] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %201 = llvm.getelementptr %79[%189] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%201, %200, %199) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %202 = llvm.mlir.constant(1 : index) : i64
    %203 = llvm.mul %65, %202  : i64
    %204 = llvm.mlir.zero : !llvm.ptr
    %205 = llvm.getelementptr %204[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %206 = llvm.ptrtoint %205 : !llvm.ptr to i64
    %207 = llvm.mul %203, %206  : i64
    %208 = llvm.getelementptr %79[%83] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %209 = llvm.extractvalue %25[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %210 = llvm.extractvalue %25[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %211 = llvm.getelementptr %209[%210] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%211, %208, %207) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %212 = llvm.extractvalue %7[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %213 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %214 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %215 = llvm.insertvalue %212, %214[0] : !llvm.struct<(ptr, ptr, i64)> 
    %216 = llvm.insertvalue %213, %215[1] : !llvm.struct<(ptr, ptr, i64)> 
    %217 = llvm.mlir.constant(0 : index) : i64
    %218 = llvm.insertvalue %217, %216[2] : !llvm.struct<(ptr, ptr, i64)> 
    %219 = llvm.extractvalue %7[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %220 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %221 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %222 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %223 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %224 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %225 = llvm.insertvalue %212, %224[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %226 = llvm.insertvalue %213, %225[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %227 = llvm.mlir.constant(0 : index) : i64
    %228 = llvm.insertvalue %227, %226[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %229 = llvm.insertvalue %60, %228[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %230 = llvm.insertvalue %222, %229[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %231 = llvm.insertvalue %29, %230[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %232 = llvm.mlir.constant(1 : index) : i64
    %233 = llvm.insertvalue %232, %231[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %234 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %235 = llvm.insertvalue %41, %234[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %236 = llvm.insertvalue %48, %235[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %237 = llvm.mlir.constant(0 : index) : i64
    %238 = llvm.insertvalue %237, %236[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %239 = llvm.insertvalue %29, %238[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %240 = llvm.mlir.constant(1 : index) : i64
    %241 = llvm.insertvalue %240, %239[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb13(%26 : i64)
  ^bb13(%242: i64):  // 2 preds: ^bb12, ^bb17
    %243 = llvm.icmp "slt" %242, %60 : i64
    llvm.cond_br %243, ^bb14, ^bb18
  ^bb14:  // pred: ^bb13
    llvm.br ^bb15(%26 : i64)
  ^bb15(%244: i64):  // 2 preds: ^bb14, ^bb16
    %245 = llvm.icmp "slt" %244, %29 : i64
    llvm.cond_br %245, ^bb16, ^bb17
  ^bb16:  // pred: ^bb15
    %246 = llvm.getelementptr %213[%227] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %247 = llvm.mul %242, %222  : i64
    %248 = llvm.add %247, %244  : i64
    %249 = llvm.getelementptr %246[%248] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %250 = llvm.load %249 : !llvm.ptr -> f64
    %251 = llvm.getelementptr %152[%242] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %252 = llvm.load %251 : !llvm.ptr -> f64
    %253 = llvm.getelementptr %48[%244] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %254 = llvm.load %253 : !llvm.ptr -> f64
    %255 = llvm.fmul %250, %252  : f64
    %256 = llvm.fadd %254, %255  : f64
    %257 = llvm.getelementptr %48[%244] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %256, %257 : f64, !llvm.ptr
    %258 = llvm.add %244, %27  : i64
    llvm.br ^bb15(%258 : i64)
  ^bb17:  // pred: ^bb15
    %259 = llvm.add %242, %27  : i64
    llvm.br ^bb13(%259 : i64)
  ^bb18:  // pred: ^bb13
    %260 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %261 = llvm.insertvalue %41, %260[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %262 = llvm.insertvalue %48, %261[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %263 = llvm.mlir.constant(0 : index) : i64
    %264 = llvm.insertvalue %263, %262[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %265 = llvm.insertvalue %29, %264[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %266 = llvm.mlir.constant(1 : index) : i64
    %267 = llvm.insertvalue %266, %265[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %268 = llvm.mlir.constant(1 : index) : i64
    %269 = llvm.mul %29, %268  : i64
    %270 = llvm.mlir.zero : !llvm.ptr
    %271 = llvm.getelementptr %270[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %272 = llvm.ptrtoint %271 : !llvm.ptr to i64
    %273 = llvm.mul %269, %272  : i64
    %274 = llvm.getelementptr %48[%237] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %275 = llvm.getelementptr %48[%263] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%275, %274, %273) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %276 = llvm.mlir.constant(1 : index) : i64
    %277 = llvm.mul %34, %276  : i64
    %278 = llvm.mlir.zero : !llvm.ptr
    %279 = llvm.getelementptr %278[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %280 = llvm.ptrtoint %279 : !llvm.ptr to i64
    %281 = llvm.mul %277, %280  : i64
    %282 = llvm.getelementptr %48[%52] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %283 = llvm.extractvalue %19[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %284 = llvm.extractvalue %19[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %285 = llvm.getelementptr %283[%284] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%285, %282, %281) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.return
  }
}

