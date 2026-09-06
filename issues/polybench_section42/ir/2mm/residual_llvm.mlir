module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.func @memrefCopy(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @kernel_2mm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: f64, %arg5: f64, %arg6: !llvm.ptr, %arg7: !llvm.ptr, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: !llvm.ptr, %arg14: !llvm.ptr, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: !llvm.ptr, %arg21: !llvm.ptr, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: i64, %arg27: !llvm.ptr, %arg28: !llvm.ptr, %arg29: i64, %arg30: i64, %arg31: i64, %arg32: i64, %arg33: i64, %arg34: !llvm.ptr, %arg35: !llvm.ptr, %arg36: i64, %arg37: i64, %arg38: i64, %arg39: i64, %arg40: i64) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg6, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.insertvalue %arg7, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.insertvalue %arg8, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.insertvalue %arg9, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.insertvalue %arg11, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.insertvalue %arg10, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.insertvalue %arg12, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.insertvalue %arg13, %8[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %10 = llvm.insertvalue %arg14, %9[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.insertvalue %arg15, %10[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %12 = llvm.insertvalue %arg16, %11[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.insertvalue %arg18, %12[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.insertvalue %arg17, %13[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.insertvalue %arg19, %14[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %17 = llvm.insertvalue %arg20, %16[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.insertvalue %arg21, %17[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %19 = llvm.insertvalue %arg22, %18[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.insertvalue %arg23, %19[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.insertvalue %arg25, %20[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.insertvalue %arg24, %21[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.insertvalue %arg26, %22[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %25 = llvm.insertvalue %arg27, %24[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %26 = llvm.insertvalue %arg28, %25[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %27 = llvm.insertvalue %arg29, %26[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %28 = llvm.insertvalue %arg30, %27[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %29 = llvm.insertvalue %arg32, %28[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %30 = llvm.insertvalue %arg31, %29[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %31 = llvm.insertvalue %arg33, %30[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %32 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %33 = llvm.insertvalue %arg34, %32[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %34 = llvm.insertvalue %arg35, %33[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %35 = llvm.insertvalue %arg36, %34[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %36 = llvm.insertvalue %arg37, %35[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %37 = llvm.insertvalue %arg39, %36[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %38 = llvm.insertvalue %arg38, %37[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %39 = llvm.insertvalue %arg40, %38[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %40 = llvm.mlir.constant(0 : index) : i64
    %41 = llvm.mlir.constant(1 : index) : i64
    %42 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %43 = llvm.sext %arg2 : i32 to i64
    %44 = llvm.sext %arg3 : i32 to i64
    %45 = llvm.sext %arg1 : i32 to i64
    %46 = llvm.sext %arg0 : i32 to i64
    %47 = llvm.mlir.constant(1 : index) : i64
    %48 = llvm.extractvalue %7[3] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %49 = llvm.alloca %47 x !llvm.array<2 x i64> : (i64) -> !llvm.ptr
    llvm.store %48, %49 : !llvm.array<2 x i64>, !llvm.ptr
    %50 = llvm.getelementptr %49[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
    %51 = llvm.load %50 : !llvm.ptr -> i64
    %52 = llvm.mlir.constant(1 : index) : i64
    %53 = llvm.extractvalue %7[3] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %54 = llvm.alloca %52 x !llvm.array<2 x i64> : (i64) -> !llvm.ptr
    llvm.store %53, %54 : !llvm.array<2 x i64>, !llvm.ptr
    %55 = llvm.getelementptr %54[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
    %56 = llvm.load %55 : !llvm.ptr -> i64
    %57 = llvm.mlir.constant(1 : index) : i64
    %58 = llvm.mul %56, %51  : i64
    %59 = llvm.mlir.zero : !llvm.ptr
    %60 = llvm.getelementptr %59[%58] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %61 = llvm.ptrtoint %60 : !llvm.ptr to i64
    %62 = llvm.mlir.constant(64 : index) : i64
    %63 = llvm.add %61, %62  : i64
    %64 = llvm.call @malloc(%63) : (i64) -> !llvm.ptr
    %65 = llvm.ptrtoint %64 : !llvm.ptr to i64
    %66 = llvm.mlir.constant(1 : index) : i64
    %67 = llvm.sub %62, %66  : i64
    %68 = llvm.add %65, %67  : i64
    %69 = llvm.urem %68, %62  : i64
    %70 = llvm.sub %68, %69  : i64
    %71 = llvm.inttoptr %70 : i64 to !llvm.ptr
    %72 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %73 = llvm.insertvalue %64, %72[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %74 = llvm.insertvalue %71, %73[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %75 = llvm.mlir.constant(0 : index) : i64
    %76 = llvm.insertvalue %75, %74[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %77 = llvm.insertvalue %51, %76[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %78 = llvm.insertvalue %56, %77[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %79 = llvm.insertvalue %56, %78[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %80 = llvm.insertvalue %57, %79[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb1(%40 : i64)
  ^bb1(%81: i64):  // 2 preds: ^bb0, ^bb5
    %82 = llvm.icmp "slt" %81, %51 : i64
    llvm.cond_br %82, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%40 : i64)
  ^bb3(%83: i64):  // 2 preds: ^bb2, ^bb4
    %84 = llvm.icmp "slt" %83, %56 : i64
    llvm.cond_br %84, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %85 = llvm.mul %81, %56  : i64
    %86 = llvm.add %85, %83  : i64
    %87 = llvm.getelementptr %71[%86] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %42, %87 : f64, !llvm.ptr
    %88 = llvm.add %83, %41  : i64
    llvm.br ^bb3(%88 : i64)
  ^bb5:  // pred: ^bb3
    %89 = llvm.add %81, %41  : i64
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
    %107 = llvm.insertvalue %46, %106[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %108 = llvm.insertvalue %100, %107[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %109 = llvm.insertvalue %43, %108[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
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
    %129 = llvm.insertvalue %43, %128[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %130 = llvm.insertvalue %122, %129[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %131 = llvm.insertvalue %45, %130[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %132 = llvm.mlir.constant(1 : index) : i64
    %133 = llvm.insertvalue %132, %131[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %134 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %135 = llvm.insertvalue %64, %134[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %136 = llvm.insertvalue %71, %135[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %137 = llvm.mlir.constant(0 : index) : i64
    %138 = llvm.insertvalue %137, %136[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %139 = llvm.insertvalue %46, %138[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %140 = llvm.insertvalue %56, %139[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %141 = llvm.insertvalue %45, %140[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %142 = llvm.mlir.constant(1 : index) : i64
    %143 = llvm.insertvalue %142, %141[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %144 = llvm.mlir.constant(1 : index) : i64
    %145 = llvm.mul %45, %46  : i64
    %146 = llvm.mlir.zero : !llvm.ptr
    %147 = llvm.getelementptr %146[%145] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %148 = llvm.ptrtoint %147 : !llvm.ptr to i64
    %149 = llvm.mlir.constant(64 : index) : i64
    %150 = llvm.add %148, %149  : i64
    %151 = llvm.call @malloc(%150) : (i64) -> !llvm.ptr
    %152 = llvm.ptrtoint %151 : !llvm.ptr to i64
    %153 = llvm.mlir.constant(1 : index) : i64
    %154 = llvm.sub %149, %153  : i64
    %155 = llvm.add %152, %154  : i64
    %156 = llvm.urem %155, %149  : i64
    %157 = llvm.sub %155, %156  : i64
    %158 = llvm.inttoptr %157 : i64 to !llvm.ptr
    %159 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %160 = llvm.insertvalue %151, %159[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %161 = llvm.insertvalue %158, %160[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %162 = llvm.mlir.constant(0 : index) : i64
    %163 = llvm.insertvalue %162, %161[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %164 = llvm.insertvalue %46, %163[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %165 = llvm.insertvalue %45, %164[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %166 = llvm.insertvalue %45, %165[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %167 = llvm.insertvalue %144, %166[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %168 = llvm.intr.stacksave : !llvm.ptr
    %169 = llvm.mlir.constant(2 : i64) : i64
    %170 = llvm.mlir.constant(1 : index) : i64
    %171 = llvm.alloca %170 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %143, %171 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %172 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %173 = llvm.insertvalue %169, %172[0] : !llvm.struct<(i64, ptr)> 
    %174 = llvm.insertvalue %171, %173[1] : !llvm.struct<(i64, ptr)> 
    %175 = llvm.mlir.constant(2 : i64) : i64
    %176 = llvm.mlir.constant(1 : index) : i64
    %177 = llvm.alloca %176 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %167, %177 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %178 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %179 = llvm.insertvalue %175, %178[0] : !llvm.struct<(i64, ptr)> 
    %180 = llvm.insertvalue %177, %179[1] : !llvm.struct<(i64, ptr)> 
    %181 = llvm.mlir.constant(1 : index) : i64
    %182 = llvm.alloca %181 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %174, %182 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %183 = llvm.alloca %181 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %180, %183 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %184 = llvm.mlir.zero : !llvm.ptr
    %185 = llvm.getelementptr %184[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %186 = llvm.ptrtoint %185 : !llvm.ptr to i64
    llvm.call @memrefCopy(%186, %182, %183) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %168 : !llvm.ptr
    llvm.br ^bb7(%40 : i64)
  ^bb7(%187: i64):  // 2 preds: ^bb6, ^bb14
    %188 = llvm.icmp "slt" %187, %46 : i64
    llvm.cond_br %188, ^bb8, ^bb15
  ^bb8:  // pred: ^bb7
    llvm.br ^bb9(%40 : i64)
  ^bb9(%189: i64):  // 2 preds: ^bb8, ^bb13
    %190 = llvm.icmp "slt" %189, %45 : i64
    llvm.cond_br %190, ^bb10, ^bb14
  ^bb10:  // pred: ^bb9
    llvm.br ^bb11(%40 : i64)
  ^bb11(%191: i64):  // 2 preds: ^bb10, ^bb12
    %192 = llvm.icmp "slt" %191, %43 : i64
    llvm.cond_br %192, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    %193 = llvm.getelementptr %91[%105] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %194 = llvm.mul %187, %100  : i64
    %195 = llvm.add %194, %191  : i64
    %196 = llvm.getelementptr %193[%195] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %197 = llvm.load %196 : !llvm.ptr -> f64
    %198 = llvm.getelementptr %113[%127] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %199 = llvm.mul %191, %122  : i64
    %200 = llvm.add %199, %189  : i64
    %201 = llvm.getelementptr %198[%200] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %202 = llvm.load %201 : !llvm.ptr -> f64
    %203 = llvm.mul %187, %45  : i64
    %204 = llvm.add %203, %189  : i64
    %205 = llvm.getelementptr %158[%204] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %206 = llvm.load %205 : !llvm.ptr -> f64
    %207 = llvm.fmul %arg4, %197  : f64
    %208 = llvm.fmul %207, %202  : f64
    %209 = llvm.fadd %206, %208  : f64
    %210 = llvm.mul %187, %45  : i64
    %211 = llvm.add %210, %189  : i64
    %212 = llvm.getelementptr %158[%211] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %209, %212 : f64, !llvm.ptr
    %213 = llvm.add %191, %41  : i64
    llvm.br ^bb11(%213 : i64)
  ^bb13:  // pred: ^bb11
    %214 = llvm.add %189, %41  : i64
    llvm.br ^bb9(%214 : i64)
  ^bb14:  // pred: ^bb9
    %215 = llvm.add %187, %41  : i64
    llvm.br ^bb7(%215 : i64)
  ^bb15:  // pred: ^bb7
    %216 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %217 = llvm.insertvalue %64, %216[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %218 = llvm.insertvalue %71, %217[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %219 = llvm.mlir.constant(0 : index) : i64
    %220 = llvm.insertvalue %219, %218[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %221 = llvm.insertvalue %46, %220[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %222 = llvm.insertvalue %56, %221[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %223 = llvm.insertvalue %45, %222[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %224 = llvm.mlir.constant(1 : index) : i64
    %225 = llvm.insertvalue %224, %223[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %226 = llvm.intr.stacksave : !llvm.ptr
    %227 = llvm.mlir.constant(2 : i64) : i64
    %228 = llvm.mlir.constant(1 : index) : i64
    %229 = llvm.alloca %228 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %167, %229 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %230 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %231 = llvm.insertvalue %227, %230[0] : !llvm.struct<(i64, ptr)> 
    %232 = llvm.insertvalue %229, %231[1] : !llvm.struct<(i64, ptr)> 
    %233 = llvm.mlir.constant(2 : i64) : i64
    %234 = llvm.mlir.constant(1 : index) : i64
    %235 = llvm.alloca %234 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %225, %235 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %236 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %237 = llvm.insertvalue %233, %236[0] : !llvm.struct<(i64, ptr)> 
    %238 = llvm.insertvalue %235, %237[1] : !llvm.struct<(i64, ptr)> 
    %239 = llvm.mlir.constant(1 : index) : i64
    %240 = llvm.alloca %239 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %232, %240 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %241 = llvm.alloca %239 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %238, %241 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %242 = llvm.mlir.zero : !llvm.ptr
    %243 = llvm.getelementptr %242[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %244 = llvm.ptrtoint %243 : !llvm.ptr to i64
    llvm.call @memrefCopy(%244, %240, %241) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %226 : !llvm.ptr
    %245 = llvm.mlir.constant(1 : index) : i64
    %246 = llvm.mul %51, %245  : i64
    %247 = llvm.mul %246, %56  : i64
    %248 = llvm.mlir.zero : !llvm.ptr
    %249 = llvm.getelementptr %248[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %250 = llvm.ptrtoint %249 : !llvm.ptr to i64
    %251 = llvm.mul %247, %250  : i64
    %252 = llvm.getelementptr %71[%75] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %253 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %254 = llvm.extractvalue %7[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %255 = llvm.getelementptr %253[%254] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%255, %252, %251) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %256 = llvm.mlir.constant(1 : index) : i64
    %257 = llvm.extractvalue %39[3] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %258 = llvm.alloca %256 x !llvm.array<2 x i64> : (i64) -> !llvm.ptr
    llvm.store %257, %258 : !llvm.array<2 x i64>, !llvm.ptr
    %259 = llvm.getelementptr %258[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
    %260 = llvm.load %259 : !llvm.ptr -> i64
    %261 = llvm.mlir.constant(1 : index) : i64
    %262 = llvm.extractvalue %39[3] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %263 = llvm.alloca %261 x !llvm.array<2 x i64> : (i64) -> !llvm.ptr
    llvm.store %262, %263 : !llvm.array<2 x i64>, !llvm.ptr
    %264 = llvm.getelementptr %263[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
    %265 = llvm.load %264 : !llvm.ptr -> i64
    %266 = llvm.mlir.constant(1 : index) : i64
    %267 = llvm.mul %265, %260  : i64
    %268 = llvm.mlir.zero : !llvm.ptr
    %269 = llvm.getelementptr %268[%267] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %270 = llvm.ptrtoint %269 : !llvm.ptr to i64
    %271 = llvm.mlir.constant(64 : index) : i64
    %272 = llvm.add %270, %271  : i64
    %273 = llvm.call @malloc(%272) : (i64) -> !llvm.ptr
    %274 = llvm.ptrtoint %273 : !llvm.ptr to i64
    %275 = llvm.mlir.constant(1 : index) : i64
    %276 = llvm.sub %271, %275  : i64
    %277 = llvm.add %274, %276  : i64
    %278 = llvm.urem %277, %271  : i64
    %279 = llvm.sub %277, %278  : i64
    %280 = llvm.inttoptr %279 : i64 to !llvm.ptr
    %281 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %282 = llvm.insertvalue %273, %281[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %283 = llvm.insertvalue %280, %282[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %284 = llvm.mlir.constant(0 : index) : i64
    %285 = llvm.insertvalue %284, %283[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %286 = llvm.insertvalue %260, %285[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %287 = llvm.insertvalue %265, %286[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %288 = llvm.insertvalue %265, %287[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %289 = llvm.insertvalue %266, %288[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %290 = llvm.mlir.constant(1 : index) : i64
    %291 = llvm.extractvalue %39[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %292 = llvm.mul %291, %290  : i64
    %293 = llvm.extractvalue %39[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %294 = llvm.mul %292, %293  : i64
    %295 = llvm.mlir.zero : !llvm.ptr
    %296 = llvm.getelementptr %295[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %297 = llvm.ptrtoint %296 : !llvm.ptr to i64
    %298 = llvm.mul %294, %297  : i64
    %299 = llvm.extractvalue %39[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %300 = llvm.extractvalue %39[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %301 = llvm.getelementptr %299[%300] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %302 = llvm.getelementptr %280[%284] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%302, %301, %298) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb16(%40 : i64)
  ^bb16(%303: i64):  // 2 preds: ^bb15, ^bb20
    %304 = llvm.icmp "slt" %303, %260 : i64
    llvm.cond_br %304, ^bb17, ^bb21
  ^bb17:  // pred: ^bb16
    llvm.br ^bb18(%40 : i64)
  ^bb18(%305: i64):  // 2 preds: ^bb17, ^bb19
    %306 = llvm.icmp "slt" %305, %265 : i64
    llvm.cond_br %306, ^bb19, ^bb20
  ^bb19:  // pred: ^bb18
    %307 = llvm.mul %303, %265  : i64
    %308 = llvm.add %307, %305  : i64
    %309 = llvm.getelementptr %280[%308] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %310 = llvm.load %309 : !llvm.ptr -> f64
    %311 = llvm.fmul %310, %arg5  : f64
    %312 = llvm.mul %303, %265  : i64
    %313 = llvm.add %312, %305  : i64
    %314 = llvm.getelementptr %280[%313] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %311, %314 : f64, !llvm.ptr
    %315 = llvm.add %305, %41  : i64
    llvm.br ^bb18(%315 : i64)
  ^bb20:  // pred: ^bb18
    %316 = llvm.add %303, %41  : i64
    llvm.br ^bb16(%316 : i64)
  ^bb21:  // pred: ^bb16
    %317 = llvm.extractvalue %31[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %318 = llvm.extractvalue %31[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %319 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %320 = llvm.insertvalue %317, %319[0] : !llvm.struct<(ptr, ptr, i64)> 
    %321 = llvm.insertvalue %318, %320[1] : !llvm.struct<(ptr, ptr, i64)> 
    %322 = llvm.mlir.constant(0 : index) : i64
    %323 = llvm.insertvalue %322, %321[2] : !llvm.struct<(ptr, ptr, i64)> 
    %324 = llvm.extractvalue %31[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %325 = llvm.extractvalue %31[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %326 = llvm.extractvalue %31[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %327 = llvm.extractvalue %31[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %328 = llvm.extractvalue %31[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %329 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %330 = llvm.insertvalue %317, %329[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %331 = llvm.insertvalue %318, %330[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %332 = llvm.mlir.constant(0 : index) : i64
    %333 = llvm.insertvalue %332, %331[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %334 = llvm.insertvalue %45, %333[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %335 = llvm.insertvalue %327, %334[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %336 = llvm.insertvalue %44, %335[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %337 = llvm.mlir.constant(1 : index) : i64
    %338 = llvm.insertvalue %337, %336[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %339 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %340 = llvm.insertvalue %273, %339[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %341 = llvm.insertvalue %280, %340[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %342 = llvm.mlir.constant(0 : index) : i64
    %343 = llvm.insertvalue %342, %341[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %344 = llvm.insertvalue %46, %343[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %345 = llvm.insertvalue %265, %344[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %346 = llvm.insertvalue %44, %345[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %347 = llvm.mlir.constant(1 : index) : i64
    %348 = llvm.insertvalue %347, %346[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb22(%40 : i64)
  ^bb22(%349: i64):  // 2 preds: ^bb21, ^bb29
    %350 = llvm.icmp "slt" %349, %46 : i64
    llvm.cond_br %350, ^bb23, ^bb30
  ^bb23:  // pred: ^bb22
    llvm.br ^bb24(%40 : i64)
  ^bb24(%351: i64):  // 2 preds: ^bb23, ^bb28
    %352 = llvm.icmp "slt" %351, %44 : i64
    llvm.cond_br %352, ^bb25, ^bb29
  ^bb25:  // pred: ^bb24
    llvm.br ^bb26(%40 : i64)
  ^bb26(%353: i64):  // 2 preds: ^bb25, ^bb27
    %354 = llvm.icmp "slt" %353, %45 : i64
    llvm.cond_br %354, ^bb27, ^bb28
  ^bb27:  // pred: ^bb26
    %355 = llvm.mul %349, %45  : i64
    %356 = llvm.add %355, %353  : i64
    %357 = llvm.getelementptr %158[%356] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %358 = llvm.load %357 : !llvm.ptr -> f64
    %359 = llvm.getelementptr %318[%332] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %360 = llvm.mul %353, %327  : i64
    %361 = llvm.add %360, %351  : i64
    %362 = llvm.getelementptr %359[%361] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %363 = llvm.load %362 : !llvm.ptr -> f64
    %364 = llvm.getelementptr %280[%342] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %365 = llvm.mul %349, %265  : i64
    %366 = llvm.add %365, %351  : i64
    %367 = llvm.getelementptr %364[%366] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %368 = llvm.load %367 : !llvm.ptr -> f64
    %369 = llvm.fmul %358, %363  : f64
    %370 = llvm.fadd %368, %369  : f64
    %371 = llvm.getelementptr %280[%342] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %372 = llvm.mul %349, %265  : i64
    %373 = llvm.add %372, %351  : i64
    %374 = llvm.getelementptr %371[%373] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %370, %374 : f64, !llvm.ptr
    %375 = llvm.add %353, %41  : i64
    llvm.br ^bb26(%375 : i64)
  ^bb28:  // pred: ^bb26
    %376 = llvm.add %351, %41  : i64
    llvm.br ^bb24(%376 : i64)
  ^bb29:  // pred: ^bb24
    %377 = llvm.add %349, %41  : i64
    llvm.br ^bb22(%377 : i64)
  ^bb30:  // pred: ^bb22
    %378 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %379 = llvm.insertvalue %273, %378[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %380 = llvm.insertvalue %280, %379[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %381 = llvm.mlir.constant(0 : index) : i64
    %382 = llvm.insertvalue %381, %380[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %383 = llvm.insertvalue %46, %382[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %384 = llvm.insertvalue %265, %383[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %385 = llvm.insertvalue %44, %384[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %386 = llvm.mlir.constant(1 : index) : i64
    %387 = llvm.insertvalue %386, %385[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %388 = llvm.intr.stacksave : !llvm.ptr
    %389 = llvm.mlir.constant(2 : i64) : i64
    %390 = llvm.mlir.constant(1 : index) : i64
    %391 = llvm.alloca %390 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %348, %391 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %392 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %393 = llvm.insertvalue %389, %392[0] : !llvm.struct<(i64, ptr)> 
    %394 = llvm.insertvalue %391, %393[1] : !llvm.struct<(i64, ptr)> 
    %395 = llvm.mlir.constant(2 : i64) : i64
    %396 = llvm.mlir.constant(1 : index) : i64
    %397 = llvm.alloca %396 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %387, %397 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %398 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %399 = llvm.insertvalue %395, %398[0] : !llvm.struct<(i64, ptr)> 
    %400 = llvm.insertvalue %397, %399[1] : !llvm.struct<(i64, ptr)> 
    %401 = llvm.mlir.constant(1 : index) : i64
    %402 = llvm.alloca %401 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %394, %402 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %403 = llvm.alloca %401 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %400, %403 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %404 = llvm.mlir.zero : !llvm.ptr
    %405 = llvm.getelementptr %404[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %406 = llvm.ptrtoint %405 : !llvm.ptr to i64
    llvm.call @memrefCopy(%406, %402, %403) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %388 : !llvm.ptr
    %407 = llvm.mlir.constant(1 : index) : i64
    %408 = llvm.mul %260, %407  : i64
    %409 = llvm.mul %408, %265  : i64
    %410 = llvm.mlir.zero : !llvm.ptr
    %411 = llvm.getelementptr %410[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %412 = llvm.ptrtoint %411 : !llvm.ptr to i64
    %413 = llvm.mul %409, %412  : i64
    %414 = llvm.getelementptr %280[%284] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %415 = llvm.extractvalue %39[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %416 = llvm.extractvalue %39[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %417 = llvm.getelementptr %415[%416] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%417, %414, %413) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.return
  }
}

