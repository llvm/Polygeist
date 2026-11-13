module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.func @memrefCopy(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @kernel_syrk(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: f64, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: !llvm.ptr, %arg12: !llvm.ptr, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64) {
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
    %16 = llvm.mlir.constant(0 : index) : i64
    %17 = llvm.mlir.constant(1 : index) : i64
    %18 = llvm.sext %arg1 : i32 to i64
    %19 = llvm.sext %arg0 : i32 to i64
    %20 = llvm.mlir.constant(1 : index) : i64
    %21 = llvm.extractvalue %7[3] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.alloca %20 x !llvm.array<2 x i64> : (i64) -> !llvm.ptr
    llvm.store %21, %22 : !llvm.array<2 x i64>, !llvm.ptr
    %23 = llvm.getelementptr %22[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
    %24 = llvm.load %23 : !llvm.ptr -> i64
    %25 = llvm.mlir.constant(1 : index) : i64
    %26 = llvm.extractvalue %7[3] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %27 = llvm.alloca %25 x !llvm.array<2 x i64> : (i64) -> !llvm.ptr
    llvm.store %26, %27 : !llvm.array<2 x i64>, !llvm.ptr
    %28 = llvm.getelementptr %27[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
    %29 = llvm.load %28 : !llvm.ptr -> i64
    %30 = llvm.mlir.constant(1 : index) : i64
    %31 = llvm.mul %29, %24  : i64
    %32 = llvm.mlir.zero : !llvm.ptr
    %33 = llvm.getelementptr %32[%31] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %34 = llvm.ptrtoint %33 : !llvm.ptr to i64
    %35 = llvm.mlir.constant(64 : index) : i64
    %36 = llvm.add %34, %35  : i64
    %37 = llvm.call @malloc(%36) : (i64) -> !llvm.ptr
    %38 = llvm.ptrtoint %37 : !llvm.ptr to i64
    %39 = llvm.mlir.constant(1 : index) : i64
    %40 = llvm.sub %35, %39  : i64
    %41 = llvm.add %38, %40  : i64
    %42 = llvm.urem %41, %35  : i64
    %43 = llvm.sub %41, %42  : i64
    %44 = llvm.inttoptr %43 : i64 to !llvm.ptr
    %45 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %46 = llvm.insertvalue %37, %45[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %47 = llvm.insertvalue %44, %46[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %48 = llvm.mlir.constant(0 : index) : i64
    %49 = llvm.insertvalue %48, %47[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %50 = llvm.insertvalue %24, %49[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %51 = llvm.insertvalue %29, %50[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %52 = llvm.insertvalue %29, %51[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %53 = llvm.insertvalue %30, %52[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %54 = llvm.mlir.constant(1 : index) : i64
    %55 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %56 = llvm.mul %55, %54  : i64
    %57 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %58 = llvm.mul %56, %57  : i64
    %59 = llvm.mlir.zero : !llvm.ptr
    %60 = llvm.getelementptr %59[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %61 = llvm.ptrtoint %60 : !llvm.ptr to i64
    %62 = llvm.mul %58, %61  : i64
    %63 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %64 = llvm.extractvalue %7[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %65 = llvm.getelementptr %63[%64] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %66 = llvm.getelementptr %44[%48] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%66, %65, %62) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb1(%16 : i64)
  ^bb1(%67: i64):  // 2 preds: ^bb0, ^bb5
    %68 = llvm.icmp "slt" %67, %24 : i64
    llvm.cond_br %68, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%16 : i64)
  ^bb3(%69: i64):  // 2 preds: ^bb2, ^bb4
    %70 = llvm.icmp "slt" %69, %29 : i64
    llvm.cond_br %70, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %71 = llvm.mul %67, %29  : i64
    %72 = llvm.add %71, %69  : i64
    %73 = llvm.getelementptr %44[%72] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %74 = llvm.load %73 : !llvm.ptr -> f64
    %75 = llvm.fmul %74, %arg3  : f64
    %76 = llvm.add %67, %17  : i64
    %77 = llvm.icmp "slt" %69, %76 : i64
    %78 = llvm.select %77, %75, %74 : i1, f64
    %79 = llvm.mul %67, %29  : i64
    %80 = llvm.add %79, %69  : i64
    %81 = llvm.getelementptr %44[%80] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %78, %81 : f64, !llvm.ptr
    %82 = llvm.add %69, %17  : i64
    llvm.br ^bb3(%82 : i64)
  ^bb5:  // pred: ^bb3
    %83 = llvm.add %67, %17  : i64
    llvm.br ^bb1(%83 : i64)
  ^bb6:  // pred: ^bb1
    %84 = llvm.extractvalue %15[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %85 = llvm.extractvalue %15[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %86 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %87 = llvm.insertvalue %84, %86[0] : !llvm.struct<(ptr, ptr, i64)> 
    %88 = llvm.insertvalue %85, %87[1] : !llvm.struct<(ptr, ptr, i64)> 
    %89 = llvm.mlir.constant(0 : index) : i64
    %90 = llvm.insertvalue %89, %88[2] : !llvm.struct<(ptr, ptr, i64)> 
    %91 = llvm.extractvalue %15[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %92 = llvm.extractvalue %15[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %93 = llvm.extractvalue %15[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %94 = llvm.extractvalue %15[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %95 = llvm.extractvalue %15[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %96 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %97 = llvm.insertvalue %84, %96[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %98 = llvm.insertvalue %85, %97[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %99 = llvm.mlir.constant(0 : index) : i64
    %100 = llvm.insertvalue %99, %98[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %101 = llvm.insertvalue %19, %100[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %102 = llvm.insertvalue %94, %101[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %103 = llvm.insertvalue %18, %102[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %104 = llvm.mlir.constant(1 : index) : i64
    %105 = llvm.insertvalue %104, %103[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %106 = llvm.extractvalue %15[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %107 = llvm.extractvalue %15[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %108 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %109 = llvm.insertvalue %106, %108[0] : !llvm.struct<(ptr, ptr, i64)> 
    %110 = llvm.insertvalue %107, %109[1] : !llvm.struct<(ptr, ptr, i64)> 
    %111 = llvm.mlir.constant(0 : index) : i64
    %112 = llvm.insertvalue %111, %110[2] : !llvm.struct<(ptr, ptr, i64)> 
    %113 = llvm.extractvalue %15[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %114 = llvm.extractvalue %15[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %115 = llvm.extractvalue %15[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %116 = llvm.extractvalue %15[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %117 = llvm.extractvalue %15[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %118 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %119 = llvm.insertvalue %106, %118[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %120 = llvm.insertvalue %107, %119[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %121 = llvm.mlir.constant(0 : index) : i64
    %122 = llvm.insertvalue %121, %120[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %123 = llvm.insertvalue %19, %122[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %124 = llvm.insertvalue %116, %123[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %125 = llvm.insertvalue %18, %124[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %126 = llvm.mlir.constant(1 : index) : i64
    %127 = llvm.insertvalue %126, %125[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %128 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %129 = llvm.insertvalue %37, %128[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %130 = llvm.insertvalue %44, %129[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %131 = llvm.mlir.constant(0 : index) : i64
    %132 = llvm.insertvalue %131, %130[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %133 = llvm.insertvalue %19, %132[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %134 = llvm.insertvalue %29, %133[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %135 = llvm.insertvalue %19, %134[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %136 = llvm.mlir.constant(1 : index) : i64
    %137 = llvm.insertvalue %136, %135[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb7(%16 : i64)
  ^bb7(%138: i64):  // 2 preds: ^bb6, ^bb14
    %139 = llvm.icmp "slt" %138, %19 : i64
    llvm.cond_br %139, ^bb8, ^bb15
  ^bb8:  // pred: ^bb7
    llvm.br ^bb9(%16 : i64)
  ^bb9(%140: i64):  // 2 preds: ^bb8, ^bb13
    %141 = llvm.icmp "slt" %140, %18 : i64
    llvm.cond_br %141, ^bb10, ^bb14
  ^bb10:  // pred: ^bb9
    llvm.br ^bb11(%16 : i64)
  ^bb11(%142: i64):  // 2 preds: ^bb10, ^bb12
    %143 = llvm.icmp "slt" %142, %19 : i64
    llvm.cond_br %143, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    %144 = llvm.getelementptr %85[%99] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %145 = llvm.mul %138, %94  : i64
    %146 = llvm.add %145, %140  : i64
    %147 = llvm.getelementptr %144[%146] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %148 = llvm.load %147 : !llvm.ptr -> f64
    %149 = llvm.getelementptr %107[%121] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %150 = llvm.mul %142, %116  : i64
    %151 = llvm.add %150, %140  : i64
    %152 = llvm.getelementptr %149[%151] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %153 = llvm.load %152 : !llvm.ptr -> f64
    %154 = llvm.getelementptr %44[%131] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %155 = llvm.mul %138, %29  : i64
    %156 = llvm.add %155, %142  : i64
    %157 = llvm.getelementptr %154[%156] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %158 = llvm.load %157 : !llvm.ptr -> f64
    %159 = llvm.fmul %arg2, %148  : f64
    %160 = llvm.fmul %159, %153  : f64
    %161 = llvm.fadd %158, %160  : f64
    %162 = llvm.add %138, %17  : i64
    %163 = llvm.icmp "slt" %142, %162 : i64
    %164 = llvm.select %163, %161, %158 : i1, f64
    %165 = llvm.getelementptr %44[%131] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %166 = llvm.mul %138, %29  : i64
    %167 = llvm.add %166, %142  : i64
    %168 = llvm.getelementptr %165[%167] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %164, %168 : f64, !llvm.ptr
    %169 = llvm.add %142, %17  : i64
    llvm.br ^bb11(%169 : i64)
  ^bb13:  // pred: ^bb11
    %170 = llvm.add %140, %17  : i64
    llvm.br ^bb9(%170 : i64)
  ^bb14:  // pred: ^bb9
    %171 = llvm.add %138, %17  : i64
    llvm.br ^bb7(%171 : i64)
  ^bb15:  // pred: ^bb7
    %172 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %173 = llvm.insertvalue %37, %172[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %174 = llvm.insertvalue %44, %173[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %175 = llvm.mlir.constant(0 : index) : i64
    %176 = llvm.insertvalue %175, %174[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %177 = llvm.insertvalue %19, %176[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %178 = llvm.insertvalue %29, %177[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %179 = llvm.insertvalue %19, %178[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %180 = llvm.mlir.constant(1 : index) : i64
    %181 = llvm.insertvalue %180, %179[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %182 = llvm.intr.stacksave : !llvm.ptr
    %183 = llvm.mlir.constant(2 : i64) : i64
    %184 = llvm.mlir.constant(1 : index) : i64
    %185 = llvm.alloca %184 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %137, %185 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %186 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %187 = llvm.insertvalue %183, %186[0] : !llvm.struct<(i64, ptr)> 
    %188 = llvm.insertvalue %185, %187[1] : !llvm.struct<(i64, ptr)> 
    %189 = llvm.mlir.constant(2 : i64) : i64
    %190 = llvm.mlir.constant(1 : index) : i64
    %191 = llvm.alloca %190 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %181, %191 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %192 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %193 = llvm.insertvalue %189, %192[0] : !llvm.struct<(i64, ptr)> 
    %194 = llvm.insertvalue %191, %193[1] : !llvm.struct<(i64, ptr)> 
    %195 = llvm.mlir.constant(1 : index) : i64
    %196 = llvm.alloca %195 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %188, %196 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %197 = llvm.alloca %195 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %194, %197 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %198 = llvm.mlir.zero : !llvm.ptr
    %199 = llvm.getelementptr %198[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %200 = llvm.ptrtoint %199 : !llvm.ptr to i64
    llvm.call @memrefCopy(%200, %196, %197) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %182 : !llvm.ptr
    %201 = llvm.mlir.constant(1 : index) : i64
    %202 = llvm.mul %24, %201  : i64
    %203 = llvm.mul %202, %29  : i64
    %204 = llvm.mlir.zero : !llvm.ptr
    %205 = llvm.getelementptr %204[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %206 = llvm.ptrtoint %205 : !llvm.ptr to i64
    %207 = llvm.mul %203, %206  : i64
    %208 = llvm.getelementptr %44[%48] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %209 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %210 = llvm.extractvalue %7[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %211 = llvm.getelementptr %209[%210] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%211, %208, %207) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.return
  }
}

