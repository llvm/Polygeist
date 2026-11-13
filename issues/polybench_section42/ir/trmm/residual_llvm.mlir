module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.func @memrefCopy(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @kernel_trmm(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: !llvm.ptr, %arg11: !llvm.ptr, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64) {
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
    %16 = llvm.mlir.constant(1 : index) : i64
    %17 = llvm.mlir.constant(0 : index) : i64
    %18 = llvm.sext %arg1 : i32 to i64
    %19 = llvm.sext %arg0 : i32 to i64
    %20 = llvm.mlir.constant(1 : index) : i64
    %21 = llvm.extractvalue %15[3] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.alloca %20 x !llvm.array<2 x i64> : (i64) -> !llvm.ptr
    llvm.store %21, %22 : !llvm.array<2 x i64>, !llvm.ptr
    %23 = llvm.getelementptr %22[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
    %24 = llvm.load %23 : !llvm.ptr -> i64
    %25 = llvm.mlir.constant(1 : index) : i64
    %26 = llvm.extractvalue %15[3] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
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
    %55 = llvm.extractvalue %15[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %56 = llvm.mul %55, %54  : i64
    %57 = llvm.extractvalue %15[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %58 = llvm.mul %56, %57  : i64
    %59 = llvm.mlir.zero : !llvm.ptr
    %60 = llvm.getelementptr %59[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %61 = llvm.ptrtoint %60 : !llvm.ptr to i64
    %62 = llvm.mul %58, %61  : i64
    %63 = llvm.extractvalue %15[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %64 = llvm.extractvalue %15[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %65 = llvm.getelementptr %63[%64] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %66 = llvm.getelementptr %44[%48] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%66, %65, %62) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb1(%17, %53 : i64, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb1(%67: i64, %68: !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>):  // 2 preds: ^bb0, ^bb11
    %69 = llvm.icmp "slt" %67, %19 : i64
    llvm.cond_br %69, ^bb2, ^bb12
  ^bb2:  // pred: ^bb1
    %70 = llvm.extractvalue %7[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %71 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %72 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %73 = llvm.insertvalue %70, %72[0] : !llvm.struct<(ptr, ptr, i64)> 
    %74 = llvm.insertvalue %71, %73[1] : !llvm.struct<(ptr, ptr, i64)> 
    %75 = llvm.mlir.constant(0 : index) : i64
    %76 = llvm.insertvalue %75, %74[2] : !llvm.struct<(ptr, ptr, i64)> 
    %77 = llvm.extractvalue %7[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %78 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %79 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %80 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %81 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %82 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %83 = llvm.insertvalue %70, %82[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %84 = llvm.insertvalue %71, %83[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %85 = llvm.insertvalue %67, %84[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %86 = llvm.insertvalue %19, %85[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %87 = llvm.insertvalue %80, %86[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %88 = llvm.extractvalue %68[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %89 = llvm.extractvalue %68[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %90 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %91 = llvm.insertvalue %88, %90[0] : !llvm.struct<(ptr, ptr, i64)> 
    %92 = llvm.insertvalue %89, %91[1] : !llvm.struct<(ptr, ptr, i64)> 
    %93 = llvm.mlir.constant(0 : index) : i64
    %94 = llvm.insertvalue %93, %92[2] : !llvm.struct<(ptr, ptr, i64)> 
    %95 = llvm.extractvalue %68[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %96 = llvm.extractvalue %68[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %97 = llvm.extractvalue %68[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %98 = llvm.extractvalue %68[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %99 = llvm.extractvalue %68[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %100 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %101 = llvm.insertvalue %88, %100[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %102 = llvm.insertvalue %89, %101[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %103 = llvm.mlir.constant(0 : index) : i64
    %104 = llvm.insertvalue %103, %102[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %105 = llvm.insertvalue %19, %104[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %106 = llvm.insertvalue %98, %105[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %107 = llvm.insertvalue %18, %106[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %108 = llvm.mlir.constant(1 : index) : i64
    %109 = llvm.insertvalue %108, %107[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %110 = llvm.mlir.constant(1 : index) : i64
    %111 = llvm.mul %18, %19  : i64
    %112 = llvm.mlir.zero : !llvm.ptr
    %113 = llvm.getelementptr %112[%111] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %114 = llvm.ptrtoint %113 : !llvm.ptr to i64
    %115 = llvm.mlir.constant(64 : index) : i64
    %116 = llvm.add %114, %115  : i64
    %117 = llvm.call @malloc(%116) : (i64) -> !llvm.ptr
    %118 = llvm.ptrtoint %117 : !llvm.ptr to i64
    %119 = llvm.mlir.constant(1 : index) : i64
    %120 = llvm.sub %115, %119  : i64
    %121 = llvm.add %118, %120  : i64
    %122 = llvm.urem %121, %115  : i64
    %123 = llvm.sub %121, %122  : i64
    %124 = llvm.inttoptr %123 : i64 to !llvm.ptr
    %125 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %126 = llvm.insertvalue %117, %125[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %127 = llvm.insertvalue %124, %126[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %128 = llvm.mlir.constant(0 : index) : i64
    %129 = llvm.insertvalue %128, %127[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %130 = llvm.insertvalue %19, %129[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %131 = llvm.insertvalue %18, %130[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %132 = llvm.insertvalue %18, %131[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %133 = llvm.insertvalue %110, %132[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %134 = llvm.intr.stacksave : !llvm.ptr
    %135 = llvm.mlir.constant(2 : i64) : i64
    %136 = llvm.mlir.constant(1 : index) : i64
    %137 = llvm.alloca %136 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %109, %137 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %138 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %139 = llvm.insertvalue %135, %138[0] : !llvm.struct<(i64, ptr)> 
    %140 = llvm.insertvalue %137, %139[1] : !llvm.struct<(i64, ptr)> 
    %141 = llvm.mlir.constant(2 : i64) : i64
    %142 = llvm.mlir.constant(1 : index) : i64
    %143 = llvm.alloca %142 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %133, %143 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %144 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %145 = llvm.insertvalue %141, %144[0] : !llvm.struct<(i64, ptr)> 
    %146 = llvm.insertvalue %143, %145[1] : !llvm.struct<(i64, ptr)> 
    %147 = llvm.mlir.constant(1 : index) : i64
    %148 = llvm.alloca %147 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %140, %148 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %149 = llvm.alloca %147 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %146, %149 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %150 = llvm.mlir.zero : !llvm.ptr
    %151 = llvm.getelementptr %150[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %152 = llvm.ptrtoint %151 : !llvm.ptr to i64
    llvm.call @memrefCopy(%152, %148, %149) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %134 : !llvm.ptr
    %153 = llvm.extractvalue %68[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %154 = llvm.extractvalue %68[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %155 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %156 = llvm.insertvalue %153, %155[0] : !llvm.struct<(ptr, ptr, i64)> 
    %157 = llvm.insertvalue %154, %156[1] : !llvm.struct<(ptr, ptr, i64)> 
    %158 = llvm.mlir.constant(0 : index) : i64
    %159 = llvm.insertvalue %158, %157[2] : !llvm.struct<(ptr, ptr, i64)> 
    %160 = llvm.extractvalue %68[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %161 = llvm.extractvalue %68[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %162 = llvm.extractvalue %68[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %163 = llvm.extractvalue %68[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %164 = llvm.extractvalue %68[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %165 = llvm.mul %67, %163  : i64
    %166 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %167 = llvm.insertvalue %153, %166[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %168 = llvm.insertvalue %154, %167[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %169 = llvm.insertvalue %165, %168[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %170 = llvm.insertvalue %18, %169[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %171 = llvm.mlir.constant(1 : index) : i64
    %172 = llvm.insertvalue %171, %170[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb3(%17 : i64)
  ^bb3(%173: i64):  // 2 preds: ^bb2, ^bb7
    %174 = llvm.icmp "slt" %173, %18 : i64
    llvm.cond_br %174, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%17 : i64)
  ^bb5(%175: i64):  // 2 preds: ^bb4, ^bb6
    %176 = llvm.icmp "slt" %175, %19 : i64
    llvm.cond_br %176, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %177 = llvm.getelementptr %71[%67] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %178 = llvm.mul %175, %80  : i64
    %179 = llvm.getelementptr %177[%178] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %180 = llvm.load %179 : !llvm.ptr -> f64
    %181 = llvm.mul %175, %18  : i64
    %182 = llvm.add %181, %173  : i64
    %183 = llvm.getelementptr %124[%182] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %184 = llvm.load %183 : !llvm.ptr -> f64
    %185 = llvm.getelementptr %154[%165] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %186 = llvm.getelementptr %185[%173] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %187 = llvm.load %186 : !llvm.ptr -> f64
    %188 = llvm.fmul %180, %184  : f64
    %189 = llvm.fadd %187, %188  : f64
    %190 = llvm.add %67, %16  : i64
    %191 = llvm.icmp "sge" %175, %190 : i64
    %192 = llvm.select %191, %189, %187 : i1, f64
    %193 = llvm.getelementptr %154[%165] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %194 = llvm.getelementptr %193[%173] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %192, %194 : f64, !llvm.ptr
    %195 = llvm.add %175, %16  : i64
    llvm.br ^bb5(%195 : i64)
  ^bb7:  // pred: ^bb5
    %196 = llvm.add %173, %16  : i64
    llvm.br ^bb3(%196 : i64)
  ^bb8:  // pred: ^bb3
    llvm.br ^bb9(%17 : i64)
  ^bb9(%197: i64):  // 2 preds: ^bb8, ^bb10
    %198 = llvm.icmp "slt" %197, %18 : i64
    llvm.cond_br %198, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    %199 = llvm.getelementptr %154[%165] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %200 = llvm.getelementptr %199[%197] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %201 = llvm.load %200 : !llvm.ptr -> f64
    %202 = llvm.fmul %arg2, %201  : f64
    %203 = llvm.getelementptr %154[%165] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %204 = llvm.getelementptr %203[%197] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %202, %204 : f64, !llvm.ptr
    %205 = llvm.add %197, %16  : i64
    llvm.br ^bb9(%205 : i64)
  ^bb11:  // pred: ^bb9
    %206 = llvm.extractvalue %68[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %207 = llvm.extractvalue %68[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %208 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %209 = llvm.insertvalue %206, %208[0] : !llvm.struct<(ptr, ptr, i64)> 
    %210 = llvm.insertvalue %207, %209[1] : !llvm.struct<(ptr, ptr, i64)> 
    %211 = llvm.mlir.constant(0 : index) : i64
    %212 = llvm.insertvalue %211, %210[2] : !llvm.struct<(ptr, ptr, i64)> 
    %213 = llvm.extractvalue %68[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %214 = llvm.extractvalue %68[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %215 = llvm.extractvalue %68[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %216 = llvm.extractvalue %68[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %217 = llvm.extractvalue %68[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %218 = llvm.mul %67, %216  : i64
    %219 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %220 = llvm.insertvalue %206, %219[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %221 = llvm.insertvalue %207, %220[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %222 = llvm.insertvalue %218, %221[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %223 = llvm.insertvalue %18, %222[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %224 = llvm.mlir.constant(1 : index) : i64
    %225 = llvm.insertvalue %224, %223[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %226 = llvm.intr.stacksave : !llvm.ptr
    %227 = llvm.mlir.constant(1 : i64) : i64
    %228 = llvm.mlir.constant(1 : index) : i64
    %229 = llvm.alloca %228 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %172, %229 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %230 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %231 = llvm.insertvalue %227, %230[0] : !llvm.struct<(i64, ptr)> 
    %232 = llvm.insertvalue %229, %231[1] : !llvm.struct<(i64, ptr)> 
    %233 = llvm.mlir.constant(1 : i64) : i64
    %234 = llvm.mlir.constant(1 : index) : i64
    %235 = llvm.alloca %234 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %225, %235 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
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
    %245 = llvm.add %67, %16  : i64
    llvm.br ^bb1(%245, %68 : i64, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb12:  // pred: ^bb1
    %246 = llvm.mlir.constant(1 : index) : i64
    %247 = llvm.extractvalue %68[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %248 = llvm.mul %247, %246  : i64
    %249 = llvm.extractvalue %68[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %250 = llvm.mul %248, %249  : i64
    %251 = llvm.mlir.zero : !llvm.ptr
    %252 = llvm.getelementptr %251[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %253 = llvm.ptrtoint %252 : !llvm.ptr to i64
    %254 = llvm.mul %250, %253  : i64
    %255 = llvm.extractvalue %68[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %256 = llvm.extractvalue %68[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %257 = llvm.getelementptr %255[%256] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %258 = llvm.extractvalue %15[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %259 = llvm.extractvalue %15[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %260 = llvm.getelementptr %258[%259] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%260, %257, %254) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.return
  }
}

