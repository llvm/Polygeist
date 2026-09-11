module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.func @memrefCopy(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @kernel_doitgen(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: !llvm.ptr, %arg13: !llvm.ptr, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: !llvm.ptr, %arg20: !llvm.ptr, %arg21: i64, %arg22: i64, %arg23: i64) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %1 = llvm.insertvalue %arg3, %0[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2 = llvm.insertvalue %arg4, %1[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %3 = llvm.insertvalue %arg5, %2[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %4 = llvm.insertvalue %arg6, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %5 = llvm.insertvalue %arg9, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %6 = llvm.insertvalue %arg7, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %7 = llvm.insertvalue %arg10, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %8 = llvm.insertvalue %arg8, %7[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %9 = llvm.insertvalue %arg11, %8[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %10 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %11 = llvm.insertvalue %arg12, %10[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %12 = llvm.insertvalue %arg13, %11[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.insertvalue %arg14, %12[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.insertvalue %arg15, %13[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.insertvalue %arg17, %14[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.insertvalue %arg16, %15[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %17 = llvm.insertvalue %arg18, %16[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %19 = llvm.insertvalue %arg19, %18[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %20 = llvm.insertvalue %arg20, %19[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %21 = llvm.insertvalue %arg21, %20[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %22 = llvm.insertvalue %arg22, %21[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %23 = llvm.insertvalue %arg23, %22[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %24 = llvm.mlir.constant(2 : index) : i64
    %25 = llvm.mlir.constant(1 : index) : i64
    %26 = llvm.mlir.constant(0 : index) : i64
    %27 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %28 = llvm.sext %arg1 : i32 to i64
    %29 = llvm.sext %arg2 : i32 to i64
    %30 = llvm.sext %arg0 : i32 to i64
    %31 = llvm.mlir.constant(1 : index) : i64
    %32 = llvm.extractvalue %23[3] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %33 = llvm.alloca %31 x !llvm.array<1 x i64> : (i64) -> !llvm.ptr
    llvm.store %32, %33 : !llvm.array<1 x i64>, !llvm.ptr
    %34 = llvm.getelementptr %33[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x i64>
    %35 = llvm.load %34 : !llvm.ptr -> i64
    %36 = llvm.mlir.constant(1 : index) : i64
    %37 = llvm.mlir.zero : !llvm.ptr
    %38 = llvm.getelementptr %37[%35] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %39 = llvm.ptrtoint %38 : !llvm.ptr to i64
    %40 = llvm.mlir.constant(64 : index) : i64
    %41 = llvm.add %39, %40  : i64
    %42 = llvm.call @malloc(%41) : (i64) -> !llvm.ptr
    %43 = llvm.ptrtoint %42 : !llvm.ptr to i64
    %44 = llvm.mlir.constant(1 : index) : i64
    %45 = llvm.sub %40, %44  : i64
    %46 = llvm.add %43, %45  : i64
    %47 = llvm.urem %46, %40  : i64
    %48 = llvm.sub %46, %47  : i64
    %49 = llvm.inttoptr %48 : i64 to !llvm.ptr
    %50 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %51 = llvm.insertvalue %42, %50[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %52 = llvm.insertvalue %49, %51[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %53 = llvm.mlir.constant(0 : index) : i64
    %54 = llvm.insertvalue %53, %52[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %55 = llvm.insertvalue %35, %54[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %56 = llvm.insertvalue %36, %55[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %57 = llvm.mlir.constant(1 : index) : i64
    %58 = llvm.extractvalue %23[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %59 = llvm.mul %58, %57  : i64
    %60 = llvm.mlir.zero : !llvm.ptr
    %61 = llvm.getelementptr %60[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %62 = llvm.ptrtoint %61 : !llvm.ptr to i64
    %63 = llvm.mul %59, %62  : i64
    %64 = llvm.extractvalue %23[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %65 = llvm.extractvalue %23[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %66 = llvm.getelementptr %64[%65] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %67 = llvm.getelementptr %49[%53] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%67, %66, %63) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %68 = llvm.mlir.constant(1 : index) : i64
    %69 = llvm.extractvalue %9[3] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %70 = llvm.alloca %68 x !llvm.array<3 x i64> : (i64) -> !llvm.ptr
    llvm.store %69, %70 : !llvm.array<3 x i64>, !llvm.ptr
    %71 = llvm.getelementptr %70[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x i64>
    %72 = llvm.load %71 : !llvm.ptr -> i64
    %73 = llvm.mlir.constant(1 : index) : i64
    %74 = llvm.extractvalue %9[3] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %75 = llvm.alloca %73 x !llvm.array<3 x i64> : (i64) -> !llvm.ptr
    llvm.store %74, %75 : !llvm.array<3 x i64>, !llvm.ptr
    %76 = llvm.getelementptr %75[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x i64>
    %77 = llvm.load %76 : !llvm.ptr -> i64
    %78 = llvm.mlir.constant(1 : index) : i64
    %79 = llvm.extractvalue %9[3] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %80 = llvm.alloca %78 x !llvm.array<3 x i64> : (i64) -> !llvm.ptr
    llvm.store %79, %80 : !llvm.array<3 x i64>, !llvm.ptr
    %81 = llvm.getelementptr %80[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x i64>
    %82 = llvm.load %81 : !llvm.ptr -> i64
    %83 = llvm.mlir.constant(1 : index) : i64
    %84 = llvm.mul %82, %77  : i64
    %85 = llvm.mul %84, %72  : i64
    %86 = llvm.mlir.zero : !llvm.ptr
    %87 = llvm.getelementptr %86[%85] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %88 = llvm.ptrtoint %87 : !llvm.ptr to i64
    %89 = llvm.mlir.constant(64 : index) : i64
    %90 = llvm.add %88, %89  : i64
    %91 = llvm.call @malloc(%90) : (i64) -> !llvm.ptr
    %92 = llvm.ptrtoint %91 : !llvm.ptr to i64
    %93 = llvm.mlir.constant(1 : index) : i64
    %94 = llvm.sub %89, %93  : i64
    %95 = llvm.add %92, %94  : i64
    %96 = llvm.urem %95, %89  : i64
    %97 = llvm.sub %95, %96  : i64
    %98 = llvm.inttoptr %97 : i64 to !llvm.ptr
    %99 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %100 = llvm.insertvalue %91, %99[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %101 = llvm.insertvalue %98, %100[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %102 = llvm.mlir.constant(0 : index) : i64
    %103 = llvm.insertvalue %102, %101[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %104 = llvm.insertvalue %72, %103[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %105 = llvm.insertvalue %77, %104[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %106 = llvm.insertvalue %82, %105[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %107 = llvm.insertvalue %84, %106[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %108 = llvm.insertvalue %82, %107[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %109 = llvm.insertvalue %83, %108[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %110 = llvm.mlir.constant(1 : index) : i64
    %111 = llvm.extractvalue %9[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %112 = llvm.mul %111, %110  : i64
    %113 = llvm.extractvalue %9[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %114 = llvm.mul %112, %113  : i64
    %115 = llvm.extractvalue %9[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %116 = llvm.mul %114, %115  : i64
    %117 = llvm.mlir.zero : !llvm.ptr
    %118 = llvm.getelementptr %117[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %119 = llvm.ptrtoint %118 : !llvm.ptr to i64
    %120 = llvm.mul %116, %119  : i64
    %121 = llvm.extractvalue %9[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %122 = llvm.extractvalue %9[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %123 = llvm.getelementptr %121[%122] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %124 = llvm.getelementptr %98[%102] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%124, %123, %120) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb1(%26, %56, %109 : i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>)
  ^bb1(%125: i64, %126: !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, %127: !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>):  // 2 preds: ^bb0, ^bb17
    %128 = llvm.icmp "slt" %125, %30 : i64
    llvm.cond_br %128, ^bb2, ^bb18
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%26, %126, %127 : i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>)
  ^bb3(%129: i64, %130: !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, %131: !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>):  // 2 preds: ^bb2, ^bb16
    %132 = llvm.icmp "slt" %129, %28 : i64
    llvm.cond_br %132, ^bb4, ^bb17
  ^bb4:  // pred: ^bb3
    %133 = llvm.mlir.constant(1 : index) : i64
    %134 = llvm.extractvalue %130[3] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %135 = llvm.alloca %133 x !llvm.array<1 x i64> : (i64) -> !llvm.ptr
    llvm.store %134, %135 : !llvm.array<1 x i64>, !llvm.ptr
    %136 = llvm.getelementptr %135[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x i64>
    %137 = llvm.load %136 : !llvm.ptr -> i64
    llvm.br ^bb5(%26 : i64)
  ^bb5(%138: i64):  // 2 preds: ^bb4, ^bb6
    %139 = llvm.icmp "slt" %138, %137 : i64
    llvm.cond_br %139, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %140 = llvm.extractvalue %130[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %141 = llvm.getelementptr %140[%138] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %27, %141 : f64, !llvm.ptr
    %142 = llvm.add %138, %25  : i64
    llvm.br ^bb5(%142 : i64)
  ^bb7:  // pred: ^bb5
    %143 = llvm.extractvalue %131[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %144 = llvm.extractvalue %131[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %145 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %146 = llvm.insertvalue %143, %145[0] : !llvm.struct<(ptr, ptr, i64)> 
    %147 = llvm.insertvalue %144, %146[1] : !llvm.struct<(ptr, ptr, i64)> 
    %148 = llvm.mlir.constant(0 : index) : i64
    %149 = llvm.insertvalue %148, %147[2] : !llvm.struct<(ptr, ptr, i64)> 
    %150 = llvm.extractvalue %131[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %151 = llvm.extractvalue %131[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %152 = llvm.extractvalue %131[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %153 = llvm.extractvalue %131[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %154 = llvm.extractvalue %131[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %155 = llvm.extractvalue %131[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %156 = llvm.extractvalue %131[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %157 = llvm.mul %125, %154  : i64
    %158 = llvm.mul %129, %155  : i64
    %159 = llvm.add %157, %158  : i64
    %160 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %161 = llvm.insertvalue %143, %160[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %162 = llvm.insertvalue %144, %161[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %163 = llvm.insertvalue %159, %162[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %164 = llvm.insertvalue %29, %163[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %165 = llvm.mlir.constant(1 : index) : i64
    %166 = llvm.insertvalue %165, %164[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %167 = llvm.extractvalue %17[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %168 = llvm.extractvalue %17[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %169 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %170 = llvm.insertvalue %167, %169[0] : !llvm.struct<(ptr, ptr, i64)> 
    %171 = llvm.insertvalue %168, %170[1] : !llvm.struct<(ptr, ptr, i64)> 
    %172 = llvm.mlir.constant(0 : index) : i64
    %173 = llvm.insertvalue %172, %171[2] : !llvm.struct<(ptr, ptr, i64)> 
    %174 = llvm.extractvalue %17[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %175 = llvm.extractvalue %17[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %176 = llvm.extractvalue %17[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %177 = llvm.extractvalue %17[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %178 = llvm.extractvalue %17[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %179 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %180 = llvm.insertvalue %167, %179[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %181 = llvm.insertvalue %168, %180[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %182 = llvm.mlir.constant(0 : index) : i64
    %183 = llvm.insertvalue %182, %181[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %184 = llvm.insertvalue %29, %183[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %185 = llvm.insertvalue %177, %184[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %186 = llvm.insertvalue %29, %185[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %187 = llvm.mlir.constant(1 : index) : i64
    %188 = llvm.insertvalue %187, %186[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %189 = llvm.extractvalue %130[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %190 = llvm.extractvalue %130[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %191 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %192 = llvm.insertvalue %189, %191[0] : !llvm.struct<(ptr, ptr, i64)> 
    %193 = llvm.insertvalue %190, %192[1] : !llvm.struct<(ptr, ptr, i64)> 
    %194 = llvm.mlir.constant(0 : index) : i64
    %195 = llvm.insertvalue %194, %193[2] : !llvm.struct<(ptr, ptr, i64)> 
    %196 = llvm.extractvalue %130[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %197 = llvm.extractvalue %130[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %198 = llvm.extractvalue %130[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %199 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %200 = llvm.insertvalue %189, %199[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %201 = llvm.insertvalue %190, %200[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %202 = llvm.mlir.constant(0 : index) : i64
    %203 = llvm.insertvalue %202, %201[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %204 = llvm.insertvalue %29, %203[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %205 = llvm.mlir.constant(1 : index) : i64
    %206 = llvm.insertvalue %205, %204[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb8(%26 : i64)
  ^bb8(%207: i64):  // 2 preds: ^bb7, ^bb12
    %208 = llvm.icmp "slt" %207, %29 : i64
    llvm.cond_br %208, ^bb9, ^bb13
  ^bb9:  // pred: ^bb8
    llvm.br ^bb10(%26 : i64)
  ^bb10(%209: i64):  // 2 preds: ^bb9, ^bb11
    %210 = llvm.icmp "slt" %209, %29 : i64
    llvm.cond_br %210, ^bb11, ^bb12
  ^bb11:  // pred: ^bb10
    %211 = llvm.getelementptr %144[%159] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %212 = llvm.getelementptr %211[%209] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %213 = llvm.load %212 : !llvm.ptr -> f64
    %214 = llvm.getelementptr %168[%182] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %215 = llvm.mul %209, %177  : i64
    %216 = llvm.add %215, %207  : i64
    %217 = llvm.getelementptr %214[%216] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %218 = llvm.load %217 : !llvm.ptr -> f64
    %219 = llvm.getelementptr %190[%207] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %220 = llvm.load %219 : !llvm.ptr -> f64
    %221 = llvm.fmul %213, %218  : f64
    %222 = llvm.fadd %220, %221  : f64
    %223 = llvm.getelementptr %190[%207] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %222, %223 : f64, !llvm.ptr
    %224 = llvm.add %209, %25  : i64
    llvm.br ^bb10(%224 : i64)
  ^bb12:  // pred: ^bb10
    %225 = llvm.add %207, %25  : i64
    llvm.br ^bb8(%225 : i64)
  ^bb13:  // pred: ^bb8
    %226 = llvm.extractvalue %130[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %227 = llvm.extractvalue %130[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %228 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %229 = llvm.insertvalue %226, %228[0] : !llvm.struct<(ptr, ptr, i64)> 
    %230 = llvm.insertvalue %227, %229[1] : !llvm.struct<(ptr, ptr, i64)> 
    %231 = llvm.mlir.constant(0 : index) : i64
    %232 = llvm.insertvalue %231, %230[2] : !llvm.struct<(ptr, ptr, i64)> 
    %233 = llvm.extractvalue %130[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %234 = llvm.extractvalue %130[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %235 = llvm.extractvalue %130[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %236 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %237 = llvm.insertvalue %226, %236[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %238 = llvm.insertvalue %227, %237[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %239 = llvm.mlir.constant(0 : index) : i64
    %240 = llvm.insertvalue %239, %238[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %241 = llvm.insertvalue %29, %240[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %242 = llvm.mlir.constant(1 : index) : i64
    %243 = llvm.insertvalue %242, %241[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %244 = llvm.mlir.constant(1 : index) : i64
    %245 = llvm.mul %29, %244  : i64
    %246 = llvm.mlir.zero : !llvm.ptr
    %247 = llvm.getelementptr %246[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %248 = llvm.ptrtoint %247 : !llvm.ptr to i64
    %249 = llvm.mul %245, %248  : i64
    %250 = llvm.getelementptr %190[%202] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %251 = llvm.getelementptr %227[%239] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%251, %250, %249) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %252 = llvm.extractvalue %131[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %253 = llvm.extractvalue %131[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %254 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %255 = llvm.insertvalue %252, %254[0] : !llvm.struct<(ptr, ptr, i64)> 
    %256 = llvm.insertvalue %253, %255[1] : !llvm.struct<(ptr, ptr, i64)> 
    %257 = llvm.mlir.constant(0 : index) : i64
    %258 = llvm.insertvalue %257, %256[2] : !llvm.struct<(ptr, ptr, i64)> 
    %259 = llvm.extractvalue %131[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %260 = llvm.extractvalue %131[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %261 = llvm.extractvalue %131[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %262 = llvm.extractvalue %131[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %263 = llvm.extractvalue %131[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %264 = llvm.extractvalue %131[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %265 = llvm.extractvalue %131[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %266 = llvm.mul %125, %263  : i64
    %267 = llvm.mul %129, %264  : i64
    %268 = llvm.add %266, %267  : i64
    %269 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %270 = llvm.insertvalue %252, %269[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %271 = llvm.insertvalue %253, %270[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %272 = llvm.insertvalue %268, %271[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %273 = llvm.insertvalue %29, %272[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %274 = llvm.mlir.constant(1 : index) : i64
    %275 = llvm.insertvalue %274, %273[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb14(%26 : i64)
  ^bb14(%276: i64):  // 2 preds: ^bb13, ^bb15
    %277 = llvm.icmp "slt" %276, %29 : i64
    llvm.cond_br %277, ^bb15, ^bb16
  ^bb15:  // pred: ^bb14
    %278 = llvm.getelementptr %190[%276] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %279 = llvm.load %278 : !llvm.ptr -> f64
    %280 = llvm.getelementptr %253[%268] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %281 = llvm.getelementptr %280[%276] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %279, %281 : f64, !llvm.ptr
    %282 = llvm.add %276, %25  : i64
    llvm.br ^bb14(%282 : i64)
  ^bb16:  // pred: ^bb14
    %283 = llvm.extractvalue %131[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %284 = llvm.extractvalue %131[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %285 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %286 = llvm.insertvalue %283, %285[0] : !llvm.struct<(ptr, ptr, i64)> 
    %287 = llvm.insertvalue %284, %286[1] : !llvm.struct<(ptr, ptr, i64)> 
    %288 = llvm.mlir.constant(0 : index) : i64
    %289 = llvm.insertvalue %288, %287[2] : !llvm.struct<(ptr, ptr, i64)> 
    %290 = llvm.extractvalue %131[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %291 = llvm.extractvalue %131[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %292 = llvm.extractvalue %131[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %293 = llvm.extractvalue %131[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %294 = llvm.extractvalue %131[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %295 = llvm.extractvalue %131[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %296 = llvm.extractvalue %131[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %297 = llvm.mul %125, %294  : i64
    %298 = llvm.mul %129, %295  : i64
    %299 = llvm.add %297, %298  : i64
    %300 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %301 = llvm.insertvalue %283, %300[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %302 = llvm.insertvalue %284, %301[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %303 = llvm.insertvalue %299, %302[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %304 = llvm.insertvalue %29, %303[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %305 = llvm.mlir.constant(1 : index) : i64
    %306 = llvm.insertvalue %305, %304[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %307 = llvm.intr.stacksave : !llvm.ptr
    %308 = llvm.mlir.constant(1 : i64) : i64
    %309 = llvm.mlir.constant(1 : index) : i64
    %310 = llvm.alloca %309 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %275, %310 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %311 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %312 = llvm.insertvalue %308, %311[0] : !llvm.struct<(i64, ptr)> 
    %313 = llvm.insertvalue %310, %312[1] : !llvm.struct<(i64, ptr)> 
    %314 = llvm.mlir.constant(1 : i64) : i64
    %315 = llvm.mlir.constant(1 : index) : i64
    %316 = llvm.alloca %315 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %306, %316 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %317 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %318 = llvm.insertvalue %314, %317[0] : !llvm.struct<(i64, ptr)> 
    %319 = llvm.insertvalue %316, %318[1] : !llvm.struct<(i64, ptr)> 
    %320 = llvm.mlir.constant(1 : index) : i64
    %321 = llvm.alloca %320 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %313, %321 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %322 = llvm.alloca %320 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %319, %322 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %323 = llvm.mlir.zero : !llvm.ptr
    %324 = llvm.getelementptr %323[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %325 = llvm.ptrtoint %324 : !llvm.ptr to i64
    llvm.call @memrefCopy(%325, %321, %322) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %307 : !llvm.ptr
    %326 = llvm.add %129, %25  : i64
    llvm.br ^bb3(%326, %130, %131 : i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>)
  ^bb17:  // pred: ^bb3
    %327 = llvm.add %125, %25  : i64
    llvm.br ^bb1(%327, %130, %131 : i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>)
  ^bb18:  // pred: ^bb1
    %328 = llvm.mlir.constant(1 : index) : i64
    %329 = llvm.extractvalue %127[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %330 = llvm.mul %329, %328  : i64
    %331 = llvm.extractvalue %127[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %332 = llvm.mul %330, %331  : i64
    %333 = llvm.extractvalue %127[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %334 = llvm.mul %332, %333  : i64
    %335 = llvm.mlir.zero : !llvm.ptr
    %336 = llvm.getelementptr %335[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %337 = llvm.ptrtoint %336 : !llvm.ptr to i64
    %338 = llvm.mul %334, %337  : i64
    %339 = llvm.extractvalue %127[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %340 = llvm.extractvalue %127[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %341 = llvm.getelementptr %339[%340] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %342 = llvm.extractvalue %9[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %343 = llvm.extractvalue %9[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %344 = llvm.getelementptr %342[%343] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%344, %341, %338) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %345 = llvm.mlir.constant(1 : index) : i64
    %346 = llvm.extractvalue %126[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %347 = llvm.mul %346, %345  : i64
    %348 = llvm.mlir.zero : !llvm.ptr
    %349 = llvm.getelementptr %348[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %350 = llvm.ptrtoint %349 : !llvm.ptr to i64
    %351 = llvm.mul %347, %350  : i64
    %352 = llvm.extractvalue %126[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %353 = llvm.extractvalue %126[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %354 = llvm.getelementptr %352[%353] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %355 = llvm.extractvalue %23[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %356 = llvm.extractvalue %23[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %357 = llvm.getelementptr %355[%356] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%357, %354, %351) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.return
  }
}

