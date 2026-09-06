module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.func @memrefCopy(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @kernel_syr2k(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: f64, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: !llvm.ptr, %arg12: !llvm.ptr, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: !llvm.ptr, %arg19: !llvm.ptr, %arg20: i64, %arg21: i64, %arg22: i64, %arg23: i64, %arg24: i64) {
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
    %25 = llvm.mlir.constant(1 : index) : i64
    %26 = llvm.sext %arg1 : i32 to i64
    %27 = llvm.sext %arg0 : i32 to i64
    %28 = llvm.mlir.constant(1 : index) : i64
    %29 = llvm.extractvalue %7[3] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %30 = llvm.alloca %28 x !llvm.array<2 x i64> : (i64) -> !llvm.ptr
    llvm.store %29, %30 : !llvm.array<2 x i64>, !llvm.ptr
    %31 = llvm.getelementptr %30[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
    %32 = llvm.load %31 : !llvm.ptr -> i64
    %33 = llvm.mlir.constant(1 : index) : i64
    %34 = llvm.extractvalue %7[3] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %35 = llvm.alloca %33 x !llvm.array<2 x i64> : (i64) -> !llvm.ptr
    llvm.store %34, %35 : !llvm.array<2 x i64>, !llvm.ptr
    %36 = llvm.getelementptr %35[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
    %37 = llvm.load %36 : !llvm.ptr -> i64
    %38 = llvm.mlir.constant(1 : index) : i64
    %39 = llvm.mul %37, %32  : i64
    %40 = llvm.mlir.zero : !llvm.ptr
    %41 = llvm.getelementptr %40[%39] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %42 = llvm.ptrtoint %41 : !llvm.ptr to i64
    %43 = llvm.mlir.constant(64 : index) : i64
    %44 = llvm.add %42, %43  : i64
    %45 = llvm.call @malloc(%44) : (i64) -> !llvm.ptr
    %46 = llvm.ptrtoint %45 : !llvm.ptr to i64
    %47 = llvm.mlir.constant(1 : index) : i64
    %48 = llvm.sub %43, %47  : i64
    %49 = llvm.add %46, %48  : i64
    %50 = llvm.urem %49, %43  : i64
    %51 = llvm.sub %49, %50  : i64
    %52 = llvm.inttoptr %51 : i64 to !llvm.ptr
    %53 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %54 = llvm.insertvalue %45, %53[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %55 = llvm.insertvalue %52, %54[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %56 = llvm.mlir.constant(0 : index) : i64
    %57 = llvm.insertvalue %56, %55[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %58 = llvm.insertvalue %32, %57[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %59 = llvm.insertvalue %37, %58[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %60 = llvm.insertvalue %37, %59[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %61 = llvm.insertvalue %38, %60[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %62 = llvm.mlir.constant(1 : index) : i64
    %63 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %64 = llvm.mul %63, %62  : i64
    %65 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %66 = llvm.mul %64, %65  : i64
    %67 = llvm.mlir.zero : !llvm.ptr
    %68 = llvm.getelementptr %67[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %69 = llvm.ptrtoint %68 : !llvm.ptr to i64
    %70 = llvm.mul %66, %69  : i64
    %71 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %72 = llvm.extractvalue %7[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %73 = llvm.getelementptr %71[%72] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %74 = llvm.getelementptr %52[%56] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%74, %73, %70) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb1(%24 : i64)
  ^bb1(%75: i64):  // 2 preds: ^bb0, ^bb5
    %76 = llvm.icmp "slt" %75, %32 : i64
    llvm.cond_br %76, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%24 : i64)
  ^bb3(%77: i64):  // 2 preds: ^bb2, ^bb4
    %78 = llvm.icmp "slt" %77, %37 : i64
    llvm.cond_br %78, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %79 = llvm.mul %75, %37  : i64
    %80 = llvm.add %79, %77  : i64
    %81 = llvm.getelementptr %52[%80] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %82 = llvm.load %81 : !llvm.ptr -> f64
    %83 = llvm.fmul %82, %arg3  : f64
    %84 = llvm.add %75, %25  : i64
    %85 = llvm.icmp "slt" %77, %84 : i64
    %86 = llvm.select %85, %83, %82 : i1, f64
    %87 = llvm.mul %75, %37  : i64
    %88 = llvm.add %87, %77  : i64
    %89 = llvm.getelementptr %52[%88] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %86, %89 : f64, !llvm.ptr
    %90 = llvm.add %77, %25  : i64
    llvm.br ^bb3(%90 : i64)
  ^bb5:  // pred: ^bb3
    %91 = llvm.add %75, %25  : i64
    llvm.br ^bb1(%91 : i64)
  ^bb6:  // pred: ^bb1
    %92 = llvm.extractvalue %15[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %93 = llvm.extractvalue %15[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %94 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %95 = llvm.insertvalue %92, %94[0] : !llvm.struct<(ptr, ptr, i64)> 
    %96 = llvm.insertvalue %93, %95[1] : !llvm.struct<(ptr, ptr, i64)> 
    %97 = llvm.mlir.constant(0 : index) : i64
    %98 = llvm.insertvalue %97, %96[2] : !llvm.struct<(ptr, ptr, i64)> 
    %99 = llvm.extractvalue %15[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %100 = llvm.extractvalue %15[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %101 = llvm.extractvalue %15[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %102 = llvm.extractvalue %15[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %103 = llvm.extractvalue %15[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %104 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %105 = llvm.insertvalue %92, %104[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %106 = llvm.insertvalue %93, %105[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %107 = llvm.mlir.constant(0 : index) : i64
    %108 = llvm.insertvalue %107, %106[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %109 = llvm.insertvalue %27, %108[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %110 = llvm.insertvalue %102, %109[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %111 = llvm.insertvalue %26, %110[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %112 = llvm.mlir.constant(1 : index) : i64
    %113 = llvm.insertvalue %112, %111[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %114 = llvm.extractvalue %23[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %115 = llvm.extractvalue %23[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %116 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %117 = llvm.insertvalue %114, %116[0] : !llvm.struct<(ptr, ptr, i64)> 
    %118 = llvm.insertvalue %115, %117[1] : !llvm.struct<(ptr, ptr, i64)> 
    %119 = llvm.mlir.constant(0 : index) : i64
    %120 = llvm.insertvalue %119, %118[2] : !llvm.struct<(ptr, ptr, i64)> 
    %121 = llvm.extractvalue %23[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %122 = llvm.extractvalue %23[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %123 = llvm.extractvalue %23[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %124 = llvm.extractvalue %23[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %125 = llvm.extractvalue %23[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %126 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %127 = llvm.insertvalue %114, %126[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %128 = llvm.insertvalue %115, %127[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %129 = llvm.mlir.constant(0 : index) : i64
    %130 = llvm.insertvalue %129, %128[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %131 = llvm.insertvalue %27, %130[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %132 = llvm.insertvalue %124, %131[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %133 = llvm.insertvalue %26, %132[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %134 = llvm.mlir.constant(1 : index) : i64
    %135 = llvm.insertvalue %134, %133[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %136 = llvm.extractvalue %23[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %137 = llvm.extractvalue %23[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %138 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %139 = llvm.insertvalue %136, %138[0] : !llvm.struct<(ptr, ptr, i64)> 
    %140 = llvm.insertvalue %137, %139[1] : !llvm.struct<(ptr, ptr, i64)> 
    %141 = llvm.mlir.constant(0 : index) : i64
    %142 = llvm.insertvalue %141, %140[2] : !llvm.struct<(ptr, ptr, i64)> 
    %143 = llvm.extractvalue %23[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %144 = llvm.extractvalue %23[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %145 = llvm.extractvalue %23[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %146 = llvm.extractvalue %23[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %147 = llvm.extractvalue %23[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %148 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %149 = llvm.insertvalue %136, %148[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %150 = llvm.insertvalue %137, %149[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %151 = llvm.mlir.constant(0 : index) : i64
    %152 = llvm.insertvalue %151, %150[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %153 = llvm.insertvalue %27, %152[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %154 = llvm.insertvalue %146, %153[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %155 = llvm.insertvalue %26, %154[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %156 = llvm.mlir.constant(1 : index) : i64
    %157 = llvm.insertvalue %156, %155[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %158 = llvm.extractvalue %15[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %159 = llvm.extractvalue %15[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %160 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %161 = llvm.insertvalue %158, %160[0] : !llvm.struct<(ptr, ptr, i64)> 
    %162 = llvm.insertvalue %159, %161[1] : !llvm.struct<(ptr, ptr, i64)> 
    %163 = llvm.mlir.constant(0 : index) : i64
    %164 = llvm.insertvalue %163, %162[2] : !llvm.struct<(ptr, ptr, i64)> 
    %165 = llvm.extractvalue %15[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %166 = llvm.extractvalue %15[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %167 = llvm.extractvalue %15[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %168 = llvm.extractvalue %15[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %169 = llvm.extractvalue %15[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %170 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %171 = llvm.insertvalue %158, %170[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %172 = llvm.insertvalue %159, %171[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %173 = llvm.mlir.constant(0 : index) : i64
    %174 = llvm.insertvalue %173, %172[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %175 = llvm.insertvalue %27, %174[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %176 = llvm.insertvalue %168, %175[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %177 = llvm.insertvalue %26, %176[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %178 = llvm.mlir.constant(1 : index) : i64
    %179 = llvm.insertvalue %178, %177[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %180 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %181 = llvm.insertvalue %45, %180[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %182 = llvm.insertvalue %52, %181[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %183 = llvm.mlir.constant(0 : index) : i64
    %184 = llvm.insertvalue %183, %182[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %185 = llvm.insertvalue %27, %184[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %186 = llvm.insertvalue %37, %185[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %187 = llvm.insertvalue %27, %186[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %188 = llvm.mlir.constant(1 : index) : i64
    %189 = llvm.insertvalue %188, %187[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb7(%24 : i64)
  ^bb7(%190: i64):  // 2 preds: ^bb6, ^bb14
    %191 = llvm.icmp "slt" %190, %27 : i64
    llvm.cond_br %191, ^bb8, ^bb15
  ^bb8:  // pred: ^bb7
    llvm.br ^bb9(%24 : i64)
  ^bb9(%192: i64):  // 2 preds: ^bb8, ^bb13
    %193 = llvm.icmp "slt" %192, %26 : i64
    llvm.cond_br %193, ^bb10, ^bb14
  ^bb10:  // pred: ^bb9
    llvm.br ^bb11(%24 : i64)
  ^bb11(%194: i64):  // 2 preds: ^bb10, ^bb12
    %195 = llvm.icmp "slt" %194, %27 : i64
    llvm.cond_br %195, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    %196 = llvm.getelementptr %93[%107] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %197 = llvm.mul %194, %102  : i64
    %198 = llvm.add %197, %192  : i64
    %199 = llvm.getelementptr %196[%198] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %200 = llvm.load %199 : !llvm.ptr -> f64
    %201 = llvm.getelementptr %115[%129] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %202 = llvm.mul %190, %124  : i64
    %203 = llvm.add %202, %192  : i64
    %204 = llvm.getelementptr %201[%203] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %205 = llvm.load %204 : !llvm.ptr -> f64
    %206 = llvm.getelementptr %137[%151] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %207 = llvm.mul %194, %146  : i64
    %208 = llvm.add %207, %192  : i64
    %209 = llvm.getelementptr %206[%208] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %210 = llvm.load %209 : !llvm.ptr -> f64
    %211 = llvm.getelementptr %159[%173] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %212 = llvm.mul %190, %168  : i64
    %213 = llvm.add %212, %192  : i64
    %214 = llvm.getelementptr %211[%213] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %215 = llvm.load %214 : !llvm.ptr -> f64
    %216 = llvm.getelementptr %52[%183] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %217 = llvm.mul %190, %37  : i64
    %218 = llvm.add %217, %194  : i64
    %219 = llvm.getelementptr %216[%218] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %220 = llvm.load %219 : !llvm.ptr -> f64
    %221 = llvm.fmul %200, %arg2  : f64
    %222 = llvm.fmul %221, %205  : f64
    %223 = llvm.fmul %210, %arg2  : f64
    %224 = llvm.fmul %223, %215  : f64
    %225 = llvm.fadd %222, %224  : f64
    %226 = llvm.fadd %220, %225  : f64
    %227 = llvm.add %190, %25  : i64
    %228 = llvm.icmp "slt" %194, %227 : i64
    %229 = llvm.select %228, %226, %220 : i1, f64
    %230 = llvm.getelementptr %52[%183] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %231 = llvm.mul %190, %37  : i64
    %232 = llvm.add %231, %194  : i64
    %233 = llvm.getelementptr %230[%232] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %229, %233 : f64, !llvm.ptr
    %234 = llvm.add %194, %25  : i64
    llvm.br ^bb11(%234 : i64)
  ^bb13:  // pred: ^bb11
    %235 = llvm.add %192, %25  : i64
    llvm.br ^bb9(%235 : i64)
  ^bb14:  // pred: ^bb9
    %236 = llvm.add %190, %25  : i64
    llvm.br ^bb7(%236 : i64)
  ^bb15:  // pred: ^bb7
    %237 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %238 = llvm.insertvalue %45, %237[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %239 = llvm.insertvalue %52, %238[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %240 = llvm.mlir.constant(0 : index) : i64
    %241 = llvm.insertvalue %240, %239[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %242 = llvm.insertvalue %27, %241[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %243 = llvm.insertvalue %37, %242[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %244 = llvm.insertvalue %27, %243[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %245 = llvm.mlir.constant(1 : index) : i64
    %246 = llvm.insertvalue %245, %244[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %247 = llvm.intr.stacksave : !llvm.ptr
    %248 = llvm.mlir.constant(2 : i64) : i64
    %249 = llvm.mlir.constant(1 : index) : i64
    %250 = llvm.alloca %249 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %189, %250 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %251 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %252 = llvm.insertvalue %248, %251[0] : !llvm.struct<(i64, ptr)> 
    %253 = llvm.insertvalue %250, %252[1] : !llvm.struct<(i64, ptr)> 
    %254 = llvm.mlir.constant(2 : i64) : i64
    %255 = llvm.mlir.constant(1 : index) : i64
    %256 = llvm.alloca %255 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %246, %256 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %257 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %258 = llvm.insertvalue %254, %257[0] : !llvm.struct<(i64, ptr)> 
    %259 = llvm.insertvalue %256, %258[1] : !llvm.struct<(i64, ptr)> 
    %260 = llvm.mlir.constant(1 : index) : i64
    %261 = llvm.alloca %260 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %253, %261 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %262 = llvm.alloca %260 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %259, %262 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %263 = llvm.mlir.zero : !llvm.ptr
    %264 = llvm.getelementptr %263[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %265 = llvm.ptrtoint %264 : !llvm.ptr to i64
    llvm.call @memrefCopy(%265, %261, %262) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %247 : !llvm.ptr
    %266 = llvm.mlir.constant(1 : index) : i64
    %267 = llvm.mul %32, %266  : i64
    %268 = llvm.mul %267, %37  : i64
    %269 = llvm.mlir.zero : !llvm.ptr
    %270 = llvm.getelementptr %269[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %271 = llvm.ptrtoint %270 : !llvm.ptr to i64
    %272 = llvm.mul %268, %271  : i64
    %273 = llvm.getelementptr %52[%56] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %274 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %275 = llvm.extractvalue %7[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %276 = llvm.getelementptr %274[%275] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%276, %273, %272) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.return
  }
}

