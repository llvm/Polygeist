module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.func @memrefCopy(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @kernel_jacobi_1d(%arg0: i32, %arg1: i32, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg2, %0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2 = llvm.insertvalue %arg3, %1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %3 = llvm.insertvalue %arg4, %2[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %4 = llvm.insertvalue %arg5, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %5 = llvm.insertvalue %arg6, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %6 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %7 = llvm.insertvalue %arg7, %6[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %8 = llvm.insertvalue %arg8, %7[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %9 = llvm.insertvalue %arg9, %8[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.insertvalue %arg10, %9[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.insertvalue %arg11, %10[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.mlir.constant(0 : index) : i64
    %13 = llvm.mlir.constant(-1 : index) : i64
    %14 = llvm.mlir.constant(3.333300e-01 : f64) : f64
    %15 = llvm.mlir.constant(1 : index) : i64
    %16 = llvm.sext %arg1 : i32 to i64
    %17 = llvm.sext %arg0 : i32 to i64
    %18 = llvm.add %16, %13  : i64
    %19 = llvm.add %16, %13  : i64
    %20 = llvm.sub %19, %15  : i64
    %21 = llvm.sub %18, %15  : i64
    %22 = llvm.mlir.constant(1 : index) : i64
    %23 = llvm.extractvalue %11[3] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %24 = llvm.alloca %22 x !llvm.array<1 x i64> : (i64) -> !llvm.ptr
    llvm.store %23, %24 : !llvm.array<1 x i64>, !llvm.ptr
    %25 = llvm.getelementptr %24[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x i64>
    %26 = llvm.load %25 : !llvm.ptr -> i64
    %27 = llvm.mlir.constant(1 : index) : i64
    %28 = llvm.mlir.zero : !llvm.ptr
    %29 = llvm.getelementptr %28[%26] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %30 = llvm.ptrtoint %29 : !llvm.ptr to i64
    %31 = llvm.mlir.constant(64 : index) : i64
    %32 = llvm.add %30, %31  : i64
    %33 = llvm.call @malloc(%32) : (i64) -> !llvm.ptr
    %34 = llvm.ptrtoint %33 : !llvm.ptr to i64
    %35 = llvm.mlir.constant(1 : index) : i64
    %36 = llvm.sub %31, %35  : i64
    %37 = llvm.add %34, %36  : i64
    %38 = llvm.urem %37, %31  : i64
    %39 = llvm.sub %37, %38  : i64
    %40 = llvm.inttoptr %39 : i64 to !llvm.ptr
    %41 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %42 = llvm.insertvalue %33, %41[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %43 = llvm.insertvalue %40, %42[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %44 = llvm.mlir.constant(0 : index) : i64
    %45 = llvm.insertvalue %44, %43[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %46 = llvm.insertvalue %26, %45[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %47 = llvm.insertvalue %27, %46[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %48 = llvm.mlir.constant(1 : index) : i64
    %49 = llvm.extractvalue %11[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %50 = llvm.mul %49, %48  : i64
    %51 = llvm.mlir.zero : !llvm.ptr
    %52 = llvm.getelementptr %51[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %53 = llvm.ptrtoint %52 : !llvm.ptr to i64
    %54 = llvm.mul %50, %53  : i64
    %55 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %56 = llvm.extractvalue %11[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %57 = llvm.getelementptr %55[%56] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %58 = llvm.getelementptr %40[%44] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%58, %57, %54) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %59 = llvm.mlir.constant(1 : index) : i64
    %60 = llvm.extractvalue %5[3] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %61 = llvm.alloca %59 x !llvm.array<1 x i64> : (i64) -> !llvm.ptr
    llvm.store %60, %61 : !llvm.array<1 x i64>, !llvm.ptr
    %62 = llvm.getelementptr %61[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x i64>
    %63 = llvm.load %62 : !llvm.ptr -> i64
    %64 = llvm.mlir.constant(1 : index) : i64
    %65 = llvm.mlir.zero : !llvm.ptr
    %66 = llvm.getelementptr %65[%63] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %67 = llvm.ptrtoint %66 : !llvm.ptr to i64
    %68 = llvm.mlir.constant(64 : index) : i64
    %69 = llvm.add %67, %68  : i64
    %70 = llvm.call @malloc(%69) : (i64) -> !llvm.ptr
    %71 = llvm.ptrtoint %70 : !llvm.ptr to i64
    %72 = llvm.mlir.constant(1 : index) : i64
    %73 = llvm.sub %68, %72  : i64
    %74 = llvm.add %71, %73  : i64
    %75 = llvm.urem %74, %68  : i64
    %76 = llvm.sub %74, %75  : i64
    %77 = llvm.inttoptr %76 : i64 to !llvm.ptr
    %78 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %79 = llvm.insertvalue %70, %78[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %80 = llvm.insertvalue %77, %79[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %81 = llvm.mlir.constant(0 : index) : i64
    %82 = llvm.insertvalue %81, %80[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %83 = llvm.insertvalue %63, %82[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %84 = llvm.insertvalue %64, %83[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %85 = llvm.mlir.constant(1 : index) : i64
    %86 = llvm.extractvalue %5[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %87 = llvm.mul %86, %85  : i64
    %88 = llvm.mlir.zero : !llvm.ptr
    %89 = llvm.getelementptr %88[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %90 = llvm.ptrtoint %89 : !llvm.ptr to i64
    %91 = llvm.mul %87, %90  : i64
    %92 = llvm.extractvalue %5[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %93 = llvm.extractvalue %5[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %94 = llvm.getelementptr %92[%93] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %95 = llvm.getelementptr %77[%81] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%95, %94, %91) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb1(%12, %47, %84 : i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)
  ^bb1(%96: i64, %97: !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, %98: !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>):  // 2 preds: ^bb0, ^bb8
    %99 = llvm.icmp "slt" %96, %17 : i64
    llvm.cond_br %99, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    %100 = llvm.extractvalue %98[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %101 = llvm.extractvalue %98[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %102 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %103 = llvm.insertvalue %100, %102[0] : !llvm.struct<(ptr, ptr, i64)> 
    %104 = llvm.insertvalue %101, %103[1] : !llvm.struct<(ptr, ptr, i64)> 
    %105 = llvm.mlir.constant(0 : index) : i64
    %106 = llvm.insertvalue %105, %104[2] : !llvm.struct<(ptr, ptr, i64)> 
    %107 = llvm.extractvalue %98[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %108 = llvm.extractvalue %98[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %109 = llvm.extractvalue %98[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %110 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %111 = llvm.insertvalue %100, %110[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %112 = llvm.insertvalue %101, %111[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %113 = llvm.mlir.constant(0 : index) : i64
    %114 = llvm.insertvalue %113, %112[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %115 = llvm.insertvalue %20, %114[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %116 = llvm.mlir.constant(1 : index) : i64
    %117 = llvm.insertvalue %116, %115[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %118 = llvm.extractvalue %98[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %119 = llvm.extractvalue %98[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %120 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %121 = llvm.insertvalue %118, %120[0] : !llvm.struct<(ptr, ptr, i64)> 
    %122 = llvm.insertvalue %119, %121[1] : !llvm.struct<(ptr, ptr, i64)> 
    %123 = llvm.mlir.constant(0 : index) : i64
    %124 = llvm.insertvalue %123, %122[2] : !llvm.struct<(ptr, ptr, i64)> 
    %125 = llvm.extractvalue %98[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %126 = llvm.extractvalue %98[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %127 = llvm.extractvalue %98[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %128 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %129 = llvm.insertvalue %118, %128[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %130 = llvm.insertvalue %119, %129[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %131 = llvm.mlir.constant(1 : index) : i64
    %132 = llvm.insertvalue %131, %130[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %133 = llvm.insertvalue %20, %132[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %134 = llvm.mlir.constant(1 : index) : i64
    %135 = llvm.insertvalue %134, %133[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %136 = llvm.extractvalue %98[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %137 = llvm.extractvalue %98[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %138 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %139 = llvm.insertvalue %136, %138[0] : !llvm.struct<(ptr, ptr, i64)> 
    %140 = llvm.insertvalue %137, %139[1] : !llvm.struct<(ptr, ptr, i64)> 
    %141 = llvm.mlir.constant(0 : index) : i64
    %142 = llvm.insertvalue %141, %140[2] : !llvm.struct<(ptr, ptr, i64)> 
    %143 = llvm.extractvalue %98[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %144 = llvm.extractvalue %98[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %145 = llvm.extractvalue %98[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %146 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %147 = llvm.insertvalue %136, %146[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %148 = llvm.insertvalue %137, %147[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %149 = llvm.mlir.constant(2 : index) : i64
    %150 = llvm.insertvalue %149, %148[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %151 = llvm.insertvalue %20, %150[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %152 = llvm.mlir.constant(1 : index) : i64
    %153 = llvm.insertvalue %152, %151[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %154 = llvm.extractvalue %97[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %155 = llvm.extractvalue %97[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %156 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %157 = llvm.insertvalue %154, %156[0] : !llvm.struct<(ptr, ptr, i64)> 
    %158 = llvm.insertvalue %155, %157[1] : !llvm.struct<(ptr, ptr, i64)> 
    %159 = llvm.mlir.constant(0 : index) : i64
    %160 = llvm.insertvalue %159, %158[2] : !llvm.struct<(ptr, ptr, i64)> 
    %161 = llvm.extractvalue %97[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %162 = llvm.extractvalue %97[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %163 = llvm.extractvalue %97[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %164 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %165 = llvm.insertvalue %154, %164[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %166 = llvm.insertvalue %155, %165[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %167 = llvm.mlir.constant(1 : index) : i64
    %168 = llvm.insertvalue %167, %166[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %169 = llvm.insertvalue %20, %168[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %170 = llvm.mlir.constant(1 : index) : i64
    %171 = llvm.insertvalue %170, %169[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb3(%12 : i64)
  ^bb3(%172: i64):  // 2 preds: ^bb2, ^bb4
    %173 = llvm.icmp "slt" %172, %20 : i64
    llvm.cond_br %173, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %174 = llvm.getelementptr %101[%172] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %175 = llvm.load %174 : !llvm.ptr -> f64
    %176 = llvm.mlir.constant(1 : index) : i64
    %177 = llvm.getelementptr %119[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %178 = llvm.getelementptr %177[%172] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %179 = llvm.load %178 : !llvm.ptr -> f64
    %180 = llvm.mlir.constant(2 : index) : i64
    %181 = llvm.getelementptr %137[2] : (!llvm.ptr) -> !llvm.ptr, f64
    %182 = llvm.getelementptr %181[%172] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %183 = llvm.load %182 : !llvm.ptr -> f64
    %184 = llvm.fadd %175, %179  : f64
    %185 = llvm.fadd %184, %183  : f64
    %186 = llvm.fmul %185, %14  : f64
    %187 = llvm.mlir.constant(1 : index) : i64
    %188 = llvm.getelementptr %155[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %189 = llvm.getelementptr %188[%172] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %186, %189 : f64, !llvm.ptr
    %190 = llvm.add %172, %15  : i64
    llvm.br ^bb3(%190 : i64)
  ^bb5:  // pred: ^bb3
    %191 = llvm.extractvalue %97[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %192 = llvm.extractvalue %97[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %193 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %194 = llvm.insertvalue %191, %193[0] : !llvm.struct<(ptr, ptr, i64)> 
    %195 = llvm.insertvalue %192, %194[1] : !llvm.struct<(ptr, ptr, i64)> 
    %196 = llvm.mlir.constant(0 : index) : i64
    %197 = llvm.insertvalue %196, %195[2] : !llvm.struct<(ptr, ptr, i64)> 
    %198 = llvm.extractvalue %97[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %199 = llvm.extractvalue %97[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %200 = llvm.extractvalue %97[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %201 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %202 = llvm.insertvalue %191, %201[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %203 = llvm.insertvalue %192, %202[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %204 = llvm.mlir.constant(1 : index) : i64
    %205 = llvm.insertvalue %204, %203[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %206 = llvm.insertvalue %20, %205[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %207 = llvm.mlir.constant(1 : index) : i64
    %208 = llvm.insertvalue %207, %206[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %209 = llvm.intr.stacksave : !llvm.ptr
    %210 = llvm.mlir.constant(1 : i64) : i64
    %211 = llvm.mlir.constant(1 : index) : i64
    %212 = llvm.alloca %211 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %171, %212 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %213 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %214 = llvm.insertvalue %210, %213[0] : !llvm.struct<(i64, ptr)> 
    %215 = llvm.insertvalue %212, %214[1] : !llvm.struct<(i64, ptr)> 
    %216 = llvm.mlir.constant(1 : i64) : i64
    %217 = llvm.mlir.constant(1 : index) : i64
    %218 = llvm.alloca %217 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %208, %218 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %219 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %220 = llvm.insertvalue %216, %219[0] : !llvm.struct<(i64, ptr)> 
    %221 = llvm.insertvalue %218, %220[1] : !llvm.struct<(i64, ptr)> 
    %222 = llvm.mlir.constant(1 : index) : i64
    %223 = llvm.alloca %222 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %215, %223 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %224 = llvm.alloca %222 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %221, %224 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %225 = llvm.mlir.zero : !llvm.ptr
    %226 = llvm.getelementptr %225[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %227 = llvm.ptrtoint %226 : !llvm.ptr to i64
    llvm.call @memrefCopy(%227, %223, %224) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %209 : !llvm.ptr
    %228 = llvm.extractvalue %97[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %229 = llvm.extractvalue %97[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %230 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %231 = llvm.insertvalue %228, %230[0] : !llvm.struct<(ptr, ptr, i64)> 
    %232 = llvm.insertvalue %229, %231[1] : !llvm.struct<(ptr, ptr, i64)> 
    %233 = llvm.mlir.constant(0 : index) : i64
    %234 = llvm.insertvalue %233, %232[2] : !llvm.struct<(ptr, ptr, i64)> 
    %235 = llvm.extractvalue %97[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %236 = llvm.extractvalue %97[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %237 = llvm.extractvalue %97[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %238 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %239 = llvm.insertvalue %228, %238[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %240 = llvm.insertvalue %229, %239[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %241 = llvm.mlir.constant(0 : index) : i64
    %242 = llvm.insertvalue %241, %240[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %243 = llvm.insertvalue %21, %242[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %244 = llvm.mlir.constant(1 : index) : i64
    %245 = llvm.insertvalue %244, %243[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %246 = llvm.extractvalue %97[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %247 = llvm.extractvalue %97[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %248 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %249 = llvm.insertvalue %246, %248[0] : !llvm.struct<(ptr, ptr, i64)> 
    %250 = llvm.insertvalue %247, %249[1] : !llvm.struct<(ptr, ptr, i64)> 
    %251 = llvm.mlir.constant(0 : index) : i64
    %252 = llvm.insertvalue %251, %250[2] : !llvm.struct<(ptr, ptr, i64)> 
    %253 = llvm.extractvalue %97[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %254 = llvm.extractvalue %97[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %255 = llvm.extractvalue %97[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %256 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %257 = llvm.insertvalue %246, %256[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %258 = llvm.insertvalue %247, %257[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %259 = llvm.mlir.constant(1 : index) : i64
    %260 = llvm.insertvalue %259, %258[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %261 = llvm.insertvalue %21, %260[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %262 = llvm.mlir.constant(1 : index) : i64
    %263 = llvm.insertvalue %262, %261[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %264 = llvm.extractvalue %97[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %265 = llvm.extractvalue %97[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %266 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %267 = llvm.insertvalue %264, %266[0] : !llvm.struct<(ptr, ptr, i64)> 
    %268 = llvm.insertvalue %265, %267[1] : !llvm.struct<(ptr, ptr, i64)> 
    %269 = llvm.mlir.constant(0 : index) : i64
    %270 = llvm.insertvalue %269, %268[2] : !llvm.struct<(ptr, ptr, i64)> 
    %271 = llvm.extractvalue %97[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %272 = llvm.extractvalue %97[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %273 = llvm.extractvalue %97[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %274 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %275 = llvm.insertvalue %264, %274[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %276 = llvm.insertvalue %265, %275[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %277 = llvm.mlir.constant(2 : index) : i64
    %278 = llvm.insertvalue %277, %276[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %279 = llvm.insertvalue %21, %278[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %280 = llvm.mlir.constant(1 : index) : i64
    %281 = llvm.insertvalue %280, %279[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %282 = llvm.extractvalue %98[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %283 = llvm.extractvalue %98[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %284 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %285 = llvm.insertvalue %282, %284[0] : !llvm.struct<(ptr, ptr, i64)> 
    %286 = llvm.insertvalue %283, %285[1] : !llvm.struct<(ptr, ptr, i64)> 
    %287 = llvm.mlir.constant(0 : index) : i64
    %288 = llvm.insertvalue %287, %286[2] : !llvm.struct<(ptr, ptr, i64)> 
    %289 = llvm.extractvalue %98[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %290 = llvm.extractvalue %98[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %291 = llvm.extractvalue %98[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %292 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %293 = llvm.insertvalue %282, %292[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %294 = llvm.insertvalue %283, %293[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %295 = llvm.mlir.constant(1 : index) : i64
    %296 = llvm.insertvalue %295, %294[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %297 = llvm.insertvalue %21, %296[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %298 = llvm.mlir.constant(1 : index) : i64
    %299 = llvm.insertvalue %298, %297[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb6(%12 : i64)
  ^bb6(%300: i64):  // 2 preds: ^bb5, ^bb7
    %301 = llvm.icmp "slt" %300, %21 : i64
    llvm.cond_br %301, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    %302 = llvm.getelementptr %229[%300] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %303 = llvm.load %302 : !llvm.ptr -> f64
    %304 = llvm.mlir.constant(1 : index) : i64
    %305 = llvm.getelementptr %247[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %306 = llvm.getelementptr %305[%300] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %307 = llvm.load %306 : !llvm.ptr -> f64
    %308 = llvm.mlir.constant(2 : index) : i64
    %309 = llvm.getelementptr %265[2] : (!llvm.ptr) -> !llvm.ptr, f64
    %310 = llvm.getelementptr %309[%300] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %311 = llvm.load %310 : !llvm.ptr -> f64
    %312 = llvm.fadd %303, %307  : f64
    %313 = llvm.fadd %312, %311  : f64
    %314 = llvm.fmul %313, %14  : f64
    %315 = llvm.mlir.constant(1 : index) : i64
    %316 = llvm.getelementptr %283[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %317 = llvm.getelementptr %316[%300] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %314, %317 : f64, !llvm.ptr
    %318 = llvm.add %300, %15  : i64
    llvm.br ^bb6(%318 : i64)
  ^bb8:  // pred: ^bb6
    %319 = llvm.extractvalue %98[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %320 = llvm.extractvalue %98[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %321 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %322 = llvm.insertvalue %319, %321[0] : !llvm.struct<(ptr, ptr, i64)> 
    %323 = llvm.insertvalue %320, %322[1] : !llvm.struct<(ptr, ptr, i64)> 
    %324 = llvm.mlir.constant(0 : index) : i64
    %325 = llvm.insertvalue %324, %323[2] : !llvm.struct<(ptr, ptr, i64)> 
    %326 = llvm.extractvalue %98[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %327 = llvm.extractvalue %98[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %328 = llvm.extractvalue %98[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %329 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %330 = llvm.insertvalue %319, %329[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %331 = llvm.insertvalue %320, %330[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %332 = llvm.mlir.constant(1 : index) : i64
    %333 = llvm.insertvalue %332, %331[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %334 = llvm.insertvalue %21, %333[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %335 = llvm.mlir.constant(1 : index) : i64
    %336 = llvm.insertvalue %335, %334[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %337 = llvm.intr.stacksave : !llvm.ptr
    %338 = llvm.mlir.constant(1 : i64) : i64
    %339 = llvm.mlir.constant(1 : index) : i64
    %340 = llvm.alloca %339 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %299, %340 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %341 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %342 = llvm.insertvalue %338, %341[0] : !llvm.struct<(i64, ptr)> 
    %343 = llvm.insertvalue %340, %342[1] : !llvm.struct<(i64, ptr)> 
    %344 = llvm.mlir.constant(1 : i64) : i64
    %345 = llvm.mlir.constant(1 : index) : i64
    %346 = llvm.alloca %345 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %336, %346 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %347 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %348 = llvm.insertvalue %344, %347[0] : !llvm.struct<(i64, ptr)> 
    %349 = llvm.insertvalue %346, %348[1] : !llvm.struct<(i64, ptr)> 
    %350 = llvm.mlir.constant(1 : index) : i64
    %351 = llvm.alloca %350 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %343, %351 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %352 = llvm.alloca %350 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %349, %352 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %353 = llvm.mlir.zero : !llvm.ptr
    %354 = llvm.getelementptr %353[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %355 = llvm.ptrtoint %354 : !llvm.ptr to i64
    llvm.call @memrefCopy(%355, %351, %352) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %337 : !llvm.ptr
    %356 = llvm.add %96, %15  : i64
    llvm.br ^bb1(%356, %97, %98 : i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)
  ^bb9:  // pred: ^bb1
    %357 = llvm.mlir.constant(1 : index) : i64
    %358 = llvm.extractvalue %98[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %359 = llvm.mul %358, %357  : i64
    %360 = llvm.mlir.zero : !llvm.ptr
    %361 = llvm.getelementptr %360[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %362 = llvm.ptrtoint %361 : !llvm.ptr to i64
    %363 = llvm.mul %359, %362  : i64
    %364 = llvm.extractvalue %98[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %365 = llvm.extractvalue %98[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %366 = llvm.getelementptr %364[%365] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %367 = llvm.extractvalue %5[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %368 = llvm.extractvalue %5[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %369 = llvm.getelementptr %367[%368] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%369, %366, %363) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %370 = llvm.mlir.constant(1 : index) : i64
    %371 = llvm.extractvalue %97[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %372 = llvm.mul %371, %370  : i64
    %373 = llvm.mlir.zero : !llvm.ptr
    %374 = llvm.getelementptr %373[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %375 = llvm.ptrtoint %374 : !llvm.ptr to i64
    %376 = llvm.mul %372, %375  : i64
    %377 = llvm.extractvalue %97[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %378 = llvm.extractvalue %97[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %379 = llvm.getelementptr %377[%378] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %380 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %381 = llvm.extractvalue %11[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %382 = llvm.getelementptr %380[%381] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%382, %379, %376) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.return
  }
}

