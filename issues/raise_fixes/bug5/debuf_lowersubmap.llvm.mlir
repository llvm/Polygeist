module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.func @memrefCopy(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @kernel_jacobi_2d(%arg0: i32, %arg1: i32, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: !llvm.ptr, %arg10: !llvm.ptr, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: i64) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg2, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.insertvalue %arg3, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.insertvalue %arg4, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.insertvalue %arg5, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.insertvalue %arg7, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.insertvalue %arg6, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.insertvalue %arg8, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.insertvalue %arg9, %8[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %10 = llvm.insertvalue %arg10, %9[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.insertvalue %arg11, %10[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %12 = llvm.insertvalue %arg12, %11[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.insertvalue %arg14, %12[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.insertvalue %arg13, %13[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.insertvalue %arg15, %14[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.mlir.constant(0 : index) : i64
    %17 = llvm.mlir.constant(-1 : index) : i64
    %18 = llvm.mlir.constant(2.000000e-01 : f64) : f64
    %19 = llvm.mlir.constant(1 : index) : i64
    %20 = llvm.sext %arg1 : i32 to i64
    %21 = llvm.sext %arg0 : i32 to i64
    %22 = llvm.add %20, %17  : i64
    %23 = llvm.add %20, %17  : i64
    %24 = llvm.sub %23, %19  : i64
    %25 = llvm.add %20, %17  : i64
    %26 = llvm.sub %25, %19  : i64
    %27 = llvm.sub %22, %19  : i64
    %28 = llvm.add %20, %17  : i64
    %29 = llvm.sub %28, %19  : i64
    %30 = llvm.mlir.constant(1 : index) : i64
    %31 = llvm.extractvalue %15[3] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %32 = llvm.alloca %30 x !llvm.array<2 x i64> : (i64) -> !llvm.ptr
    llvm.store %31, %32 : !llvm.array<2 x i64>, !llvm.ptr
    %33 = llvm.getelementptr %32[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
    %34 = llvm.load %33 : !llvm.ptr -> i64
    %35 = llvm.mlir.constant(1 : index) : i64
    %36 = llvm.extractvalue %15[3] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %37 = llvm.alloca %35 x !llvm.array<2 x i64> : (i64) -> !llvm.ptr
    llvm.store %36, %37 : !llvm.array<2 x i64>, !llvm.ptr
    %38 = llvm.getelementptr %37[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
    %39 = llvm.load %38 : !llvm.ptr -> i64
    %40 = llvm.mlir.constant(1 : index) : i64
    %41 = llvm.mul %39, %34  : i64
    %42 = llvm.mlir.zero : !llvm.ptr
    %43 = llvm.getelementptr %42[%41] : (!llvm.ptr, i64) -> !llvm.ptr, f64
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
    %55 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %56 = llvm.insertvalue %47, %55[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %57 = llvm.insertvalue %54, %56[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %58 = llvm.mlir.constant(0 : index) : i64
    %59 = llvm.insertvalue %58, %57[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %60 = llvm.insertvalue %34, %59[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %61 = llvm.insertvalue %39, %60[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %62 = llvm.insertvalue %39, %61[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %63 = llvm.insertvalue %40, %62[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %64 = llvm.mlir.constant(1 : index) : i64
    %65 = llvm.extractvalue %15[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %66 = llvm.mul %65, %64  : i64
    %67 = llvm.extractvalue %15[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %68 = llvm.mul %66, %67  : i64
    %69 = llvm.mlir.zero : !llvm.ptr
    %70 = llvm.getelementptr %69[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %71 = llvm.ptrtoint %70 : !llvm.ptr to i64
    %72 = llvm.mul %68, %71  : i64
    %73 = llvm.extractvalue %15[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %74 = llvm.extractvalue %15[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %75 = llvm.getelementptr %73[%74] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %76 = llvm.getelementptr %54[%58] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%76, %75, %72) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %77 = llvm.mlir.constant(1 : index) : i64
    %78 = llvm.extractvalue %7[3] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %79 = llvm.alloca %77 x !llvm.array<2 x i64> : (i64) -> !llvm.ptr
    llvm.store %78, %79 : !llvm.array<2 x i64>, !llvm.ptr
    %80 = llvm.getelementptr %79[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
    %81 = llvm.load %80 : !llvm.ptr -> i64
    %82 = llvm.mlir.constant(1 : index) : i64
    %83 = llvm.extractvalue %7[3] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %84 = llvm.alloca %82 x !llvm.array<2 x i64> : (i64) -> !llvm.ptr
    llvm.store %83, %84 : !llvm.array<2 x i64>, !llvm.ptr
    %85 = llvm.getelementptr %84[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
    %86 = llvm.load %85 : !llvm.ptr -> i64
    %87 = llvm.mlir.constant(1 : index) : i64
    %88 = llvm.mul %86, %81  : i64
    %89 = llvm.mlir.zero : !llvm.ptr
    %90 = llvm.getelementptr %89[%88] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %91 = llvm.ptrtoint %90 : !llvm.ptr to i64
    %92 = llvm.mlir.constant(64 : index) : i64
    %93 = llvm.add %91, %92  : i64
    %94 = llvm.call @malloc(%93) : (i64) -> !llvm.ptr
    %95 = llvm.ptrtoint %94 : !llvm.ptr to i64
    %96 = llvm.mlir.constant(1 : index) : i64
    %97 = llvm.sub %92, %96  : i64
    %98 = llvm.add %95, %97  : i64
    %99 = llvm.urem %98, %92  : i64
    %100 = llvm.sub %98, %99  : i64
    %101 = llvm.inttoptr %100 : i64 to !llvm.ptr
    %102 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %103 = llvm.insertvalue %94, %102[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %104 = llvm.insertvalue %101, %103[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %105 = llvm.mlir.constant(0 : index) : i64
    %106 = llvm.insertvalue %105, %104[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %107 = llvm.insertvalue %81, %106[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %108 = llvm.insertvalue %86, %107[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %109 = llvm.insertvalue %86, %108[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %110 = llvm.insertvalue %87, %109[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %111 = llvm.mlir.constant(1 : index) : i64
    %112 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %113 = llvm.mul %112, %111  : i64
    %114 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %115 = llvm.mul %113, %114  : i64
    %116 = llvm.mlir.zero : !llvm.ptr
    %117 = llvm.getelementptr %116[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %118 = llvm.ptrtoint %117 : !llvm.ptr to i64
    %119 = llvm.mul %115, %118  : i64
    %120 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %121 = llvm.extractvalue %7[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %122 = llvm.getelementptr %120[%121] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %123 = llvm.getelementptr %101[%105] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%123, %122, %119) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb1(%16, %63, %110 : i64, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb1(%124: i64, %125: !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, %126: !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>):  // 2 preds: ^bb0, ^bb26
    %127 = llvm.icmp "slt" %124, %21 : i64
    llvm.cond_br %127, ^bb2, ^bb27
  ^bb2:  // pred: ^bb1
    %128 = llvm.mlir.constant(1 : index) : i64
    %129 = llvm.mul %24, %26  : i64
    %130 = llvm.mlir.zero : !llvm.ptr
    %131 = llvm.getelementptr %130[%129] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %132 = llvm.ptrtoint %131 : !llvm.ptr to i64
    %133 = llvm.mlir.constant(64 : index) : i64
    %134 = llvm.add %132, %133  : i64
    %135 = llvm.call @malloc(%134) : (i64) -> !llvm.ptr
    %136 = llvm.ptrtoint %135 : !llvm.ptr to i64
    %137 = llvm.mlir.constant(1 : index) : i64
    %138 = llvm.sub %133, %137  : i64
    %139 = llvm.add %136, %138  : i64
    %140 = llvm.urem %139, %133  : i64
    %141 = llvm.sub %139, %140  : i64
    %142 = llvm.inttoptr %141 : i64 to !llvm.ptr
    %143 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %144 = llvm.insertvalue %135, %143[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %145 = llvm.insertvalue %142, %144[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %146 = llvm.mlir.constant(0 : index) : i64
    %147 = llvm.insertvalue %146, %145[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %148 = llvm.insertvalue %26, %147[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %149 = llvm.insertvalue %24, %148[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %150 = llvm.insertvalue %24, %149[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %151 = llvm.insertvalue %128, %150[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb3(%16 : i64)
  ^bb3(%152: i64):  // 2 preds: ^bb2, ^bb7
    %153 = llvm.icmp "slt" %152, %26 : i64
    llvm.cond_br %153, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%16 : i64)
  ^bb5(%154: i64):  // 2 preds: ^bb4, ^bb6
    %155 = llvm.icmp "slt" %154, %24 : i64
    llvm.cond_br %155, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %156 = llvm.mlir.constant(1 : index) : i64
    %157 = llvm.add %152, %156  : i64
    %158 = llvm.mlir.constant(1 : index) : i64
    %159 = llvm.add %154, %158  : i64
    %160 = llvm.extractvalue %125[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %161 = llvm.extractvalue %125[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %162 = llvm.mul %157, %161  : i64
    %163 = llvm.add %162, %159  : i64
    %164 = llvm.getelementptr %160[%163] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %165 = llvm.load %164 : !llvm.ptr -> f64
    %166 = llvm.mul %152, %24  : i64
    %167 = llvm.add %166, %154  : i64
    %168 = llvm.getelementptr %142[%167] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %165, %168 : f64, !llvm.ptr
    %169 = llvm.add %154, %19  : i64
    llvm.br ^bb5(%169 : i64)
  ^bb7:  // pred: ^bb5
    %170 = llvm.add %152, %19  : i64
    llvm.br ^bb3(%170 : i64)
  ^bb8:  // pred: ^bb3
    %171 = llvm.mlir.constant(1 : index) : i64
    %172 = llvm.extractvalue %126[3] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %173 = llvm.alloca %171 x !llvm.array<2 x i64> : (i64) -> !llvm.ptr
    llvm.store %172, %173 : !llvm.array<2 x i64>, !llvm.ptr
    %174 = llvm.getelementptr %173[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
    %175 = llvm.load %174 : !llvm.ptr -> i64
    %176 = llvm.mlir.constant(1 : index) : i64
    %177 = llvm.extractvalue %126[3] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %178 = llvm.alloca %176 x !llvm.array<2 x i64> : (i64) -> !llvm.ptr
    llvm.store %177, %178 : !llvm.array<2 x i64>, !llvm.ptr
    %179 = llvm.getelementptr %178[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
    %180 = llvm.load %179 : !llvm.ptr -> i64
    llvm.br ^bb9(%16 : i64)
  ^bb9(%181: i64):  // 2 preds: ^bb8, ^bb13
    %182 = llvm.icmp "slt" %181, %180 : i64
    llvm.cond_br %182, ^bb10, ^bb14
  ^bb10:  // pred: ^bb9
    llvm.br ^bb11(%16 : i64)
  ^bb11(%183: i64):  // 2 preds: ^bb10, ^bb12
    %184 = llvm.icmp "slt" %183, %175 : i64
    llvm.cond_br %184, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    %185 = llvm.mlir.constant(1 : index) : i64
    %186 = llvm.add %181, %185  : i64
    %187 = llvm.mlir.constant(1 : index) : i64
    %188 = llvm.add %183, %187  : i64
    %189 = llvm.extractvalue %126[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %190 = llvm.extractvalue %126[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %191 = llvm.mul %186, %190  : i64
    %192 = llvm.add %191, %188  : i64
    %193 = llvm.getelementptr %189[%192] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %194 = llvm.load %193 : !llvm.ptr -> f64
    %195 = llvm.mlir.constant(1 : index) : i64
    %196 = llvm.add %181, %195  : i64
    %197 = llvm.extractvalue %126[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %198 = llvm.extractvalue %126[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %199 = llvm.mul %196, %198  : i64
    %200 = llvm.add %199, %183  : i64
    %201 = llvm.getelementptr %197[%200] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %202 = llvm.load %201 : !llvm.ptr -> f64
    %203 = llvm.mlir.constant(1 : index) : i64
    %204 = llvm.add %181, %203  : i64
    %205 = llvm.mlir.constant(2 : index) : i64
    %206 = llvm.add %183, %205  : i64
    %207 = llvm.extractvalue %126[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %208 = llvm.extractvalue %126[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %209 = llvm.mul %204, %208  : i64
    %210 = llvm.add %209, %206  : i64
    %211 = llvm.getelementptr %207[%210] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %212 = llvm.load %211 : !llvm.ptr -> f64
    %213 = llvm.mlir.constant(2 : index) : i64
    %214 = llvm.add %181, %213  : i64
    %215 = llvm.mlir.constant(1 : index) : i64
    %216 = llvm.add %183, %215  : i64
    %217 = llvm.extractvalue %126[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %218 = llvm.extractvalue %126[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %219 = llvm.mul %214, %218  : i64
    %220 = llvm.add %219, %216  : i64
    %221 = llvm.getelementptr %217[%220] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %222 = llvm.load %221 : !llvm.ptr -> f64
    %223 = llvm.mlir.constant(1 : index) : i64
    %224 = llvm.add %183, %223  : i64
    %225 = llvm.extractvalue %126[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %226 = llvm.extractvalue %126[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %227 = llvm.mul %181, %226  : i64
    %228 = llvm.add %227, %224  : i64
    %229 = llvm.getelementptr %225[%228] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %230 = llvm.load %229 : !llvm.ptr -> f64
    %231 = llvm.fadd %194, %202  : f64
    %232 = llvm.fadd %231, %212  : f64
    %233 = llvm.fadd %232, %222  : f64
    %234 = llvm.fadd %233, %230  : f64
    %235 = llvm.fmul %234, %18  : f64
    %236 = llvm.mul %181, %24  : i64
    %237 = llvm.add %236, %183  : i64
    %238 = llvm.getelementptr %142[%237] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %235, %238 : f64, !llvm.ptr
    %239 = llvm.add %183, %19  : i64
    llvm.br ^bb11(%239 : i64)
  ^bb13:  // pred: ^bb11
    %240 = llvm.add %181, %19  : i64
    llvm.br ^bb9(%240 : i64)
  ^bb14:  // pred: ^bb9
    %241 = llvm.extractvalue %125[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %242 = llvm.extractvalue %125[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %243 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %244 = llvm.insertvalue %241, %243[0] : !llvm.struct<(ptr, ptr, i64)> 
    %245 = llvm.insertvalue %242, %244[1] : !llvm.struct<(ptr, ptr, i64)> 
    %246 = llvm.mlir.constant(0 : index) : i64
    %247 = llvm.insertvalue %246, %245[2] : !llvm.struct<(ptr, ptr, i64)> 
    %248 = llvm.extractvalue %125[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %249 = llvm.extractvalue %125[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %250 = llvm.extractvalue %125[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %251 = llvm.extractvalue %125[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %252 = llvm.extractvalue %125[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %253 = llvm.mlir.constant(1 : index) : i64
    %254 = llvm.add %251, %253  : i64
    %255 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %256 = llvm.insertvalue %241, %255[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %257 = llvm.insertvalue %242, %256[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %258 = llvm.insertvalue %254, %257[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %259 = llvm.insertvalue %26, %258[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %260 = llvm.insertvalue %251, %259[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %261 = llvm.insertvalue %24, %260[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %262 = llvm.mlir.constant(1 : index) : i64
    %263 = llvm.insertvalue %262, %261[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %264 = llvm.intr.stacksave : !llvm.ptr
    %265 = llvm.mlir.constant(2 : i64) : i64
    %266 = llvm.mlir.constant(1 : index) : i64
    %267 = llvm.alloca %266 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %151, %267 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %268 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %269 = llvm.insertvalue %265, %268[0] : !llvm.struct<(i64, ptr)> 
    %270 = llvm.insertvalue %267, %269[1] : !llvm.struct<(i64, ptr)> 
    %271 = llvm.mlir.constant(2 : i64) : i64
    %272 = llvm.mlir.constant(1 : index) : i64
    %273 = llvm.alloca %272 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %263, %273 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %274 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %275 = llvm.insertvalue %271, %274[0] : !llvm.struct<(i64, ptr)> 
    %276 = llvm.insertvalue %273, %275[1] : !llvm.struct<(i64, ptr)> 
    %277 = llvm.mlir.constant(1 : index) : i64
    %278 = llvm.alloca %277 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %270, %278 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %279 = llvm.alloca %277 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %276, %279 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %280 = llvm.mlir.zero : !llvm.ptr
    %281 = llvm.getelementptr %280[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %282 = llvm.ptrtoint %281 : !llvm.ptr to i64
    llvm.call @memrefCopy(%282, %278, %279) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %264 : !llvm.ptr
    %283 = llvm.mlir.constant(1 : index) : i64
    %284 = llvm.mul %27, %29  : i64
    %285 = llvm.mlir.zero : !llvm.ptr
    %286 = llvm.getelementptr %285[%284] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %287 = llvm.ptrtoint %286 : !llvm.ptr to i64
    %288 = llvm.mlir.constant(64 : index) : i64
    %289 = llvm.add %287, %288  : i64
    %290 = llvm.call @malloc(%289) : (i64) -> !llvm.ptr
    %291 = llvm.ptrtoint %290 : !llvm.ptr to i64
    %292 = llvm.mlir.constant(1 : index) : i64
    %293 = llvm.sub %288, %292  : i64
    %294 = llvm.add %291, %293  : i64
    %295 = llvm.urem %294, %288  : i64
    %296 = llvm.sub %294, %295  : i64
    %297 = llvm.inttoptr %296 : i64 to !llvm.ptr
    %298 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %299 = llvm.insertvalue %290, %298[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %300 = llvm.insertvalue %297, %299[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %301 = llvm.mlir.constant(0 : index) : i64
    %302 = llvm.insertvalue %301, %300[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %303 = llvm.insertvalue %29, %302[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %304 = llvm.insertvalue %27, %303[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %305 = llvm.insertvalue %27, %304[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %306 = llvm.insertvalue %283, %305[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb15(%16 : i64)
  ^bb15(%307: i64):  // 2 preds: ^bb14, ^bb19
    %308 = llvm.icmp "slt" %307, %29 : i64
    llvm.cond_br %308, ^bb16, ^bb20
  ^bb16:  // pred: ^bb15
    llvm.br ^bb17(%16 : i64)
  ^bb17(%309: i64):  // 2 preds: ^bb16, ^bb18
    %310 = llvm.icmp "slt" %309, %27 : i64
    llvm.cond_br %310, ^bb18, ^bb19
  ^bb18:  // pred: ^bb17
    %311 = llvm.mlir.constant(1 : index) : i64
    %312 = llvm.add %307, %311  : i64
    %313 = llvm.mlir.constant(1 : index) : i64
    %314 = llvm.add %309, %313  : i64
    %315 = llvm.extractvalue %126[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %316 = llvm.extractvalue %126[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %317 = llvm.mul %312, %316  : i64
    %318 = llvm.add %317, %314  : i64
    %319 = llvm.getelementptr %315[%318] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %320 = llvm.load %319 : !llvm.ptr -> f64
    %321 = llvm.mul %307, %27  : i64
    %322 = llvm.add %321, %309  : i64
    %323 = llvm.getelementptr %297[%322] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %320, %323 : f64, !llvm.ptr
    %324 = llvm.add %309, %19  : i64
    llvm.br ^bb17(%324 : i64)
  ^bb19:  // pred: ^bb17
    %325 = llvm.add %307, %19  : i64
    llvm.br ^bb15(%325 : i64)
  ^bb20:  // pred: ^bb15
    %326 = llvm.mlir.constant(1 : index) : i64
    %327 = llvm.extractvalue %125[3] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %328 = llvm.alloca %326 x !llvm.array<2 x i64> : (i64) -> !llvm.ptr
    llvm.store %327, %328 : !llvm.array<2 x i64>, !llvm.ptr
    %329 = llvm.getelementptr %328[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
    %330 = llvm.load %329 : !llvm.ptr -> i64
    %331 = llvm.mlir.constant(1 : index) : i64
    %332 = llvm.extractvalue %125[3] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %333 = llvm.alloca %331 x !llvm.array<2 x i64> : (i64) -> !llvm.ptr
    llvm.store %332, %333 : !llvm.array<2 x i64>, !llvm.ptr
    %334 = llvm.getelementptr %333[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
    %335 = llvm.load %334 : !llvm.ptr -> i64
    llvm.br ^bb21(%16 : i64)
  ^bb21(%336: i64):  // 2 preds: ^bb20, ^bb25
    %337 = llvm.icmp "slt" %336, %335 : i64
    llvm.cond_br %337, ^bb22, ^bb26
  ^bb22:  // pred: ^bb21
    llvm.br ^bb23(%16 : i64)
  ^bb23(%338: i64):  // 2 preds: ^bb22, ^bb24
    %339 = llvm.icmp "slt" %338, %330 : i64
    llvm.cond_br %339, ^bb24, ^bb25
  ^bb24:  // pred: ^bb23
    %340 = llvm.mlir.constant(1 : index) : i64
    %341 = llvm.add %336, %340  : i64
    %342 = llvm.mlir.constant(1 : index) : i64
    %343 = llvm.add %338, %342  : i64
    %344 = llvm.extractvalue %125[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %345 = llvm.extractvalue %125[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %346 = llvm.mul %341, %345  : i64
    %347 = llvm.add %346, %343  : i64
    %348 = llvm.getelementptr %344[%347] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %349 = llvm.load %348 : !llvm.ptr -> f64
    %350 = llvm.mlir.constant(1 : index) : i64
    %351 = llvm.add %336, %350  : i64
    %352 = llvm.extractvalue %125[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %353 = llvm.extractvalue %125[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %354 = llvm.mul %351, %353  : i64
    %355 = llvm.add %354, %338  : i64
    %356 = llvm.getelementptr %352[%355] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %357 = llvm.load %356 : !llvm.ptr -> f64
    %358 = llvm.mlir.constant(1 : index) : i64
    %359 = llvm.add %336, %358  : i64
    %360 = llvm.mlir.constant(2 : index) : i64
    %361 = llvm.add %338, %360  : i64
    %362 = llvm.extractvalue %125[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %363 = llvm.extractvalue %125[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %364 = llvm.mul %359, %363  : i64
    %365 = llvm.add %364, %361  : i64
    %366 = llvm.getelementptr %362[%365] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %367 = llvm.load %366 : !llvm.ptr -> f64
    %368 = llvm.mlir.constant(2 : index) : i64
    %369 = llvm.add %336, %368  : i64
    %370 = llvm.mlir.constant(1 : index) : i64
    %371 = llvm.add %338, %370  : i64
    %372 = llvm.extractvalue %125[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %373 = llvm.extractvalue %125[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %374 = llvm.mul %369, %373  : i64
    %375 = llvm.add %374, %371  : i64
    %376 = llvm.getelementptr %372[%375] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %377 = llvm.load %376 : !llvm.ptr -> f64
    %378 = llvm.mlir.constant(1 : index) : i64
    %379 = llvm.add %338, %378  : i64
    %380 = llvm.extractvalue %125[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %381 = llvm.extractvalue %125[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %382 = llvm.mul %336, %381  : i64
    %383 = llvm.add %382, %379  : i64
    %384 = llvm.getelementptr %380[%383] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %385 = llvm.load %384 : !llvm.ptr -> f64
    %386 = llvm.fadd %349, %357  : f64
    %387 = llvm.fadd %386, %367  : f64
    %388 = llvm.fadd %387, %377  : f64
    %389 = llvm.fadd %388, %385  : f64
    %390 = llvm.fmul %389, %18  : f64
    %391 = llvm.mul %336, %27  : i64
    %392 = llvm.add %391, %338  : i64
    %393 = llvm.getelementptr %297[%392] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %390, %393 : f64, !llvm.ptr
    %394 = llvm.add %338, %19  : i64
    llvm.br ^bb23(%394 : i64)
  ^bb25:  // pred: ^bb23
    %395 = llvm.add %336, %19  : i64
    llvm.br ^bb21(%395 : i64)
  ^bb26:  // pred: ^bb21
    %396 = llvm.extractvalue %126[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %397 = llvm.extractvalue %126[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %398 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %399 = llvm.insertvalue %396, %398[0] : !llvm.struct<(ptr, ptr, i64)> 
    %400 = llvm.insertvalue %397, %399[1] : !llvm.struct<(ptr, ptr, i64)> 
    %401 = llvm.mlir.constant(0 : index) : i64
    %402 = llvm.insertvalue %401, %400[2] : !llvm.struct<(ptr, ptr, i64)> 
    %403 = llvm.extractvalue %126[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %404 = llvm.extractvalue %126[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %405 = llvm.extractvalue %126[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %406 = llvm.extractvalue %126[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %407 = llvm.extractvalue %126[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %408 = llvm.mlir.constant(1 : index) : i64
    %409 = llvm.add %406, %408  : i64
    %410 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %411 = llvm.insertvalue %396, %410[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %412 = llvm.insertvalue %397, %411[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %413 = llvm.insertvalue %409, %412[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %414 = llvm.insertvalue %29, %413[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %415 = llvm.insertvalue %406, %414[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %416 = llvm.insertvalue %27, %415[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %417 = llvm.mlir.constant(1 : index) : i64
    %418 = llvm.insertvalue %417, %416[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %419 = llvm.intr.stacksave : !llvm.ptr
    %420 = llvm.mlir.constant(2 : i64) : i64
    %421 = llvm.mlir.constant(1 : index) : i64
    %422 = llvm.alloca %421 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %306, %422 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %423 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %424 = llvm.insertvalue %420, %423[0] : !llvm.struct<(i64, ptr)> 
    %425 = llvm.insertvalue %422, %424[1] : !llvm.struct<(i64, ptr)> 
    %426 = llvm.mlir.constant(2 : i64) : i64
    %427 = llvm.mlir.constant(1 : index) : i64
    %428 = llvm.alloca %427 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %418, %428 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %429 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %430 = llvm.insertvalue %426, %429[0] : !llvm.struct<(i64, ptr)> 
    %431 = llvm.insertvalue %428, %430[1] : !llvm.struct<(i64, ptr)> 
    %432 = llvm.mlir.constant(1 : index) : i64
    %433 = llvm.alloca %432 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %425, %433 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %434 = llvm.alloca %432 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %431, %434 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %435 = llvm.mlir.zero : !llvm.ptr
    %436 = llvm.getelementptr %435[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %437 = llvm.ptrtoint %436 : !llvm.ptr to i64
    llvm.call @memrefCopy(%437, %433, %434) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %419 : !llvm.ptr
    %438 = llvm.add %124, %19  : i64
    llvm.br ^bb1(%438, %125, %126 : i64, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb27:  // pred: ^bb1
    %439 = llvm.mlir.constant(1 : index) : i64
    %440 = llvm.extractvalue %125[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %441 = llvm.mul %440, %439  : i64
    %442 = llvm.extractvalue %125[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %443 = llvm.mul %441, %442  : i64
    %444 = llvm.mlir.zero : !llvm.ptr
    %445 = llvm.getelementptr %444[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %446 = llvm.ptrtoint %445 : !llvm.ptr to i64
    %447 = llvm.mul %443, %446  : i64
    %448 = llvm.extractvalue %125[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %449 = llvm.extractvalue %125[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %450 = llvm.getelementptr %448[%449] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %451 = llvm.extractvalue %15[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %452 = llvm.extractvalue %15[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %453 = llvm.getelementptr %451[%452] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%453, %450, %447) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %454 = llvm.mlir.constant(1 : index) : i64
    %455 = llvm.extractvalue %126[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %456 = llvm.mul %455, %454  : i64
    %457 = llvm.extractvalue %126[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %458 = llvm.mul %456, %457  : i64
    %459 = llvm.mlir.zero : !llvm.ptr
    %460 = llvm.getelementptr %459[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %461 = llvm.ptrtoint %460 : !llvm.ptr to i64
    %462 = llvm.mul %458, %461  : i64
    %463 = llvm.extractvalue %126[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %464 = llvm.extractvalue %126[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %465 = llvm.getelementptr %463[%464] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %466 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %467 = llvm.extractvalue %7[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %468 = llvm.getelementptr %466[%467] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%468, %465, %462) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.return
  }
}

