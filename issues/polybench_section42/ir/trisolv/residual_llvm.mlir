module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @kernel_trisolv(%arg0: i32, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: !llvm.ptr, %arg9: !llvm.ptr, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: !llvm.ptr, %arg14: !llvm.ptr, %arg15: i64, %arg16: i64, %arg17: i64) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg1, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.insertvalue %arg2, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.insertvalue %arg3, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.insertvalue %arg4, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.insertvalue %arg6, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.insertvalue %arg5, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.insertvalue %arg7, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.insertvalue %arg8, %8[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.insertvalue %arg9, %9[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.insertvalue %arg10, %10[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.insertvalue %arg11, %11[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = llvm.insertvalue %arg12, %12[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %15 = llvm.insertvalue %arg13, %14[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %16 = llvm.insertvalue %arg14, %15[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %17 = llvm.insertvalue %arg15, %16[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %18 = llvm.insertvalue %arg16, %17[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %19 = llvm.insertvalue %arg17, %18[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %20 = llvm.mlir.constant(0 : index) : i64
    %21 = llvm.mlir.constant(1 : index) : i64
    %22 = llvm.sext %arg0 : i32 to i64
    %23 = llvm.sub %22, %21  : i64
    %24 = llvm.mlir.constant(1 : index) : i64
    %25 = llvm.extractvalue %13[3] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %26 = llvm.alloca %24 x !llvm.array<1 x i64> : (i64) -> !llvm.ptr
    llvm.store %25, %26 : !llvm.array<1 x i64>, !llvm.ptr
    %27 = llvm.getelementptr %26[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x i64>
    %28 = llvm.load %27 : !llvm.ptr -> i64
    %29 = llvm.mlir.constant(1 : index) : i64
    %30 = llvm.mlir.zero : !llvm.ptr
    %31 = llvm.getelementptr %30[%28] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %32 = llvm.ptrtoint %31 : !llvm.ptr to i64
    %33 = llvm.mlir.constant(64 : index) : i64
    %34 = llvm.add %32, %33  : i64
    %35 = llvm.call @malloc(%34) : (i64) -> !llvm.ptr
    %36 = llvm.ptrtoint %35 : !llvm.ptr to i64
    %37 = llvm.mlir.constant(1 : index) : i64
    %38 = llvm.sub %33, %37  : i64
    %39 = llvm.add %36, %38  : i64
    %40 = llvm.urem %39, %33  : i64
    %41 = llvm.sub %39, %40  : i64
    %42 = llvm.inttoptr %41 : i64 to !llvm.ptr
    %43 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %44 = llvm.insertvalue %35, %43[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %45 = llvm.insertvalue %42, %44[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %46 = llvm.mlir.constant(0 : index) : i64
    %47 = llvm.insertvalue %46, %45[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %48 = llvm.insertvalue %28, %47[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %49 = llvm.insertvalue %29, %48[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %50 = llvm.mlir.constant(1 : index) : i64
    %51 = llvm.extractvalue %13[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %52 = llvm.mul %51, %50  : i64
    %53 = llvm.mlir.zero : !llvm.ptr
    %54 = llvm.getelementptr %53[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %55 = llvm.ptrtoint %54 : !llvm.ptr to i64
    %56 = llvm.mul %52, %55  : i64
    %57 = llvm.extractvalue %13[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %58 = llvm.extractvalue %13[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %59 = llvm.getelementptr %57[%58] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %60 = llvm.getelementptr %42[%46] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%60, %59, %56) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb1(%20, %49 : i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)
  ^bb1(%61: i64, %62: !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>):  // 2 preds: ^bb0, ^bb5
    %63 = llvm.icmp "slt" %61, %22 : i64
    llvm.cond_br %63, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %64 = llvm.extractvalue %19[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %65 = llvm.getelementptr %64[%61] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %66 = llvm.load %65 : !llvm.ptr -> f64
    %67 = llvm.extractvalue %62[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %68 = llvm.getelementptr %67[%61] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %66, %68 : f64, !llvm.ptr
    %69 = llvm.extractvalue %7[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %70 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %71 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %72 = llvm.insertvalue %69, %71[0] : !llvm.struct<(ptr, ptr, i64)> 
    %73 = llvm.insertvalue %70, %72[1] : !llvm.struct<(ptr, ptr, i64)> 
    %74 = llvm.mlir.constant(0 : index) : i64
    %75 = llvm.insertvalue %74, %73[2] : !llvm.struct<(ptr, ptr, i64)> 
    %76 = llvm.extractvalue %7[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %77 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %78 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %79 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %80 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %81 = llvm.mul %61, %79  : i64
    %82 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %83 = llvm.insertvalue %69, %82[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %84 = llvm.insertvalue %70, %83[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %85 = llvm.insertvalue %81, %84[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %86 = llvm.insertvalue %23, %85[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %87 = llvm.mlir.constant(1 : index) : i64
    %88 = llvm.insertvalue %87, %86[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %89 = llvm.extractvalue %62[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %90 = llvm.extractvalue %62[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %91 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %92 = llvm.insertvalue %89, %91[0] : !llvm.struct<(ptr, ptr, i64)> 
    %93 = llvm.insertvalue %90, %92[1] : !llvm.struct<(ptr, ptr, i64)> 
    %94 = llvm.mlir.constant(0 : index) : i64
    %95 = llvm.insertvalue %94, %93[2] : !llvm.struct<(ptr, ptr, i64)> 
    %96 = llvm.extractvalue %62[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %97 = llvm.extractvalue %62[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %98 = llvm.extractvalue %62[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %99 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %100 = llvm.insertvalue %89, %99[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %101 = llvm.insertvalue %90, %100[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %102 = llvm.mlir.constant(0 : index) : i64
    %103 = llvm.insertvalue %102, %101[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %104 = llvm.insertvalue %23, %103[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %105 = llvm.mlir.constant(1 : index) : i64
    %106 = llvm.insertvalue %105, %104[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %107 = llvm.mlir.constant(1 : index) : i64
    %108 = llvm.mlir.zero : !llvm.ptr
    %109 = llvm.getelementptr %108[%23] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %110 = llvm.ptrtoint %109 : !llvm.ptr to i64
    %111 = llvm.mlir.constant(64 : index) : i64
    %112 = llvm.add %110, %111  : i64
    %113 = llvm.call @malloc(%112) : (i64) -> !llvm.ptr
    %114 = llvm.ptrtoint %113 : !llvm.ptr to i64
    %115 = llvm.mlir.constant(1 : index) : i64
    %116 = llvm.sub %111, %115  : i64
    %117 = llvm.add %114, %116  : i64
    %118 = llvm.urem %117, %111  : i64
    %119 = llvm.sub %117, %118  : i64
    %120 = llvm.inttoptr %119 : i64 to !llvm.ptr
    %121 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %122 = llvm.insertvalue %113, %121[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %123 = llvm.insertvalue %120, %122[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %124 = llvm.mlir.constant(0 : index) : i64
    %125 = llvm.insertvalue %124, %123[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %126 = llvm.insertvalue %23, %125[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %127 = llvm.insertvalue %107, %126[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %128 = llvm.mlir.constant(1 : index) : i64
    %129 = llvm.mul %23, %128  : i64
    %130 = llvm.mlir.zero : !llvm.ptr
    %131 = llvm.getelementptr %130[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %132 = llvm.ptrtoint %131 : !llvm.ptr to i64
    %133 = llvm.mul %129, %132  : i64
    %134 = llvm.getelementptr %90[%102] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %135 = llvm.getelementptr %120[%124] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%135, %134, %133) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %136 = llvm.extractvalue %62[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %137 = llvm.extractvalue %62[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %138 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %139 = llvm.insertvalue %136, %138[0] : !llvm.struct<(ptr, ptr, i64)> 
    %140 = llvm.insertvalue %137, %139[1] : !llvm.struct<(ptr, ptr, i64)> 
    %141 = llvm.mlir.constant(0 : index) : i64
    %142 = llvm.insertvalue %141, %140[2] : !llvm.struct<(ptr, ptr, i64)> 
    %143 = llvm.extractvalue %62[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %144 = llvm.extractvalue %62[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %145 = llvm.extractvalue %62[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %146 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %147 = llvm.insertvalue %136, %146[0] : !llvm.struct<(ptr, ptr, i64)> 
    %148 = llvm.insertvalue %137, %147[1] : !llvm.struct<(ptr, ptr, i64)> 
    %149 = llvm.insertvalue %61, %148[2] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.br ^bb3(%20 : i64)
  ^bb3(%150: i64):  // 2 preds: ^bb2, ^bb4
    %151 = llvm.icmp "slt" %150, %23 : i64
    llvm.cond_br %151, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %152 = llvm.getelementptr %70[%81] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %153 = llvm.getelementptr %152[%150] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %154 = llvm.load %153 : !llvm.ptr -> f64
    %155 = llvm.getelementptr %120[%150] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %156 = llvm.load %155 : !llvm.ptr -> f64
    %157 = llvm.getelementptr %137[%61] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %158 = llvm.load %157 : !llvm.ptr -> f64
    %159 = llvm.fmul %154, %156  : f64
    %160 = llvm.fsub %158, %159  : f64
    %161 = llvm.icmp "slt" %150, %61 : i64
    %162 = llvm.select %161, %160, %158 : i1, f64
    %163 = llvm.getelementptr %137[%61] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %162, %163 : f64, !llvm.ptr
    %164 = llvm.add %150, %21  : i64
    llvm.br ^bb3(%164 : i64)
  ^bb5:  // pred: ^bb3
    %165 = llvm.extractvalue %62[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %166 = llvm.extractvalue %62[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %167 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %168 = llvm.insertvalue %165, %167[0] : !llvm.struct<(ptr, ptr, i64)> 
    %169 = llvm.insertvalue %166, %168[1] : !llvm.struct<(ptr, ptr, i64)> 
    %170 = llvm.mlir.constant(0 : index) : i64
    %171 = llvm.insertvalue %170, %169[2] : !llvm.struct<(ptr, ptr, i64)> 
    %172 = llvm.extractvalue %62[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %173 = llvm.extractvalue %62[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %174 = llvm.extractvalue %62[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %175 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %176 = llvm.insertvalue %165, %175[0] : !llvm.struct<(ptr, ptr, i64)> 
    %177 = llvm.insertvalue %166, %176[1] : !llvm.struct<(ptr, ptr, i64)> 
    %178 = llvm.insertvalue %61, %177[2] : !llvm.struct<(ptr, ptr, i64)> 
    %179 = llvm.mlir.constant(1 : index) : i64
    %180 = llvm.mlir.zero : !llvm.ptr
    %181 = llvm.getelementptr %180[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %182 = llvm.ptrtoint %181 : !llvm.ptr to i64
    %183 = llvm.mul %182, %179  : i64
    %184 = llvm.getelementptr %137[%61] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %185 = llvm.getelementptr %166[%61] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%185, %184, %183) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %186 = llvm.extractvalue %62[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %187 = llvm.getelementptr %186[%61] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %188 = llvm.load %187 : !llvm.ptr -> f64
    %189 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %190 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %191 = llvm.mul %61, %190  : i64
    %192 = llvm.add %191, %61  : i64
    %193 = llvm.getelementptr %189[%192] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %194 = llvm.load %193 : !llvm.ptr -> f64
    %195 = llvm.fdiv %188, %194  : f64
    %196 = llvm.extractvalue %62[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %197 = llvm.getelementptr %196[%61] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %195, %197 : f64, !llvm.ptr
    %198 = llvm.add %61, %21  : i64
    llvm.br ^bb1(%198, %62 : i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)
  ^bb6:  // pred: ^bb1
    %199 = llvm.mlir.constant(1 : index) : i64
    %200 = llvm.extractvalue %62[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %201 = llvm.mul %200, %199  : i64
    %202 = llvm.mlir.zero : !llvm.ptr
    %203 = llvm.getelementptr %202[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %204 = llvm.ptrtoint %203 : !llvm.ptr to i64
    %205 = llvm.mul %201, %204  : i64
    %206 = llvm.extractvalue %62[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %207 = llvm.extractvalue %62[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %208 = llvm.getelementptr %206[%207] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %209 = llvm.extractvalue %13[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %210 = llvm.extractvalue %13[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %211 = llvm.getelementptr %209[%210] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%211, %208, %205) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.return
  }
}

