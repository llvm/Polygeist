module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.func @memrefCopy(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @kernel_floyd_warshall(%arg0: i32, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg1, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.insertvalue %arg2, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.insertvalue %arg3, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.insertvalue %arg4, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.insertvalue %arg6, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.insertvalue %arg5, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.insertvalue %arg7, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.mlir.constant(0 : index) : i64
    %9 = llvm.mlir.constant(1 : index) : i64
    %10 = llvm.sext %arg0 : i32 to i64
    %11 = llvm.extractvalue %7[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %12 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %14 = llvm.insertvalue %11, %13[0] : !llvm.struct<(ptr, ptr, i64)> 
    %15 = llvm.insertvalue %12, %14[1] : !llvm.struct<(ptr, ptr, i64)> 
    %16 = llvm.mlir.constant(0 : index) : i64
    %17 = llvm.insertvalue %16, %15[2] : !llvm.struct<(ptr, ptr, i64)> 
    %18 = llvm.extractvalue %7[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %19 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %24 = llvm.insertvalue %11, %23[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %25 = llvm.insertvalue %12, %24[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %26 = llvm.mlir.constant(0 : index) : i64
    %27 = llvm.insertvalue %26, %25[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %28 = llvm.insertvalue %10, %27[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %29 = llvm.insertvalue %21, %28[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %30 = llvm.insertvalue %10, %29[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %31 = llvm.mlir.constant(1 : index) : i64
    %32 = llvm.insertvalue %31, %30[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %33 = llvm.extractvalue %7[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %34 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %35 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %36 = llvm.insertvalue %33, %35[0] : !llvm.struct<(ptr, ptr, i64)> 
    %37 = llvm.insertvalue %34, %36[1] : !llvm.struct<(ptr, ptr, i64)> 
    %38 = llvm.mlir.constant(0 : index) : i64
    %39 = llvm.insertvalue %38, %37[2] : !llvm.struct<(ptr, ptr, i64)> 
    %40 = llvm.extractvalue %7[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %41 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %42 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %43 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %44 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %45 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %46 = llvm.insertvalue %33, %45[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %47 = llvm.insertvalue %34, %46[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %48 = llvm.mlir.constant(0 : index) : i64
    %49 = llvm.insertvalue %48, %47[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %50 = llvm.insertvalue %10, %49[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %51 = llvm.insertvalue %43, %50[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %52 = llvm.insertvalue %10, %51[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %53 = llvm.mlir.constant(1 : index) : i64
    %54 = llvm.insertvalue %53, %52[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %55 = llvm.extractvalue %7[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %56 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %57 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %58 = llvm.insertvalue %55, %57[0] : !llvm.struct<(ptr, ptr, i64)> 
    %59 = llvm.insertvalue %56, %58[1] : !llvm.struct<(ptr, ptr, i64)> 
    %60 = llvm.mlir.constant(0 : index) : i64
    %61 = llvm.insertvalue %60, %59[2] : !llvm.struct<(ptr, ptr, i64)> 
    %62 = llvm.extractvalue %7[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %63 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %64 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %65 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %66 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %67 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %68 = llvm.insertvalue %55, %67[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %69 = llvm.insertvalue %56, %68[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %70 = llvm.mlir.constant(0 : index) : i64
    %71 = llvm.insertvalue %70, %69[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %72 = llvm.insertvalue %10, %71[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %73 = llvm.insertvalue %65, %72[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %74 = llvm.insertvalue %10, %73[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %75 = llvm.mlir.constant(1 : index) : i64
    %76 = llvm.insertvalue %75, %74[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %77 = llvm.mlir.constant(1 : index) : i64
    %78 = llvm.mul %10, %10  : i64
    %79 = llvm.mlir.zero : !llvm.ptr
    %80 = llvm.getelementptr %79[%78] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %81 = llvm.ptrtoint %80 : !llvm.ptr to i64
    %82 = llvm.mlir.constant(64 : index) : i64
    %83 = llvm.add %81, %82  : i64
    %84 = llvm.call @malloc(%83) : (i64) -> !llvm.ptr
    %85 = llvm.ptrtoint %84 : !llvm.ptr to i64
    %86 = llvm.mlir.constant(1 : index) : i64
    %87 = llvm.sub %82, %86  : i64
    %88 = llvm.add %85, %87  : i64
    %89 = llvm.urem %88, %82  : i64
    %90 = llvm.sub %88, %89  : i64
    %91 = llvm.inttoptr %90 : i64 to !llvm.ptr
    %92 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %93 = llvm.insertvalue %84, %92[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %94 = llvm.insertvalue %91, %93[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %95 = llvm.mlir.constant(0 : index) : i64
    %96 = llvm.insertvalue %95, %94[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %97 = llvm.insertvalue %10, %96[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %98 = llvm.insertvalue %10, %97[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %99 = llvm.insertvalue %10, %98[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %100 = llvm.insertvalue %77, %99[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %101 = llvm.intr.stacksave : !llvm.ptr
    %102 = llvm.mlir.constant(2 : i64) : i64
    %103 = llvm.mlir.constant(1 : index) : i64
    %104 = llvm.alloca %103 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %76, %104 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %105 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %106 = llvm.insertvalue %102, %105[0] : !llvm.struct<(i64, ptr)> 
    %107 = llvm.insertvalue %104, %106[1] : !llvm.struct<(i64, ptr)> 
    %108 = llvm.mlir.constant(2 : i64) : i64
    %109 = llvm.mlir.constant(1 : index) : i64
    %110 = llvm.alloca %109 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %100, %110 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %111 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %112 = llvm.insertvalue %108, %111[0] : !llvm.struct<(i64, ptr)> 
    %113 = llvm.insertvalue %110, %112[1] : !llvm.struct<(i64, ptr)> 
    %114 = llvm.mlir.constant(1 : index) : i64
    %115 = llvm.alloca %114 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %107, %115 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %116 = llvm.alloca %114 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %113, %116 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %117 = llvm.mlir.zero : !llvm.ptr
    %118 = llvm.getelementptr %117[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %119 = llvm.ptrtoint %118 : !llvm.ptr to i64
    llvm.call @memrefCopy(%119, %115, %116) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %101 : !llvm.ptr
    llvm.br ^bb1(%8 : i64)
  ^bb1(%120: i64):  // 2 preds: ^bb0, ^bb8
    %121 = llvm.icmp "slt" %120, %10 : i64
    llvm.cond_br %121, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%8 : i64)
  ^bb3(%122: i64):  // 2 preds: ^bb2, ^bb7
    %123 = llvm.icmp "slt" %122, %10 : i64
    llvm.cond_br %123, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%8 : i64)
  ^bb5(%124: i64):  // 2 preds: ^bb4, ^bb6
    %125 = llvm.icmp "slt" %124, %10 : i64
    llvm.cond_br %125, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %126 = llvm.getelementptr %12[%26] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %127 = llvm.mul %122, %21  : i64
    %128 = llvm.add %127, %120  : i64
    %129 = llvm.getelementptr %126[%128] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %130 = llvm.load %129 : !llvm.ptr -> f64
    %131 = llvm.getelementptr %34[%48] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %132 = llvm.mul %120, %43  : i64
    %133 = llvm.add %132, %124  : i64
    %134 = llvm.getelementptr %131[%133] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %135 = llvm.load %134 : !llvm.ptr -> f64
    %136 = llvm.mul %122, %10  : i64
    %137 = llvm.add %136, %124  : i64
    %138 = llvm.getelementptr %91[%137] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %139 = llvm.load %138 : !llvm.ptr -> f64
    %140 = llvm.fadd %130, %135  : f64
    %141 = llvm.fcmp "olt" %139, %140 : f64
    %142 = llvm.select %141, %139, %140 : i1, f64
    %143 = llvm.mul %122, %10  : i64
    %144 = llvm.add %143, %124  : i64
    %145 = llvm.getelementptr %91[%144] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %142, %145 : f64, !llvm.ptr
    %146 = llvm.add %124, %9  : i64
    llvm.br ^bb5(%146 : i64)
  ^bb7:  // pred: ^bb5
    %147 = llvm.add %122, %9  : i64
    llvm.br ^bb3(%147 : i64)
  ^bb8:  // pred: ^bb3
    %148 = llvm.add %120, %9  : i64
    llvm.br ^bb1(%148 : i64)
  ^bb9:  // pred: ^bb1
    %149 = llvm.mlir.constant(1 : index) : i64
    %150 = llvm.extractvalue %7[3] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %151 = llvm.alloca %149 x !llvm.array<2 x i64> : (i64) -> !llvm.ptr
    llvm.store %150, %151 : !llvm.array<2 x i64>, !llvm.ptr
    %152 = llvm.getelementptr %151[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
    %153 = llvm.load %152 : !llvm.ptr -> i64
    %154 = llvm.mlir.constant(1 : index) : i64
    %155 = llvm.extractvalue %7[3] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %156 = llvm.alloca %154 x !llvm.array<2 x i64> : (i64) -> !llvm.ptr
    llvm.store %155, %156 : !llvm.array<2 x i64>, !llvm.ptr
    %157 = llvm.getelementptr %156[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
    %158 = llvm.load %157 : !llvm.ptr -> i64
    %159 = llvm.mlir.constant(1 : index) : i64
    %160 = llvm.mul %158, %153  : i64
    %161 = llvm.mlir.zero : !llvm.ptr
    %162 = llvm.getelementptr %161[%160] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %163 = llvm.ptrtoint %162 : !llvm.ptr to i64
    %164 = llvm.mlir.constant(64 : index) : i64
    %165 = llvm.add %163, %164  : i64
    %166 = llvm.call @malloc(%165) : (i64) -> !llvm.ptr
    %167 = llvm.ptrtoint %166 : !llvm.ptr to i64
    %168 = llvm.mlir.constant(1 : index) : i64
    %169 = llvm.sub %164, %168  : i64
    %170 = llvm.add %167, %169  : i64
    %171 = llvm.urem %170, %164  : i64
    %172 = llvm.sub %170, %171  : i64
    %173 = llvm.inttoptr %172 : i64 to !llvm.ptr
    %174 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %175 = llvm.insertvalue %166, %174[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %176 = llvm.insertvalue %173, %175[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %177 = llvm.mlir.constant(0 : index) : i64
    %178 = llvm.insertvalue %177, %176[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %179 = llvm.insertvalue %153, %178[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %180 = llvm.insertvalue %158, %179[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %181 = llvm.insertvalue %158, %180[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %182 = llvm.insertvalue %159, %181[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %183 = llvm.mlir.constant(1 : index) : i64
    %184 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %185 = llvm.mul %184, %183  : i64
    %186 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %187 = llvm.mul %185, %186  : i64
    %188 = llvm.mlir.zero : !llvm.ptr
    %189 = llvm.getelementptr %188[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %190 = llvm.ptrtoint %189 : !llvm.ptr to i64
    %191 = llvm.mul %187, %190  : i64
    %192 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %193 = llvm.extractvalue %7[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %194 = llvm.getelementptr %192[%193] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %195 = llvm.getelementptr %173[%177] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%195, %194, %191) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %196 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %197 = llvm.insertvalue %166, %196[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %198 = llvm.insertvalue %173, %197[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %199 = llvm.mlir.constant(0 : index) : i64
    %200 = llvm.insertvalue %199, %198[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %201 = llvm.insertvalue %10, %200[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %202 = llvm.insertvalue %158, %201[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %203 = llvm.insertvalue %10, %202[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %204 = llvm.mlir.constant(1 : index) : i64
    %205 = llvm.insertvalue %204, %203[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %206 = llvm.intr.stacksave : !llvm.ptr
    %207 = llvm.mlir.constant(2 : i64) : i64
    %208 = llvm.mlir.constant(1 : index) : i64
    %209 = llvm.alloca %208 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %100, %209 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %210 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %211 = llvm.insertvalue %207, %210[0] : !llvm.struct<(i64, ptr)> 
    %212 = llvm.insertvalue %209, %211[1] : !llvm.struct<(i64, ptr)> 
    %213 = llvm.mlir.constant(2 : i64) : i64
    %214 = llvm.mlir.constant(1 : index) : i64
    %215 = llvm.alloca %214 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %205, %215 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %216 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %217 = llvm.insertvalue %213, %216[0] : !llvm.struct<(i64, ptr)> 
    %218 = llvm.insertvalue %215, %217[1] : !llvm.struct<(i64, ptr)> 
    %219 = llvm.mlir.constant(1 : index) : i64
    %220 = llvm.alloca %219 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %212, %220 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %221 = llvm.alloca %219 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %218, %221 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %222 = llvm.mlir.zero : !llvm.ptr
    %223 = llvm.getelementptr %222[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %224 = llvm.ptrtoint %223 : !llvm.ptr to i64
    llvm.call @memrefCopy(%224, %220, %221) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %206 : !llvm.ptr
    %225 = llvm.mlir.constant(1 : index) : i64
    %226 = llvm.mul %153, %225  : i64
    %227 = llvm.mul %226, %158  : i64
    %228 = llvm.mlir.zero : !llvm.ptr
    %229 = llvm.getelementptr %228[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %230 = llvm.ptrtoint %229 : !llvm.ptr to i64
    %231 = llvm.mul %227, %230  : i64
    %232 = llvm.getelementptr %173[%177] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %233 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %234 = llvm.extractvalue %7[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %235 = llvm.getelementptr %233[%234] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%235, %232, %231) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.return
  }
}

