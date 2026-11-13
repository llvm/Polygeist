module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.func @memrefCopy(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @kernel_cholesky(%arg0: i32, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64) {
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
    %11 = llvm.sub %10, %9  : i64
    %12 = llvm.mlir.constant(1 : index) : i64
    %13 = llvm.extractvalue %7[3] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.alloca %12 x !llvm.array<2 x i64> : (i64) -> !llvm.ptr
    llvm.store %13, %14 : !llvm.array<2 x i64>, !llvm.ptr
    %15 = llvm.getelementptr %14[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
    %16 = llvm.load %15 : !llvm.ptr -> i64
    %17 = llvm.mlir.constant(1 : index) : i64
    %18 = llvm.extractvalue %7[3] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %19 = llvm.alloca %17 x !llvm.array<2 x i64> : (i64) -> !llvm.ptr
    llvm.store %18, %19 : !llvm.array<2 x i64>, !llvm.ptr
    %20 = llvm.getelementptr %19[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
    %21 = llvm.load %20 : !llvm.ptr -> i64
    %22 = llvm.mlir.constant(1 : index) : i64
    %23 = llvm.mul %21, %16  : i64
    %24 = llvm.mlir.zero : !llvm.ptr
    %25 = llvm.getelementptr %24[%23] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %26 = llvm.ptrtoint %25 : !llvm.ptr to i64
    %27 = llvm.mlir.constant(64 : index) : i64
    %28 = llvm.add %26, %27  : i64
    %29 = llvm.call @malloc(%28) : (i64) -> !llvm.ptr
    %30 = llvm.ptrtoint %29 : !llvm.ptr to i64
    %31 = llvm.mlir.constant(1 : index) : i64
    %32 = llvm.sub %27, %31  : i64
    %33 = llvm.add %30, %32  : i64
    %34 = llvm.urem %33, %27  : i64
    %35 = llvm.sub %33, %34  : i64
    %36 = llvm.inttoptr %35 : i64 to !llvm.ptr
    %37 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %38 = llvm.insertvalue %29, %37[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %39 = llvm.insertvalue %36, %38[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %40 = llvm.mlir.constant(0 : index) : i64
    %41 = llvm.insertvalue %40, %39[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %42 = llvm.insertvalue %16, %41[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %43 = llvm.insertvalue %21, %42[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %44 = llvm.insertvalue %21, %43[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %45 = llvm.insertvalue %22, %44[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %46 = llvm.mlir.constant(1 : index) : i64
    %47 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %48 = llvm.mul %47, %46  : i64
    %49 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %50 = llvm.mul %48, %49  : i64
    %51 = llvm.mlir.zero : !llvm.ptr
    %52 = llvm.getelementptr %51[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %53 = llvm.ptrtoint %52 : !llvm.ptr to i64
    %54 = llvm.mul %50, %53  : i64
    %55 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %56 = llvm.extractvalue %7[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %57 = llvm.getelementptr %55[%56] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %58 = llvm.getelementptr %36[%40] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%58, %57, %54) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb1(%8, %45 : i64, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb1(%59: i64, %60: !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>):  // 2 preds: ^bb0, ^bb11
    %61 = llvm.icmp "slt" %59, %10 : i64
    llvm.cond_br %61, ^bb2, ^bb12
  ^bb2:  // pred: ^bb1
    %62 = llvm.sub %59, %9  : i64
    llvm.br ^bb3(%8, %60 : i64, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb3(%63: i64, %64: !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>):  // 2 preds: ^bb2, ^bb7
    %65 = llvm.icmp "slt" %63, %59 : i64
    llvm.cond_br %65, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    %66 = llvm.extractvalue %64[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %67 = llvm.extractvalue %64[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %68 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %69 = llvm.insertvalue %66, %68[0] : !llvm.struct<(ptr, ptr, i64)> 
    %70 = llvm.insertvalue %67, %69[1] : !llvm.struct<(ptr, ptr, i64)> 
    %71 = llvm.mlir.constant(0 : index) : i64
    %72 = llvm.insertvalue %71, %70[2] : !llvm.struct<(ptr, ptr, i64)> 
    %73 = llvm.extractvalue %64[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %74 = llvm.extractvalue %64[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %75 = llvm.extractvalue %64[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %76 = llvm.extractvalue %64[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %77 = llvm.extractvalue %64[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %78 = llvm.mul %59, %76  : i64
    %79 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %80 = llvm.insertvalue %66, %79[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %81 = llvm.insertvalue %67, %80[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %82 = llvm.insertvalue %78, %81[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %83 = llvm.insertvalue %62, %82[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %84 = llvm.mlir.constant(1 : index) : i64
    %85 = llvm.insertvalue %84, %83[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %86 = llvm.mlir.constant(1 : index) : i64
    %87 = llvm.mlir.zero : !llvm.ptr
    %88 = llvm.getelementptr %87[%62] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %89 = llvm.ptrtoint %88 : !llvm.ptr to i64
    %90 = llvm.mlir.constant(64 : index) : i64
    %91 = llvm.add %89, %90  : i64
    %92 = llvm.call @malloc(%91) : (i64) -> !llvm.ptr
    %93 = llvm.ptrtoint %92 : !llvm.ptr to i64
    %94 = llvm.mlir.constant(1 : index) : i64
    %95 = llvm.sub %90, %94  : i64
    %96 = llvm.add %93, %95  : i64
    %97 = llvm.urem %96, %90  : i64
    %98 = llvm.sub %96, %97  : i64
    %99 = llvm.inttoptr %98 : i64 to !llvm.ptr
    %100 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %101 = llvm.insertvalue %92, %100[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %102 = llvm.insertvalue %99, %101[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %103 = llvm.mlir.constant(0 : index) : i64
    %104 = llvm.insertvalue %103, %102[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %105 = llvm.insertvalue %62, %104[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %106 = llvm.insertvalue %86, %105[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %107 = llvm.intr.stacksave : !llvm.ptr
    %108 = llvm.mlir.constant(1 : i64) : i64
    %109 = llvm.mlir.constant(1 : index) : i64
    %110 = llvm.alloca %109 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %85, %110 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %111 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %112 = llvm.insertvalue %108, %111[0] : !llvm.struct<(i64, ptr)> 
    %113 = llvm.insertvalue %110, %112[1] : !llvm.struct<(i64, ptr)> 
    %114 = llvm.mlir.constant(1 : i64) : i64
    %115 = llvm.mlir.constant(1 : index) : i64
    %116 = llvm.alloca %115 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %106, %116 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %117 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %118 = llvm.insertvalue %114, %117[0] : !llvm.struct<(i64, ptr)> 
    %119 = llvm.insertvalue %116, %118[1] : !llvm.struct<(i64, ptr)> 
    %120 = llvm.mlir.constant(1 : index) : i64
    %121 = llvm.alloca %120 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %113, %121 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %122 = llvm.alloca %120 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %119, %122 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %123 = llvm.mlir.zero : !llvm.ptr
    %124 = llvm.getelementptr %123[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %125 = llvm.ptrtoint %124 : !llvm.ptr to i64
    llvm.call @memrefCopy(%125, %121, %122) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %107 : !llvm.ptr
    %126 = llvm.extractvalue %64[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %127 = llvm.extractvalue %64[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %128 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %129 = llvm.insertvalue %126, %128[0] : !llvm.struct<(ptr, ptr, i64)> 
    %130 = llvm.insertvalue %127, %129[1] : !llvm.struct<(ptr, ptr, i64)> 
    %131 = llvm.mlir.constant(0 : index) : i64
    %132 = llvm.insertvalue %131, %130[2] : !llvm.struct<(ptr, ptr, i64)> 
    %133 = llvm.extractvalue %64[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %134 = llvm.extractvalue %64[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %135 = llvm.extractvalue %64[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %136 = llvm.extractvalue %64[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %137 = llvm.extractvalue %64[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %138 = llvm.mul %63, %136  : i64
    %139 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %140 = llvm.insertvalue %126, %139[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %141 = llvm.insertvalue %127, %140[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %142 = llvm.insertvalue %138, %141[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %143 = llvm.insertvalue %62, %142[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %144 = llvm.mlir.constant(1 : index) : i64
    %145 = llvm.insertvalue %144, %143[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %146 = llvm.mlir.constant(1 : index) : i64
    %147 = llvm.mlir.zero : !llvm.ptr
    %148 = llvm.getelementptr %147[%62] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %149 = llvm.ptrtoint %148 : !llvm.ptr to i64
    %150 = llvm.mlir.constant(64 : index) : i64
    %151 = llvm.add %149, %150  : i64
    %152 = llvm.call @malloc(%151) : (i64) -> !llvm.ptr
    %153 = llvm.ptrtoint %152 : !llvm.ptr to i64
    %154 = llvm.mlir.constant(1 : index) : i64
    %155 = llvm.sub %150, %154  : i64
    %156 = llvm.add %153, %155  : i64
    %157 = llvm.urem %156, %150  : i64
    %158 = llvm.sub %156, %157  : i64
    %159 = llvm.inttoptr %158 : i64 to !llvm.ptr
    %160 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %161 = llvm.insertvalue %152, %160[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %162 = llvm.insertvalue %159, %161[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %163 = llvm.mlir.constant(0 : index) : i64
    %164 = llvm.insertvalue %163, %162[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %165 = llvm.insertvalue %62, %164[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %166 = llvm.insertvalue %146, %165[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %167 = llvm.intr.stacksave : !llvm.ptr
    %168 = llvm.mlir.constant(1 : i64) : i64
    %169 = llvm.mlir.constant(1 : index) : i64
    %170 = llvm.alloca %169 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %145, %170 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %171 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %172 = llvm.insertvalue %168, %171[0] : !llvm.struct<(i64, ptr)> 
    %173 = llvm.insertvalue %170, %172[1] : !llvm.struct<(i64, ptr)> 
    %174 = llvm.mlir.constant(1 : i64) : i64
    %175 = llvm.mlir.constant(1 : index) : i64
    %176 = llvm.alloca %175 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %166, %176 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %177 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %178 = llvm.insertvalue %174, %177[0] : !llvm.struct<(i64, ptr)> 
    %179 = llvm.insertvalue %176, %178[1] : !llvm.struct<(i64, ptr)> 
    %180 = llvm.mlir.constant(1 : index) : i64
    %181 = llvm.alloca %180 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %173, %181 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %182 = llvm.alloca %180 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %179, %182 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %183 = llvm.mlir.zero : !llvm.ptr
    %184 = llvm.getelementptr %183[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %185 = llvm.ptrtoint %184 : !llvm.ptr to i64
    llvm.call @memrefCopy(%185, %181, %182) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %167 : !llvm.ptr
    %186 = llvm.extractvalue %64[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %187 = llvm.extractvalue %64[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %188 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %189 = llvm.insertvalue %186, %188[0] : !llvm.struct<(ptr, ptr, i64)> 
    %190 = llvm.insertvalue %187, %189[1] : !llvm.struct<(ptr, ptr, i64)> 
    %191 = llvm.mlir.constant(0 : index) : i64
    %192 = llvm.insertvalue %191, %190[2] : !llvm.struct<(ptr, ptr, i64)> 
    %193 = llvm.extractvalue %64[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %194 = llvm.extractvalue %64[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %195 = llvm.extractvalue %64[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %196 = llvm.extractvalue %64[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %197 = llvm.extractvalue %64[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %198 = llvm.mul %59, %196  : i64
    %199 = llvm.add %198, %63  : i64
    %200 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %201 = llvm.insertvalue %186, %200[0] : !llvm.struct<(ptr, ptr, i64)> 
    %202 = llvm.insertvalue %187, %201[1] : !llvm.struct<(ptr, ptr, i64)> 
    %203 = llvm.insertvalue %199, %202[2] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.br ^bb5(%8 : i64)
  ^bb5(%204: i64):  // 2 preds: ^bb4, ^bb6
    %205 = llvm.icmp "slt" %204, %62 : i64
    llvm.cond_br %205, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %206 = llvm.getelementptr %99[%204] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %207 = llvm.load %206 : !llvm.ptr -> f64
    %208 = llvm.getelementptr %159[%204] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %209 = llvm.load %208 : !llvm.ptr -> f64
    %210 = llvm.getelementptr %187[%199] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %211 = llvm.load %210 : !llvm.ptr -> f64
    %212 = llvm.fmul %207, %209  : f64
    %213 = llvm.fsub %211, %212  : f64
    %214 = llvm.icmp "slt" %204, %63 : i64
    %215 = llvm.select %214, %213, %211 : i1, f64
    %216 = llvm.getelementptr %187[%199] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %215, %216 : f64, !llvm.ptr
    %217 = llvm.add %204, %9  : i64
    llvm.br ^bb5(%217 : i64)
  ^bb7:  // pred: ^bb5
    %218 = llvm.extractvalue %64[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %219 = llvm.extractvalue %64[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %220 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %221 = llvm.insertvalue %218, %220[0] : !llvm.struct<(ptr, ptr, i64)> 
    %222 = llvm.insertvalue %219, %221[1] : !llvm.struct<(ptr, ptr, i64)> 
    %223 = llvm.mlir.constant(0 : index) : i64
    %224 = llvm.insertvalue %223, %222[2] : !llvm.struct<(ptr, ptr, i64)> 
    %225 = llvm.extractvalue %64[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %226 = llvm.extractvalue %64[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %227 = llvm.extractvalue %64[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %228 = llvm.extractvalue %64[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %229 = llvm.extractvalue %64[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %230 = llvm.mul %59, %228  : i64
    %231 = llvm.add %230, %63  : i64
    %232 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %233 = llvm.insertvalue %218, %232[0] : !llvm.struct<(ptr, ptr, i64)> 
    %234 = llvm.insertvalue %219, %233[1] : !llvm.struct<(ptr, ptr, i64)> 
    %235 = llvm.insertvalue %231, %234[2] : !llvm.struct<(ptr, ptr, i64)> 
    %236 = llvm.mlir.constant(1 : index) : i64
    %237 = llvm.mlir.zero : !llvm.ptr
    %238 = llvm.getelementptr %237[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %239 = llvm.ptrtoint %238 : !llvm.ptr to i64
    %240 = llvm.mul %239, %236  : i64
    %241 = llvm.getelementptr %187[%199] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %242 = llvm.getelementptr %219[%231] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%242, %241, %240) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %243 = llvm.extractvalue %64[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %244 = llvm.extractvalue %64[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %245 = llvm.mul %63, %244  : i64
    %246 = llvm.add %245, %63  : i64
    %247 = llvm.getelementptr %243[%246] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %248 = llvm.load %247 : !llvm.ptr -> f64
    %249 = llvm.extractvalue %64[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %250 = llvm.extractvalue %64[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %251 = llvm.mul %59, %250  : i64
    %252 = llvm.add %251, %63  : i64
    %253 = llvm.getelementptr %249[%252] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %254 = llvm.load %253 : !llvm.ptr -> f64
    %255 = llvm.fdiv %254, %248  : f64
    %256 = llvm.extractvalue %64[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %257 = llvm.extractvalue %64[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %258 = llvm.mul %59, %257  : i64
    %259 = llvm.add %258, %63  : i64
    %260 = llvm.getelementptr %256[%259] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %255, %260 : f64, !llvm.ptr
    %261 = llvm.add %63, %9  : i64
    llvm.br ^bb3(%261, %64 : i64, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb8:  // pred: ^bb3
    %262 = llvm.extractvalue %64[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %263 = llvm.extractvalue %64[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %264 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %265 = llvm.insertvalue %262, %264[0] : !llvm.struct<(ptr, ptr, i64)> 
    %266 = llvm.insertvalue %263, %265[1] : !llvm.struct<(ptr, ptr, i64)> 
    %267 = llvm.mlir.constant(0 : index) : i64
    %268 = llvm.insertvalue %267, %266[2] : !llvm.struct<(ptr, ptr, i64)> 
    %269 = llvm.extractvalue %64[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %270 = llvm.extractvalue %64[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %271 = llvm.extractvalue %64[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %272 = llvm.extractvalue %64[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %273 = llvm.extractvalue %64[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %274 = llvm.mul %59, %272  : i64
    %275 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %276 = llvm.insertvalue %262, %275[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %277 = llvm.insertvalue %263, %276[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %278 = llvm.insertvalue %274, %277[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %279 = llvm.insertvalue %11, %278[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %280 = llvm.mlir.constant(1 : index) : i64
    %281 = llvm.insertvalue %280, %279[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %282 = llvm.mlir.constant(1 : index) : i64
    %283 = llvm.mlir.zero : !llvm.ptr
    %284 = llvm.getelementptr %283[%11] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %285 = llvm.ptrtoint %284 : !llvm.ptr to i64
    %286 = llvm.mlir.constant(64 : index) : i64
    %287 = llvm.add %285, %286  : i64
    %288 = llvm.call @malloc(%287) : (i64) -> !llvm.ptr
    %289 = llvm.ptrtoint %288 : !llvm.ptr to i64
    %290 = llvm.mlir.constant(1 : index) : i64
    %291 = llvm.sub %286, %290  : i64
    %292 = llvm.add %289, %291  : i64
    %293 = llvm.urem %292, %286  : i64
    %294 = llvm.sub %292, %293  : i64
    %295 = llvm.inttoptr %294 : i64 to !llvm.ptr
    %296 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %297 = llvm.insertvalue %288, %296[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %298 = llvm.insertvalue %295, %297[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %299 = llvm.mlir.constant(0 : index) : i64
    %300 = llvm.insertvalue %299, %298[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %301 = llvm.insertvalue %11, %300[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %302 = llvm.insertvalue %282, %301[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %303 = llvm.intr.stacksave : !llvm.ptr
    %304 = llvm.mlir.constant(1 : i64) : i64
    %305 = llvm.mlir.constant(1 : index) : i64
    %306 = llvm.alloca %305 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %281, %306 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %307 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %308 = llvm.insertvalue %304, %307[0] : !llvm.struct<(i64, ptr)> 
    %309 = llvm.insertvalue %306, %308[1] : !llvm.struct<(i64, ptr)> 
    %310 = llvm.mlir.constant(1 : i64) : i64
    %311 = llvm.mlir.constant(1 : index) : i64
    %312 = llvm.alloca %311 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %302, %312 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %313 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %314 = llvm.insertvalue %310, %313[0] : !llvm.struct<(i64, ptr)> 
    %315 = llvm.insertvalue %312, %314[1] : !llvm.struct<(i64, ptr)> 
    %316 = llvm.mlir.constant(1 : index) : i64
    %317 = llvm.alloca %316 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %309, %317 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %318 = llvm.alloca %316 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %315, %318 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %319 = llvm.mlir.zero : !llvm.ptr
    %320 = llvm.getelementptr %319[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %321 = llvm.ptrtoint %320 : !llvm.ptr to i64
    llvm.call @memrefCopy(%321, %317, %318) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %303 : !llvm.ptr
    %322 = llvm.extractvalue %64[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %323 = llvm.extractvalue %64[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %324 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %325 = llvm.insertvalue %322, %324[0] : !llvm.struct<(ptr, ptr, i64)> 
    %326 = llvm.insertvalue %323, %325[1] : !llvm.struct<(ptr, ptr, i64)> 
    %327 = llvm.mlir.constant(0 : index) : i64
    %328 = llvm.insertvalue %327, %326[2] : !llvm.struct<(ptr, ptr, i64)> 
    %329 = llvm.extractvalue %64[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %330 = llvm.extractvalue %64[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %331 = llvm.extractvalue %64[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %332 = llvm.extractvalue %64[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %333 = llvm.extractvalue %64[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %334 = llvm.mul %59, %332  : i64
    %335 = llvm.add %59, %334  : i64
    %336 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %337 = llvm.insertvalue %322, %336[0] : !llvm.struct<(ptr, ptr, i64)> 
    %338 = llvm.insertvalue %323, %337[1] : !llvm.struct<(ptr, ptr, i64)> 
    %339 = llvm.insertvalue %335, %338[2] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.br ^bb9(%8 : i64)
  ^bb9(%340: i64):  // 2 preds: ^bb8, ^bb10
    %341 = llvm.icmp "slt" %340, %11 : i64
    llvm.cond_br %341, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    %342 = llvm.getelementptr %295[%340] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %343 = llvm.load %342 : !llvm.ptr -> f64
    %344 = llvm.getelementptr %323[%335] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %345 = llvm.load %344 : !llvm.ptr -> f64
    %346 = llvm.fmul %343, %343  : f64
    %347 = llvm.fsub %345, %346  : f64
    %348 = llvm.icmp "slt" %340, %59 : i64
    %349 = llvm.select %348, %347, %345 : i1, f64
    %350 = llvm.getelementptr %323[%335] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %349, %350 : f64, !llvm.ptr
    %351 = llvm.add %340, %9  : i64
    llvm.br ^bb9(%351 : i64)
  ^bb11:  // pred: ^bb9
    %352 = llvm.extractvalue %64[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %353 = llvm.extractvalue %64[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %354 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %355 = llvm.insertvalue %352, %354[0] : !llvm.struct<(ptr, ptr, i64)> 
    %356 = llvm.insertvalue %353, %355[1] : !llvm.struct<(ptr, ptr, i64)> 
    %357 = llvm.mlir.constant(0 : index) : i64
    %358 = llvm.insertvalue %357, %356[2] : !llvm.struct<(ptr, ptr, i64)> 
    %359 = llvm.extractvalue %64[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %360 = llvm.extractvalue %64[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %361 = llvm.extractvalue %64[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %362 = llvm.extractvalue %64[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %363 = llvm.extractvalue %64[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %364 = llvm.mul %59, %362  : i64
    %365 = llvm.add %59, %364  : i64
    %366 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %367 = llvm.insertvalue %352, %366[0] : !llvm.struct<(ptr, ptr, i64)> 
    %368 = llvm.insertvalue %353, %367[1] : !llvm.struct<(ptr, ptr, i64)> 
    %369 = llvm.insertvalue %365, %368[2] : !llvm.struct<(ptr, ptr, i64)> 
    %370 = llvm.mlir.constant(1 : index) : i64
    %371 = llvm.mlir.zero : !llvm.ptr
    %372 = llvm.getelementptr %371[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %373 = llvm.ptrtoint %372 : !llvm.ptr to i64
    %374 = llvm.mul %373, %370  : i64
    %375 = llvm.getelementptr %323[%335] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %376 = llvm.getelementptr %353[%365] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%376, %375, %374) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %377 = llvm.extractvalue %64[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %378 = llvm.extractvalue %64[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %379 = llvm.mul %59, %378  : i64
    %380 = llvm.add %379, %59  : i64
    %381 = llvm.getelementptr %377[%380] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %382 = llvm.load %381 : !llvm.ptr -> f64
    %383 = llvm.intr.sqrt(%382)  : (f64) -> f64
    %384 = llvm.extractvalue %64[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %385 = llvm.extractvalue %64[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %386 = llvm.mul %59, %385  : i64
    %387 = llvm.add %386, %59  : i64
    %388 = llvm.getelementptr %384[%387] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %383, %388 : f64, !llvm.ptr
    %389 = llvm.add %59, %9  : i64
    llvm.br ^bb1(%389, %64 : i64, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb12:  // pred: ^bb1
    %390 = llvm.mlir.constant(1 : index) : i64
    %391 = llvm.extractvalue %60[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %392 = llvm.mul %391, %390  : i64
    %393 = llvm.extractvalue %60[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %394 = llvm.mul %392, %393  : i64
    %395 = llvm.mlir.zero : !llvm.ptr
    %396 = llvm.getelementptr %395[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %397 = llvm.ptrtoint %396 : !llvm.ptr to i64
    %398 = llvm.mul %394, %397  : i64
    %399 = llvm.extractvalue %60[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %400 = llvm.extractvalue %60[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %401 = llvm.getelementptr %399[%400] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %402 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %403 = llvm.extractvalue %7[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %404 = llvm.getelementptr %402[%403] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%404, %401, %398) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.return
  }
}

