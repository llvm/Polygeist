module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.func @memrefCopy(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @kernel_lu(%arg0: i32, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64) {
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
  ^bb1(%59: i64, %60: !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>):  // 2 preds: ^bb0, ^bb14
    %61 = llvm.icmp "slt" %59, %10 : i64
    llvm.cond_br %61, ^bb2, ^bb15
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
    %138 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %139 = llvm.insertvalue %126, %138[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %140 = llvm.insertvalue %127, %139[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %141 = llvm.insertvalue %63, %140[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %142 = llvm.insertvalue %62, %141[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %143 = llvm.insertvalue %136, %142[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %144 = llvm.mlir.constant(1 : index) : i64
    %145 = llvm.mlir.zero : !llvm.ptr
    %146 = llvm.getelementptr %145[%62] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %147 = llvm.ptrtoint %146 : !llvm.ptr to i64
    %148 = llvm.mlir.constant(64 : index) : i64
    %149 = llvm.add %147, %148  : i64
    %150 = llvm.call @malloc(%149) : (i64) -> !llvm.ptr
    %151 = llvm.ptrtoint %150 : !llvm.ptr to i64
    %152 = llvm.mlir.constant(1 : index) : i64
    %153 = llvm.sub %148, %152  : i64
    %154 = llvm.add %151, %153  : i64
    %155 = llvm.urem %154, %148  : i64
    %156 = llvm.sub %154, %155  : i64
    %157 = llvm.inttoptr %156 : i64 to !llvm.ptr
    %158 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %159 = llvm.insertvalue %150, %158[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %160 = llvm.insertvalue %157, %159[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %161 = llvm.mlir.constant(0 : index) : i64
    %162 = llvm.insertvalue %161, %160[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %163 = llvm.insertvalue %62, %162[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %164 = llvm.insertvalue %144, %163[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %165 = llvm.intr.stacksave : !llvm.ptr
    %166 = llvm.mlir.constant(1 : i64) : i64
    %167 = llvm.mlir.constant(1 : index) : i64
    %168 = llvm.alloca %167 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %143, %168 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %169 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %170 = llvm.insertvalue %166, %169[0] : !llvm.struct<(i64, ptr)> 
    %171 = llvm.insertvalue %168, %170[1] : !llvm.struct<(i64, ptr)> 
    %172 = llvm.mlir.constant(1 : i64) : i64
    %173 = llvm.mlir.constant(1 : index) : i64
    %174 = llvm.alloca %173 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %164, %174 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %175 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %176 = llvm.insertvalue %172, %175[0] : !llvm.struct<(i64, ptr)> 
    %177 = llvm.insertvalue %174, %176[1] : !llvm.struct<(i64, ptr)> 
    %178 = llvm.mlir.constant(1 : index) : i64
    %179 = llvm.alloca %178 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %171, %179 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %180 = llvm.alloca %178 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %177, %180 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %181 = llvm.mlir.zero : !llvm.ptr
    %182 = llvm.getelementptr %181[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %183 = llvm.ptrtoint %182 : !llvm.ptr to i64
    llvm.call @memrefCopy(%183, %179, %180) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %165 : !llvm.ptr
    %184 = llvm.extractvalue %64[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %185 = llvm.extractvalue %64[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %186 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %187 = llvm.insertvalue %184, %186[0] : !llvm.struct<(ptr, ptr, i64)> 
    %188 = llvm.insertvalue %185, %187[1] : !llvm.struct<(ptr, ptr, i64)> 
    %189 = llvm.mlir.constant(0 : index) : i64
    %190 = llvm.insertvalue %189, %188[2] : !llvm.struct<(ptr, ptr, i64)> 
    %191 = llvm.extractvalue %64[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %192 = llvm.extractvalue %64[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %193 = llvm.extractvalue %64[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %194 = llvm.extractvalue %64[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %195 = llvm.extractvalue %64[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %196 = llvm.mul %59, %194  : i64
    %197 = llvm.add %196, %63  : i64
    %198 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %199 = llvm.insertvalue %184, %198[0] : !llvm.struct<(ptr, ptr, i64)> 
    %200 = llvm.insertvalue %185, %199[1] : !llvm.struct<(ptr, ptr, i64)> 
    %201 = llvm.insertvalue %197, %200[2] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.br ^bb5(%8 : i64)
  ^bb5(%202: i64):  // 2 preds: ^bb4, ^bb6
    %203 = llvm.icmp "slt" %202, %62 : i64
    llvm.cond_br %203, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %204 = llvm.getelementptr %99[%202] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %205 = llvm.load %204 : !llvm.ptr -> f64
    %206 = llvm.getelementptr %157[%202] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %207 = llvm.load %206 : !llvm.ptr -> f64
    %208 = llvm.getelementptr %185[%197] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %209 = llvm.load %208 : !llvm.ptr -> f64
    %210 = llvm.fmul %205, %207  : f64
    %211 = llvm.fsub %209, %210  : f64
    %212 = llvm.icmp "slt" %202, %63 : i64
    %213 = llvm.select %212, %211, %209 : i1, f64
    %214 = llvm.getelementptr %185[%197] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %213, %214 : f64, !llvm.ptr
    %215 = llvm.add %202, %9  : i64
    llvm.br ^bb5(%215 : i64)
  ^bb7:  // pred: ^bb5
    %216 = llvm.extractvalue %64[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %217 = llvm.extractvalue %64[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %218 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %219 = llvm.insertvalue %216, %218[0] : !llvm.struct<(ptr, ptr, i64)> 
    %220 = llvm.insertvalue %217, %219[1] : !llvm.struct<(ptr, ptr, i64)> 
    %221 = llvm.mlir.constant(0 : index) : i64
    %222 = llvm.insertvalue %221, %220[2] : !llvm.struct<(ptr, ptr, i64)> 
    %223 = llvm.extractvalue %64[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %224 = llvm.extractvalue %64[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %225 = llvm.extractvalue %64[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %226 = llvm.extractvalue %64[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %227 = llvm.extractvalue %64[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %228 = llvm.mul %59, %226  : i64
    %229 = llvm.add %228, %63  : i64
    %230 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %231 = llvm.insertvalue %216, %230[0] : !llvm.struct<(ptr, ptr, i64)> 
    %232 = llvm.insertvalue %217, %231[1] : !llvm.struct<(ptr, ptr, i64)> 
    %233 = llvm.insertvalue %229, %232[2] : !llvm.struct<(ptr, ptr, i64)> 
    %234 = llvm.mlir.constant(1 : index) : i64
    %235 = llvm.mlir.zero : !llvm.ptr
    %236 = llvm.getelementptr %235[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %237 = llvm.ptrtoint %236 : !llvm.ptr to i64
    %238 = llvm.mul %237, %234  : i64
    %239 = llvm.getelementptr %185[%197] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %240 = llvm.getelementptr %217[%229] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%240, %239, %238) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %241 = llvm.extractvalue %64[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %242 = llvm.extractvalue %64[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %243 = llvm.mul %63, %242  : i64
    %244 = llvm.add %243, %63  : i64
    %245 = llvm.getelementptr %241[%244] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %246 = llvm.load %245 : !llvm.ptr -> f64
    %247 = llvm.extractvalue %64[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %248 = llvm.extractvalue %64[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %249 = llvm.mul %59, %248  : i64
    %250 = llvm.add %249, %63  : i64
    %251 = llvm.getelementptr %247[%250] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %252 = llvm.load %251 : !llvm.ptr -> f64
    %253 = llvm.fdiv %252, %246  : f64
    %254 = llvm.extractvalue %64[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %255 = llvm.extractvalue %64[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %256 = llvm.mul %59, %255  : i64
    %257 = llvm.add %256, %63  : i64
    %258 = llvm.getelementptr %254[%257] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %253, %258 : f64, !llvm.ptr
    %259 = llvm.add %63, %9  : i64
    llvm.br ^bb3(%259, %64 : i64, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb8:  // pred: ^bb3
    %260 = llvm.extractvalue %64[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %261 = llvm.extractvalue %64[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %262 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %263 = llvm.insertvalue %260, %262[0] : !llvm.struct<(ptr, ptr, i64)> 
    %264 = llvm.insertvalue %261, %263[1] : !llvm.struct<(ptr, ptr, i64)> 
    %265 = llvm.mlir.constant(0 : index) : i64
    %266 = llvm.insertvalue %265, %264[2] : !llvm.struct<(ptr, ptr, i64)> 
    %267 = llvm.extractvalue %64[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %268 = llvm.extractvalue %64[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %269 = llvm.extractvalue %64[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %270 = llvm.extractvalue %64[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %271 = llvm.extractvalue %64[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %272 = llvm.mul %59, %270  : i64
    %273 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %274 = llvm.insertvalue %260, %273[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %275 = llvm.insertvalue %261, %274[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %276 = llvm.insertvalue %272, %275[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %277 = llvm.insertvalue %11, %276[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %278 = llvm.mlir.constant(1 : index) : i64
    %279 = llvm.insertvalue %278, %277[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %280 = llvm.mlir.constant(1 : index) : i64
    %281 = llvm.mlir.zero : !llvm.ptr
    %282 = llvm.getelementptr %281[%11] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %283 = llvm.ptrtoint %282 : !llvm.ptr to i64
    %284 = llvm.mlir.constant(64 : index) : i64
    %285 = llvm.add %283, %284  : i64
    %286 = llvm.call @malloc(%285) : (i64) -> !llvm.ptr
    %287 = llvm.ptrtoint %286 : !llvm.ptr to i64
    %288 = llvm.mlir.constant(1 : index) : i64
    %289 = llvm.sub %284, %288  : i64
    %290 = llvm.add %287, %289  : i64
    %291 = llvm.urem %290, %284  : i64
    %292 = llvm.sub %290, %291  : i64
    %293 = llvm.inttoptr %292 : i64 to !llvm.ptr
    %294 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %295 = llvm.insertvalue %286, %294[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %296 = llvm.insertvalue %293, %295[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %297 = llvm.mlir.constant(0 : index) : i64
    %298 = llvm.insertvalue %297, %296[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %299 = llvm.insertvalue %11, %298[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %300 = llvm.insertvalue %280, %299[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %301 = llvm.intr.stacksave : !llvm.ptr
    %302 = llvm.mlir.constant(1 : i64) : i64
    %303 = llvm.mlir.constant(1 : index) : i64
    %304 = llvm.alloca %303 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %279, %304 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %305 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %306 = llvm.insertvalue %302, %305[0] : !llvm.struct<(i64, ptr)> 
    %307 = llvm.insertvalue %304, %306[1] : !llvm.struct<(i64, ptr)> 
    %308 = llvm.mlir.constant(1 : i64) : i64
    %309 = llvm.mlir.constant(1 : index) : i64
    %310 = llvm.alloca %309 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %300, %310 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %311 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %312 = llvm.insertvalue %308, %311[0] : !llvm.struct<(i64, ptr)> 
    %313 = llvm.insertvalue %310, %312[1] : !llvm.struct<(i64, ptr)> 
    %314 = llvm.mlir.constant(1 : index) : i64
    %315 = llvm.alloca %314 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %307, %315 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %316 = llvm.alloca %314 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %313, %316 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %317 = llvm.mlir.zero : !llvm.ptr
    %318 = llvm.getelementptr %317[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %319 = llvm.ptrtoint %318 : !llvm.ptr to i64
    llvm.call @memrefCopy(%319, %315, %316) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %301 : !llvm.ptr
    %320 = llvm.extractvalue %64[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %321 = llvm.extractvalue %64[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %322 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %323 = llvm.insertvalue %320, %322[0] : !llvm.struct<(ptr, ptr, i64)> 
    %324 = llvm.insertvalue %321, %323[1] : !llvm.struct<(ptr, ptr, i64)> 
    %325 = llvm.mlir.constant(0 : index) : i64
    %326 = llvm.insertvalue %325, %324[2] : !llvm.struct<(ptr, ptr, i64)> 
    %327 = llvm.extractvalue %64[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %328 = llvm.extractvalue %64[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %329 = llvm.extractvalue %64[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %330 = llvm.extractvalue %64[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %331 = llvm.extractvalue %64[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %332 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %333 = llvm.insertvalue %320, %332[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %334 = llvm.insertvalue %321, %333[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %335 = llvm.mlir.constant(0 : index) : i64
    %336 = llvm.insertvalue %335, %334[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %337 = llvm.insertvalue %11, %336[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %338 = llvm.insertvalue %330, %337[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %339 = llvm.insertvalue %10, %338[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %340 = llvm.mlir.constant(1 : index) : i64
    %341 = llvm.insertvalue %340, %339[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %342 = llvm.mlir.constant(1 : index) : i64
    %343 = llvm.mul %10, %11  : i64
    %344 = llvm.mlir.zero : !llvm.ptr
    %345 = llvm.getelementptr %344[%343] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %346 = llvm.ptrtoint %345 : !llvm.ptr to i64
    %347 = llvm.mlir.constant(64 : index) : i64
    %348 = llvm.add %346, %347  : i64
    %349 = llvm.call @malloc(%348) : (i64) -> !llvm.ptr
    %350 = llvm.ptrtoint %349 : !llvm.ptr to i64
    %351 = llvm.mlir.constant(1 : index) : i64
    %352 = llvm.sub %347, %351  : i64
    %353 = llvm.add %350, %352  : i64
    %354 = llvm.urem %353, %347  : i64
    %355 = llvm.sub %353, %354  : i64
    %356 = llvm.inttoptr %355 : i64 to !llvm.ptr
    %357 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %358 = llvm.insertvalue %349, %357[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %359 = llvm.insertvalue %356, %358[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %360 = llvm.mlir.constant(0 : index) : i64
    %361 = llvm.insertvalue %360, %359[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %362 = llvm.insertvalue %11, %361[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %363 = llvm.insertvalue %10, %362[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %364 = llvm.insertvalue %10, %363[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %365 = llvm.insertvalue %342, %364[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %366 = llvm.intr.stacksave : !llvm.ptr
    %367 = llvm.mlir.constant(2 : i64) : i64
    %368 = llvm.mlir.constant(1 : index) : i64
    %369 = llvm.alloca %368 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %341, %369 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %370 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %371 = llvm.insertvalue %367, %370[0] : !llvm.struct<(i64, ptr)> 
    %372 = llvm.insertvalue %369, %371[1] : !llvm.struct<(i64, ptr)> 
    %373 = llvm.mlir.constant(2 : i64) : i64
    %374 = llvm.mlir.constant(1 : index) : i64
    %375 = llvm.alloca %374 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %365, %375 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %376 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %377 = llvm.insertvalue %373, %376[0] : !llvm.struct<(i64, ptr)> 
    %378 = llvm.insertvalue %375, %377[1] : !llvm.struct<(i64, ptr)> 
    %379 = llvm.mlir.constant(1 : index) : i64
    %380 = llvm.alloca %379 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %372, %380 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %381 = llvm.alloca %379 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %378, %381 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %382 = llvm.mlir.zero : !llvm.ptr
    %383 = llvm.getelementptr %382[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %384 = llvm.ptrtoint %383 : !llvm.ptr to i64
    llvm.call @memrefCopy(%384, %380, %381) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %366 : !llvm.ptr
    %385 = llvm.extractvalue %64[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %386 = llvm.extractvalue %64[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %387 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %388 = llvm.insertvalue %385, %387[0] : !llvm.struct<(ptr, ptr, i64)> 
    %389 = llvm.insertvalue %386, %388[1] : !llvm.struct<(ptr, ptr, i64)> 
    %390 = llvm.mlir.constant(0 : index) : i64
    %391 = llvm.insertvalue %390, %389[2] : !llvm.struct<(ptr, ptr, i64)> 
    %392 = llvm.extractvalue %64[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %393 = llvm.extractvalue %64[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %394 = llvm.extractvalue %64[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %395 = llvm.extractvalue %64[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %396 = llvm.extractvalue %64[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %397 = llvm.mul %59, %395  : i64
    %398 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %399 = llvm.insertvalue %385, %398[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %400 = llvm.insertvalue %386, %399[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %401 = llvm.insertvalue %397, %400[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %402 = llvm.insertvalue %10, %401[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %403 = llvm.mlir.constant(1 : index) : i64
    %404 = llvm.insertvalue %403, %402[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb9(%8 : i64)
  ^bb9(%405: i64):  // 2 preds: ^bb8, ^bb13
    %406 = llvm.icmp "slt" %405, %10 : i64
    llvm.cond_br %406, ^bb10, ^bb14
  ^bb10:  // pred: ^bb9
    llvm.br ^bb11(%8 : i64)
  ^bb11(%407: i64):  // 2 preds: ^bb10, ^bb12
    %408 = llvm.icmp "slt" %407, %11 : i64
    llvm.cond_br %408, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    %409 = llvm.getelementptr %293[%407] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %410 = llvm.load %409 : !llvm.ptr -> f64
    %411 = llvm.mul %407, %10  : i64
    %412 = llvm.add %411, %405  : i64
    %413 = llvm.getelementptr %356[%412] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %414 = llvm.load %413 : !llvm.ptr -> f64
    %415 = llvm.getelementptr %386[%397] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %416 = llvm.getelementptr %415[%405] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %417 = llvm.load %416 : !llvm.ptr -> f64
    %418 = llvm.fmul %410, %414  : f64
    %419 = llvm.fsub %417, %418  : f64
    %420 = llvm.icmp "slt" %407, %59 : i64
    %421 = llvm.select %420, %419, %417 : i1, f64
    %422 = llvm.getelementptr %386[%397] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %423 = llvm.getelementptr %422[%405] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %421, %423 : f64, !llvm.ptr
    %424 = llvm.add %407, %9  : i64
    llvm.br ^bb11(%424 : i64)
  ^bb13:  // pred: ^bb11
    %425 = llvm.add %405, %9  : i64
    llvm.br ^bb9(%425 : i64)
  ^bb14:  // pred: ^bb9
    %426 = llvm.extractvalue %64[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %427 = llvm.extractvalue %64[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %428 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %429 = llvm.insertvalue %426, %428[0] : !llvm.struct<(ptr, ptr, i64)> 
    %430 = llvm.insertvalue %427, %429[1] : !llvm.struct<(ptr, ptr, i64)> 
    %431 = llvm.mlir.constant(0 : index) : i64
    %432 = llvm.insertvalue %431, %430[2] : !llvm.struct<(ptr, ptr, i64)> 
    %433 = llvm.extractvalue %64[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %434 = llvm.extractvalue %64[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %435 = llvm.extractvalue %64[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %436 = llvm.extractvalue %64[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %437 = llvm.extractvalue %64[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %438 = llvm.mul %59, %436  : i64
    %439 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %440 = llvm.insertvalue %426, %439[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %441 = llvm.insertvalue %427, %440[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %442 = llvm.insertvalue %438, %441[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %443 = llvm.insertvalue %10, %442[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %444 = llvm.mlir.constant(1 : index) : i64
    %445 = llvm.insertvalue %444, %443[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %446 = llvm.intr.stacksave : !llvm.ptr
    %447 = llvm.mlir.constant(1 : i64) : i64
    %448 = llvm.mlir.constant(1 : index) : i64
    %449 = llvm.alloca %448 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %404, %449 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %450 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %451 = llvm.insertvalue %447, %450[0] : !llvm.struct<(i64, ptr)> 
    %452 = llvm.insertvalue %449, %451[1] : !llvm.struct<(i64, ptr)> 
    %453 = llvm.mlir.constant(1 : i64) : i64
    %454 = llvm.mlir.constant(1 : index) : i64
    %455 = llvm.alloca %454 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %445, %455 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %456 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %457 = llvm.insertvalue %453, %456[0] : !llvm.struct<(i64, ptr)> 
    %458 = llvm.insertvalue %455, %457[1] : !llvm.struct<(i64, ptr)> 
    %459 = llvm.mlir.constant(1 : index) : i64
    %460 = llvm.alloca %459 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %452, %460 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %461 = llvm.alloca %459 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %458, %461 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %462 = llvm.mlir.zero : !llvm.ptr
    %463 = llvm.getelementptr %462[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %464 = llvm.ptrtoint %463 : !llvm.ptr to i64
    llvm.call @memrefCopy(%464, %460, %461) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %446 : !llvm.ptr
    %465 = llvm.add %59, %9  : i64
    llvm.br ^bb1(%465, %64 : i64, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb15:  // pred: ^bb1
    %466 = llvm.mlir.constant(1 : index) : i64
    %467 = llvm.extractvalue %60[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %468 = llvm.mul %467, %466  : i64
    %469 = llvm.extractvalue %60[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %470 = llvm.mul %468, %469  : i64
    %471 = llvm.mlir.zero : !llvm.ptr
    %472 = llvm.getelementptr %471[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %473 = llvm.ptrtoint %472 : !llvm.ptr to i64
    %474 = llvm.mul %470, %473  : i64
    %475 = llvm.extractvalue %60[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %476 = llvm.extractvalue %60[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %477 = llvm.getelementptr %475[%476] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %478 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %479 = llvm.extractvalue %7[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %480 = llvm.getelementptr %478[%479] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%480, %477, %474) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.return
  }
}

