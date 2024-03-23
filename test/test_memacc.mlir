module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.func @_Z6gatherPdS_Piii(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32, %arg4: i32) {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = llvm.mlir.constant(0 : index) : i64
    %2 = llvm.sext %arg3 : i32 to i64
    %3 = llvm.sitofp %arg4 : i32 to f64
    llvm.br ^bb1(%1 : i64)
  ^bb1(%4: i64):  // 2 preds: ^bb0, ^bb2
    %5 = llvm.icmp "slt" %4, %2 : i64
    llvm.cond_br %5, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %6 = memacc.generic_load region {
      %13 = llvm.getelementptr %arg2[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %14 = llvm.load %13 : !llvm.ptr -> i32
      %15 = llvm.mul %14, %arg4  : i32
      %16 = llvm.sext %15 : i32 to i64
      %17 = llvm.getelementptr %arg1[%16] : (!llvm.ptr, i64) -> !llvm.ptr, f64
      %18 = llvm.load %17 : !llvm.ptr -> f64
      %19 = memacc.yield %18 : (f64) -> f64
    } indirection_level = 1 : () -> f64
    %7 = llvm.fmul %3, %6  : f64
    %8 = llvm.getelementptr %arg0[%4] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %9 = llvm.load %8 : !llvm.ptr -> f64
    %10 = llvm.fadd %9, %7  : f64
    %11 = llvm.getelementptr %arg0[%4] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %10, %11 : f64, !llvm.ptr
    %12 = llvm.add %4, %0  : i64
    llvm.br ^bb1(%12 : i64)
  ^bb3:  // pred: ^bb1
    llvm.return
  }
  llvm.func @_Z7scatterPdS_PiS0_ii(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: i32, %arg5: i32) {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = llvm.mlir.constant(0 : index) : i64
    %2 = llvm.sext %arg4 : i32 to i64
    %3 = llvm.sitofp %arg5 : i32 to f64
    llvm.br ^bb1(%1 : i64)
  ^bb1(%4: i64):  // 2 preds: ^bb0, ^bb2
    %5 = llvm.icmp "slt" %4, %2 : i64
    llvm.cond_br %5, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %6 = llvm.getelementptr %arg1[%4] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %7 = llvm.load %6 : !llvm.ptr -> f64
    %8 = llvm.fmul %3, %7  : f64
    %9:2 = memacc.generic_load region {
      %13 = llvm.getelementptr %arg3[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %14 = llvm.load %13 : !llvm.ptr -> i32
      %15 = llvm.sext %14 : i32 to i64
      %16 = llvm.getelementptr %arg2[%15] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %17 = llvm.load %16 : !llvm.ptr -> i32
      %18 = llvm.sext %17 : i32 to i64
      %19 = builtin.unrealized_conversion_cast %18 : i64 to index
      %20 = llvm.getelementptr %arg0[%18] : (!llvm.ptr, i64) -> !llvm.ptr, f64
      %21 = llvm.load %20 : !llvm.ptr -> f64
      %22:2 = memacc.yield %19, %21 : (index, f64) -> (index, f64)
    } indirection_level = 2 : () -> (index, f64)
    %10 = builtin.unrealized_conversion_cast %9#0 : index to i64
    %11 = llvm.fadd %9#1, %8  : f64
    memacc.generic_store region {
      %13 = llvm.getelementptr %arg0[%10] : (!llvm.ptr, i64) -> !llvm.ptr, f64
      llvm.store %11, %13 : f64, !llvm.ptr
      memacc.yield  : () -> ()
    } : () -> ()
    %12 = llvm.add %4, %0  : i64
    llvm.br ^bb1(%12 : i64)
  ^bb3:  // pred: ^bb1
    llvm.return
  }
}

