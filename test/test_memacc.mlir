module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.func @_Z6gatherPdS_Piii(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32, %arg4: i32) {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = builtin.unrealized_conversion_cast %arg2 : !llvm.ptr to memref<?xi32>
    %3 = builtin.unrealized_conversion_cast %arg1 : !llvm.ptr to memref<?xf64>
    %4 = builtin.unrealized_conversion_cast %arg0 : !llvm.ptr to memref<?xf64>
    %5 = llvm.sext %arg3 : i32 to i64
    %6 = llvm.sitofp %arg4 : i32 to f64
    llvm.br ^bb1(%0 : i64)
  ^bb1(%7: i64):  // 2 preds: ^bb0, ^bb2
    %8 = builtin.unrealized_conversion_cast %7 : i64 to index
    %9 = llvm.icmp "slt" %7, %5 : i64
    llvm.cond_br %9, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %10 = memacc.generic_load region {
      %15 = memacc.load %2[%8] : memref<?xi32>
      %16 = memacc.muli %15, %arg4 : i32
      %17 = memacc.index_cast %16 : i32 to index
      %18 = memacc.load %3[%17] : memref<?xf64>
      %19 = memacc.yield %18 : (f64) -> f64
    } indirection_level = 1 : () -> f64
    %11 = llvm.fmul %6, %10  : f64
    %12 = memacc.generic_load region {
      %15 = memacc.load %4[%8] : memref<?xf64>
      %16 = memacc.yield %15 : (f64) -> f64
    } indirection_level = 0 : () -> f64
    %13 = llvm.fadd %12, %11  : f64
    memacc.generic_store region {
      memacc.store %13, %4[%8] : memref<?xf64>
      memacc.yield  : () -> ()
    } : () -> ()
    %14 = llvm.add %7, %1  : i64
    llvm.br ^bb1(%14 : i64)
  ^bb3:  // pred: ^bb1
    llvm.return
  }
  llvm.func @_Z7scatterPdS_PiS0_ii(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: i32, %arg5: i32) {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = builtin.unrealized_conversion_cast %arg3 : !llvm.ptr to memref<?xi32>
    %3 = builtin.unrealized_conversion_cast %arg2 : !llvm.ptr to memref<?xi32>
    %4 = builtin.unrealized_conversion_cast %arg1 : !llvm.ptr to memref<?xf64>
    %5 = builtin.unrealized_conversion_cast %arg0 : !llvm.ptr to memref<?xf64>
    %6 = llvm.sext %arg4 : i32 to i64
    %7 = llvm.sitofp %arg5 : i32 to f64
    llvm.br ^bb1(%0 : i64)
  ^bb1(%8: i64):  // 2 preds: ^bb0, ^bb2
    %9 = builtin.unrealized_conversion_cast %8 : i64 to index
    %10 = llvm.icmp "slt" %8, %6 : i64
    llvm.cond_br %10, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %11 = memacc.generic_load region {
      %16 = memacc.load %4[%9] : memref<?xf64>
      %17 = memacc.yield %16 : (f64) -> f64
    } indirection_level = 0 : () -> f64
    %12 = llvm.fmul %7, %11  : f64
    %13:2 = memacc.generic_load region {
      %16 = memacc.load %2[%9] : memref<?xi32>
      %17 = memacc.index_cast %16 : i32 to index
      %18 = memacc.load %3[%17] : memref<?xi32>
      %19 = memacc.index_cast %18 : i32 to index
      %20 = memacc.load %5[%19] : memref<?xf64>
      %21:2 = memacc.yield %19, %20 : (index, f64) -> (index, f64)
    } indirection_level = 2 : () -> (index, f64)
    %14 = llvm.fadd %13#1, %12  : f64
    memacc.generic_store region {
      memacc.store %14, %5[%13#0] : memref<?xf64>
      memacc.yield  : () -> ()
    } : () -> ()
    %15 = llvm.add %8, %1  : i64
    llvm.br ^bb1(%15 : i64)
  ^bb3:  // pred: ^bb1
    llvm.return
  }
}

