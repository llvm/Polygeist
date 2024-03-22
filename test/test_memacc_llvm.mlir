module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.mlir.global private @"_Z14uselessinstr1sv@static@_ZZ14uselessinstr1svE1d"() {addr_space = 0 : i32} : !llvm.array<100 x f32> {
    %0 = llvm.mlir.undef : !llvm.array<100 x f32>
    llvm.return %0 : !llvm.array<100 x f32>
  }
  llvm.func @_Z14uselessinstr1sv() {
    %0 = llvm.mlir.constant(1.887000e+03 : f32) : f32
    %1 = llvm.mlir.addressof @"_Z14uselessinstr1sv@static@_ZZ14uselessinstr1svE1d" : !llvm.ptr
    %2 = llvm.getelementptr %1[5] : (!llvm.ptr) -> !llvm.ptr, f32
    llvm.store %0, %2 : f32, !llvm.ptr
    llvm.return
  }
  llvm.func @_Z6gatherPdS_Piii(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32, %arg4: i32) {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = llvm.mlir.constant(0 : index) : i64
    %2 = llvm.mlir.constant(1.887000e+03 : f32) : f32
    %3 = llvm.sext %arg4 : i32 to i64
    %4 = llvm.sext %arg3 : i32 to i64
    %5 = llvm.sitofp %arg4 : i32 to f64
    %6 = llvm.mlir.addressof @"_Z14uselessinstr1sv@static@_ZZ14uselessinstr1svE1d" : !llvm.ptr
    llvm.br ^bb1(%1 : i64)
  ^bb1(%7: i64):  // 2 preds: ^bb0, ^bb2
    %8 = llvm.icmp "slt" %7, %4 : i64
    llvm.cond_br %8, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %9 = llvm.getelementptr %6[5] : (!llvm.ptr) -> !llvm.ptr, f32
    llvm.store %2, %9 : f32, !llvm.ptr
    %10 = llvm.add %7, %3  : i64
    %11 = llvm.getelementptr %arg2[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %12 = llvm.load %11 : !llvm.ptr -> i32
    %13 = llvm.mul %12, %arg4  : i32
    %14 = llvm.sext %13 : i32 to i64
    %15 = llvm.getelementptr %arg1[%14] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %16 = llvm.load %15 : !llvm.ptr -> f64
    %17 = llvm.fmul %5, %16  : f64
    %18 = llvm.getelementptr %arg0[%7] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %19 = llvm.load %18 : !llvm.ptr -> f64
    %20 = llvm.fadd %19, %17  : f64
    %21 = llvm.getelementptr %arg0[%7] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %20, %21 : f64, !llvm.ptr
    %22 = llvm.getelementptr %6[5] : (!llvm.ptr) -> !llvm.ptr, f32
    llvm.store %2, %22 : f32, !llvm.ptr
    %23 = llvm.add %7, %0  : i64
    llvm.br ^bb1(%23 : i64)
  ^bb3:  // pred: ^bb1
    llvm.return
  }
  llvm.func @_Z7scatterPdS_Piii(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32, %arg4: i32) {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = llvm.mlir.constant(0 : index) : i64
    %2 = llvm.sext %arg3 : i32 to i64
    %3 = llvm.sitofp %arg4 : i32 to f64
    llvm.br ^bb1(%1 : i64)
  ^bb1(%4: i64):  // 2 preds: ^bb0, ^bb2
    %5 = llvm.icmp "slt" %4, %2 : i64
    llvm.cond_br %5, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %6 = llvm.getelementptr %arg2[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %7 = llvm.load %6 : !llvm.ptr -> i32
    %8 = llvm.sext %7 : i32 to i64
    %9 = llvm.getelementptr %arg1[%4] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %10 = llvm.load %9 : !llvm.ptr -> f64
    %11 = llvm.fmul %3, %10  : f64
    %12 = llvm.getelementptr %arg0[%8] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %13 = llvm.load %12 : !llvm.ptr -> f64
    %14 = llvm.fadd %13, %11  : f64
    memacc.generic_store region {
      %16 = llvm.getelementptr %arg0[%8] : (!llvm.ptr, i64) -> !llvm.ptr, f64
      llvm.store %14, %16 : f64, !llvm.ptr
      memacc.yield  : () -> ()
    } : () -> ()
    %15 = llvm.add %4, %0  : i64
    llvm.br ^bb1(%15 : i64)
  ^bb3:  // pred: ^bb1
    llvm.return
  }
  llvm.func @main() -> i32 {
    %0 = llvm.mlir.constant(2 : index) : i64
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.mlir.constant(10 : index) : i64
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.mlir.constant(2.000000e+00 : f64) : f64
    %5 = llvm.mlir.constant(1.887000e+03 : f32) : f32
    %6 = llvm.mlir.constant(2 : i32) : i32
    %7 = llvm.mlir.undef : i32
    %8 = llvm.alloca %2 x i32 : (i64) -> !llvm.ptr
    %9 = llvm.alloca %2 x f64 : (i64) -> !llvm.ptr
    %10 = llvm.alloca %2 x f64 : (i64) -> !llvm.ptr
    %11 = llvm.mlir.addressof @"_Z14uselessinstr1sv@static@_ZZ14uselessinstr1svE1d" : !llvm.ptr
    %12 = llvm.getelementptr %11[5] : (!llvm.ptr) -> !llvm.ptr, f32
    llvm.store %5, %12 : f32, !llvm.ptr
    %13 = llvm.getelementptr %11[5] : (!llvm.ptr) -> !llvm.ptr, f32
    llvm.store %5, %13 : f32, !llvm.ptr
    llvm.br ^bb1(%3 : i64)
  ^bb1(%14: i64):  // 2 preds: ^bb0, ^bb2
    %15 = llvm.icmp "slt" %14, %2 : i64
    llvm.cond_br %15, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %16 = llvm.add %14, %0  : i64
    %17 = llvm.getelementptr %8[%16] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %18 = llvm.load %17 : !llvm.ptr -> i32
    %19 = llvm.mul %18, %6  : i32
    %20 = llvm.sext %19 : i32 to i64
    %21 = llvm.getelementptr %9[%20] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %22 = llvm.load %21 : !llvm.ptr -> f64
    %23 = llvm.fmul %22, %4  : f64
    %24 = llvm.getelementptr %10[%14] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %25 = llvm.load %24 : !llvm.ptr -> f64
    %26 = llvm.fadd %25, %23  : f64
    %27 = llvm.getelementptr %10[%14] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %26, %27 : f64, !llvm.ptr
    %28 = llvm.add %14, %1  : i64
    llvm.br ^bb1(%28 : i64)
  ^bb3:  // pred: ^bb1
    llvm.br ^bb4(%3 : i64)
  ^bb4(%29: i64):  // 2 preds: ^bb3, ^bb5
    %30 = llvm.icmp "slt" %29, %2 : i64
    llvm.cond_br %30, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %31 = llvm.getelementptr %8[%29] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %32 = llvm.load %31 : !llvm.ptr -> i32
    %33 = llvm.sext %32 : i32 to i64
    %34 = llvm.getelementptr %9[%29] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %35 = llvm.load %34 : !llvm.ptr -> f64
    %36 = llvm.fmul %35, %4  : f64
    %37 = llvm.getelementptr %10[%33] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %38 = llvm.load %37 : !llvm.ptr -> f64
    %39 = llvm.fadd %38, %36  : f64
    memacc.generic_store region {
      %41 = llvm.getelementptr %10[%33] : (!llvm.ptr, i64) -> !llvm.ptr, f64
      llvm.store %39, %41 : f64, !llvm.ptr
      memacc.yield  : () -> ()
    } : () -> ()
    %40 = llvm.add %29, %1  : i64
    llvm.br ^bb4(%40 : i64)
  ^bb6:  // pred: ^bb4
    llvm.return %7 : i32
  }
}

