module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_2mm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: f64, %arg5: f64, %arg6: memref<?x?xf64>, %arg7: memref<?x?xf64>, %arg8: memref<?x?xf64>, %arg9: memref<?x?xf64>, %arg10: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f64
    %0 = arith.index_cast %arg2 : i32 to index
    %1 = arith.index_cast %arg3 : i32 to index
    %2 = arith.index_cast %arg1 : i32 to index
    %3 = arith.index_cast %arg0 : i32 to index
    %dim = memref.dim %arg6, %c0 : memref<?x?xf64>
    %dim_0 = memref.dim %arg6, %c1 : memref<?x?xf64>
    %alloc = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf64>
    cf.br ^bb1(%c0 : index)
  ^bb1(%4: index):  // 2 preds: ^bb0, ^bb4
    %5 = arith.cmpi slt, %4, %dim : index
    cf.cond_br %5, ^bb2(%c0 : index), ^bb5
  ^bb2(%6: index):  // 2 preds: ^bb1, ^bb3
    %7 = arith.cmpi slt, %6, %dim_0 : index
    cf.cond_br %7, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    memref.store %cst, %alloc[%4, %6] : memref<?x?xf64>
    %8 = arith.addi %6, %c1 : index
    cf.br ^bb2(%8 : index)
  ^bb4:  // pred: ^bb2
    %9 = arith.addi %4, %c1 : index
    cf.br ^bb1(%9 : index)
  ^bb5:  // pred: ^bb1
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg7 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [%3, %0], strides: [%strides#0, 1] : memref<f64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %base_buffer_1, %offset_2, %sizes_3:2, %strides_4:2 = memref.extract_strided_metadata %arg8 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %reinterpret_cast_5 = memref.reinterpret_cast %base_buffer_1 to offset: [0], sizes: [%0, %2], strides: [%strides_4#0, 1] : memref<f64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %reinterpret_cast_6 = memref.reinterpret_cast %alloc to offset: [0], sizes: [%3, %2], strides: [%dim_0, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %alloc_7 = memref.alloc(%3, %2) {alignment = 64 : i64} : memref<?x?xf64>
    memref.copy %reinterpret_cast_6, %alloc_7 : memref<?x?xf64, strided<[?, 1], offset: ?>> to memref<?x?xf64>
    cf.br ^bb6(%c0 : index)
  ^bb6(%10: index):  // 2 preds: ^bb5, ^bb11
    %11 = arith.cmpi slt, %10, %3 : index
    cf.cond_br %11, ^bb7(%c0 : index), ^bb12
  ^bb7(%12: index):  // 2 preds: ^bb6, ^bb10
    %13 = arith.cmpi slt, %12, %2 : index
    cf.cond_br %13, ^bb8(%c0 : index), ^bb11
  ^bb8(%14: index):  // 2 preds: ^bb7, ^bb9
    %15 = arith.cmpi slt, %14, %0 : index
    cf.cond_br %15, ^bb9, ^bb10
  ^bb9:  // pred: ^bb8
    %16 = memref.load %reinterpret_cast[%10, %14] : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %17 = memref.load %reinterpret_cast_5[%14, %12] : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %18 = memref.load %alloc_7[%10, %12] : memref<?x?xf64>
    %19 = arith.mulf %arg4, %16 : f64
    %20 = arith.mulf %19, %17 : f64
    %21 = arith.addf %18, %20 : f64
    memref.store %21, %alloc_7[%10, %12] : memref<?x?xf64>
    %22 = arith.addi %14, %c1 : index
    cf.br ^bb8(%22 : index)
  ^bb10:  // pred: ^bb8
    %23 = arith.addi %12, %c1 : index
    cf.br ^bb7(%23 : index)
  ^bb11:  // pred: ^bb7
    %24 = arith.addi %10, %c1 : index
    cf.br ^bb6(%24 : index)
  ^bb12:  // pred: ^bb6
    %reinterpret_cast_8 = memref.reinterpret_cast %alloc to offset: [0], sizes: [%3, %2], strides: [%dim_0, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    memref.copy %alloc_7, %reinterpret_cast_8 : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    memref.copy %alloc, %arg6 : memref<?x?xf64> to memref<?x?xf64>
    %dim_9 = memref.dim %arg10, %c0 : memref<?x?xf64>
    %dim_10 = memref.dim %arg10, %c1 : memref<?x?xf64>
    %alloc_11 = memref.alloc(%dim_9, %dim_10) {alignment = 64 : i64} : memref<?x?xf64>
    memref.copy %arg10, %alloc_11 : memref<?x?xf64> to memref<?x?xf64>
    cf.br ^bb13(%c0 : index)
  ^bb13(%25: index):  // 2 preds: ^bb12, ^bb16
    %26 = arith.cmpi slt, %25, %dim_9 : index
    cf.cond_br %26, ^bb14(%c0 : index), ^bb17
  ^bb14(%27: index):  // 2 preds: ^bb13, ^bb15
    %28 = arith.cmpi slt, %27, %dim_10 : index
    cf.cond_br %28, ^bb15, ^bb16
  ^bb15:  // pred: ^bb14
    %29 = memref.load %alloc_11[%25, %27] : memref<?x?xf64>
    %30 = arith.mulf %29, %arg5 : f64
    memref.store %30, %alloc_11[%25, %27] : memref<?x?xf64>
    %31 = arith.addi %27, %c1 : index
    cf.br ^bb14(%31 : index)
  ^bb16:  // pred: ^bb14
    %32 = arith.addi %25, %c1 : index
    cf.br ^bb13(%32 : index)
  ^bb17:  // pred: ^bb13
    %base_buffer_12, %offset_13, %sizes_14:2, %strides_15:2 = memref.extract_strided_metadata %arg9 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %reinterpret_cast_16 = memref.reinterpret_cast %base_buffer_12 to offset: [0], sizes: [%2, %1], strides: [%strides_15#0, 1] : memref<f64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    %reinterpret_cast_17 = memref.reinterpret_cast %alloc_11 to offset: [0], sizes: [%3, %1], strides: [%dim_10, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    cf.br ^bb18(%c0 : index)
  ^bb18(%33: index):  // 2 preds: ^bb17, ^bb23
    %34 = arith.cmpi slt, %33, %3 : index
    cf.cond_br %34, ^bb19(%c0 : index), ^bb24
  ^bb19(%35: index):  // 2 preds: ^bb18, ^bb22
    %36 = arith.cmpi slt, %35, %1 : index
    cf.cond_br %36, ^bb20(%c0 : index), ^bb23
  ^bb20(%37: index):  // 2 preds: ^bb19, ^bb21
    %38 = arith.cmpi slt, %37, %2 : index
    cf.cond_br %38, ^bb21, ^bb22
  ^bb21:  // pred: ^bb20
    %39 = memref.load %alloc_7[%33, %37] : memref<?x?xf64>
    %40 = memref.load %reinterpret_cast_16[%37, %35] : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %41 = memref.load %reinterpret_cast_17[%33, %35] : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %42 = arith.mulf %39, %40 : f64
    %43 = arith.addf %41, %42 : f64
    memref.store %43, %reinterpret_cast_17[%33, %35] : memref<?x?xf64, strided<[?, 1], offset: ?>>
    %44 = arith.addi %37, %c1 : index
    cf.br ^bb20(%44 : index)
  ^bb22:  // pred: ^bb20
    %45 = arith.addi %35, %c1 : index
    cf.br ^bb19(%45 : index)
  ^bb23:  // pred: ^bb19
    %46 = arith.addi %33, %c1 : index
    cf.br ^bb18(%46 : index)
  ^bb24:  // pred: ^bb18
    %reinterpret_cast_18 = memref.reinterpret_cast %alloc_11 to offset: [0], sizes: [%3, %1], strides: [%dim_10, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    memref.copy %reinterpret_cast_17, %reinterpret_cast_18 : memref<?x?xf64, strided<[?, 1], offset: ?>> to memref<?x?xf64, strided<[?, 1], offset: ?>>
    memref.copy %alloc_11, %arg10 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
  llvm.mlir.global internal constant @str6("==END   DUMP_ARRAYS==\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str5("\0Aend   dump: %s\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str4("%0.2lf \00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str3("\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str2("D\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str1("begin dump: %s\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str0("==BEGIN DUMP_ARRAYS==\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global external @stderr() {addr_space = 0 : i32} : !llvm.ptr
  llvm.func @fprintf(!llvm.ptr, !llvm.ptr, ...) -> i32
  func.func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1200 = arith.constant 1200 : index
    %c900 = arith.constant 900 : index
    %c1100 = arith.constant 1100 : index
    %c1 = arith.constant 1 : index
    %c800 = arith.constant 800 : index
    %cst = arith.constant 1.100000e+03 : f64
    %cst_0 = arith.constant 1.200000e+03 : f64
    %cst_1 = arith.constant 9.000000e+02 : f64
    %cst_2 = arith.constant 8.000000e+02 : f64
    %c20 = arith.constant 20 : index
    %cst_3 = arith.constant 0.000000e+00 : f64
    %cst_4 = arith.constant 1.500000e+00 : f64
    %cst_5 = arith.constant 1.200000e+00 : f64
    %c1_i32 = arith.constant 1 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1200_i32 = arith.constant 1200 : i32
    %c1100_i32 = arith.constant 1100 : i32
    %c900_i32 = arith.constant 900 : i32
    %c800_i32 = arith.constant 800 : i32
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<800x900xf64>
    %alloc_6 = memref.alloc() : memref<800x1100xf64>
    %alloc_7 = memref.alloc() : memref<1100x900xf64>
    %alloc_8 = memref.alloc() : memref<900x1200xf64>
    %alloc_9 = memref.alloc() : memref<800x1200xf64>
    %0 = "polygeist.subindex"(%alloc_6, %c0) : (memref<800x1100xf64>, index) -> memref<?xf64>
    %1 = "polygeist.memref2pointer"(%0) : (memref<?xf64>) -> !llvm.ptr
    %2 = "polygeist.pointer2memref"(%1) : (!llvm.ptr) -> memref<?x?xf64>
    %3 = "polygeist.subindex"(%alloc_7, %c0) : (memref<1100x900xf64>, index) -> memref<?xf64>
    %4 = "polygeist.memref2pointer"(%3) : (memref<?xf64>) -> !llvm.ptr
    %5 = "polygeist.pointer2memref"(%4) : (!llvm.ptr) -> memref<?x?xf64>
    %6 = "polygeist.subindex"(%alloc_8, %c0) : (memref<900x1200xf64>, index) -> memref<?xf64>
    %7 = "polygeist.memref2pointer"(%6) : (memref<?xf64>) -> !llvm.ptr
    %8 = "polygeist.pointer2memref"(%7) : (!llvm.ptr) -> memref<?x?xf64>
    %9 = "polygeist.subindex"(%alloc_9, %c0) : (memref<800x1200xf64>, index) -> memref<?xf64>
    %10 = "polygeist.memref2pointer"(%9) : (memref<?xf64>) -> !llvm.ptr
    %11 = "polygeist.pointer2memref"(%10) : (!llvm.ptr) -> memref<?x?xf64>
    cf.br ^bb1(%c0 : index)
  ^bb1(%12: index):  // 2 preds: ^bb0, ^bb5
    %13 = arith.cmpi slt, %12, %c800 : index
    cf.cond_br %13, ^bb2, ^bb6(%c0 : index)
  ^bb2:  // pred: ^bb1
    %14 = arith.index_cast %12 : index to i32
    cf.br ^bb3(%c0 : index)
  ^bb3(%15: index):  // 2 preds: ^bb2, ^bb4
    %16 = arith.cmpi slt, %15, %c1100 : index
    cf.cond_br %16, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %17 = arith.index_cast %15 : index to i32
    %18 = arith.muli %14, %17 : i32
    %19 = arith.addi %18, %c1_i32 : i32
    %20 = arith.remsi %19, %c800_i32 : i32
    %21 = arith.sitofp %20 : i32 to f64
    %22 = arith.divf %21, %cst_2 : f64
    memref.store %22, %2[%12, %15] : memref<?x?xf64>
    %23 = arith.addi %15, %c1 : index
    cf.br ^bb3(%23 : index)
  ^bb5:  // pred: ^bb3
    %24 = arith.addi %12, %c1 : index
    cf.br ^bb1(%24 : index)
  ^bb6(%25: index):  // 2 preds: ^bb1, ^bb10
    %26 = arith.cmpi slt, %25, %c1100 : index
    cf.cond_br %26, ^bb7, ^bb11(%c0 : index)
  ^bb7:  // pred: ^bb6
    %27 = arith.index_cast %25 : index to i32
    cf.br ^bb8(%c0 : index)
  ^bb8(%28: index):  // 2 preds: ^bb7, ^bb9
    %29 = arith.cmpi slt, %28, %c900 : index
    cf.cond_br %29, ^bb9, ^bb10
  ^bb9:  // pred: ^bb8
    %30 = arith.index_cast %28 : index to i32
    %31 = arith.addi %30, %c1_i32 : i32
    %32 = arith.muli %27, %31 : i32
    %33 = arith.remsi %32, %c900_i32 : i32
    %34 = arith.sitofp %33 : i32 to f64
    %35 = arith.divf %34, %cst_1 : f64
    memref.store %35, %5[%25, %28] : memref<?x?xf64>
    %36 = arith.addi %28, %c1 : index
    cf.br ^bb8(%36 : index)
  ^bb10:  // pred: ^bb8
    %37 = arith.addi %25, %c1 : index
    cf.br ^bb6(%37 : index)
  ^bb11(%38: index):  // 2 preds: ^bb6, ^bb15
    %39 = arith.cmpi slt, %38, %c900 : index
    cf.cond_br %39, ^bb12, ^bb16(%c0 : index)
  ^bb12:  // pred: ^bb11
    %40 = arith.index_cast %38 : index to i32
    cf.br ^bb13(%c0 : index)
  ^bb13(%41: index):  // 2 preds: ^bb12, ^bb14
    %42 = arith.cmpi slt, %41, %c1200 : index
    cf.cond_br %42, ^bb14, ^bb15
  ^bb14:  // pred: ^bb13
    %43 = arith.index_cast %41 : index to i32
    %44 = arith.addi %43, %c3_i32 : i32
    %45 = arith.muli %40, %44 : i32
    %46 = arith.addi %45, %c1_i32 : i32
    %47 = arith.remsi %46, %c1200_i32 : i32
    %48 = arith.sitofp %47 : i32 to f64
    %49 = arith.divf %48, %cst_0 : f64
    memref.store %49, %8[%38, %41] : memref<?x?xf64>
    %50 = arith.addi %41, %c1 : index
    cf.br ^bb13(%50 : index)
  ^bb15:  // pred: ^bb13
    %51 = arith.addi %38, %c1 : index
    cf.br ^bb11(%51 : index)
  ^bb16(%52: index):  // 2 preds: ^bb11, ^bb20
    %53 = arith.cmpi slt, %52, %c800 : index
    cf.cond_br %53, ^bb17, ^bb21
  ^bb17:  // pred: ^bb16
    %54 = arith.index_cast %52 : index to i32
    cf.br ^bb18(%c0 : index)
  ^bb18(%55: index):  // 2 preds: ^bb17, ^bb19
    %56 = arith.cmpi slt, %55, %c1200 : index
    cf.cond_br %56, ^bb19, ^bb20
  ^bb19:  // pred: ^bb18
    %57 = arith.index_cast %55 : index to i32
    %58 = arith.addi %57, %c2_i32 : i32
    %59 = arith.muli %54, %58 : i32
    %60 = arith.remsi %59, %c1100_i32 : i32
    %61 = arith.sitofp %60 : i32 to f64
    %62 = arith.divf %61, %cst : f64
    memref.store %62, %11[%52, %55] : memref<?x?xf64>
    %63 = arith.addi %55, %c1 : index
    cf.br ^bb18(%63 : index)
  ^bb20:  // pred: ^bb18
    %64 = arith.addi %52, %c1 : index
    cf.br ^bb16(%64 : index)
  ^bb21:  // pred: ^bb16
    %65 = "polygeist.subindex"(%alloc, %c0) : (memref<800x900xf64>, index) -> memref<?xf64>
    %66 = "polygeist.memref2pointer"(%65) : (memref<?xf64>) -> !llvm.ptr
    %67 = "polygeist.pointer2memref"(%66) : (!llvm.ptr) -> memref<?x?xf64>
    cf.br ^bb22(%c0 : index)
  ^bb22(%68: index):  // 2 preds: ^bb21, ^bb28
    %69 = arith.cmpi slt, %68, %c800 : index
    cf.cond_br %69, ^bb23(%c0 : index), ^bb29(%c0 : index)
  ^bb23(%70: index):  // 2 preds: ^bb22, ^bb27
    %71 = arith.cmpi slt, %70, %c900 : index
    cf.cond_br %71, ^bb24, ^bb28
  ^bb24:  // pred: ^bb23
    memref.store %cst_3, %67[%68, %70] : memref<?x?xf64>
    cf.br ^bb25(%c0 : index)
  ^bb25(%72: index):  // 2 preds: ^bb24, ^bb26
    %73 = arith.cmpi slt, %72, %c1100 : index
    cf.cond_br %73, ^bb26, ^bb27
  ^bb26:  // pred: ^bb25
    %74 = memref.load %2[%68, %72] : memref<?x?xf64>
    %75 = arith.mulf %74, %cst_4 : f64
    %76 = memref.load %5[%72, %70] : memref<?x?xf64>
    %77 = arith.mulf %75, %76 : f64
    %78 = memref.load %67[%68, %70] : memref<?x?xf64>
    %79 = arith.addf %78, %77 : f64
    memref.store %79, %67[%68, %70] : memref<?x?xf64>
    %80 = arith.addi %72, %c1 : index
    cf.br ^bb25(%80 : index)
  ^bb27:  // pred: ^bb25
    %81 = arith.addi %70, %c1 : index
    cf.br ^bb23(%81 : index)
  ^bb28:  // pred: ^bb23
    %82 = arith.addi %68, %c1 : index
    cf.br ^bb22(%82 : index)
  ^bb29(%83: index):  // 2 preds: ^bb22, ^bb35
    %84 = arith.cmpi slt, %83, %c800 : index
    cf.cond_br %84, ^bb30(%c0 : index), ^bb36
  ^bb30(%85: index):  // 2 preds: ^bb29, ^bb34
    %86 = arith.cmpi slt, %85, %c1200 : index
    cf.cond_br %86, ^bb31, ^bb35
  ^bb31:  // pred: ^bb30
    %87 = memref.load %11[%83, %85] : memref<?x?xf64>
    %88 = arith.mulf %87, %cst_5 : f64
    memref.store %88, %11[%83, %85] : memref<?x?xf64>
    cf.br ^bb32(%c0 : index)
  ^bb32(%89: index):  // 2 preds: ^bb31, ^bb33
    %90 = arith.cmpi slt, %89, %c900 : index
    cf.cond_br %90, ^bb33, ^bb34
  ^bb33:  // pred: ^bb32
    %91 = memref.load %67[%83, %89] : memref<?x?xf64>
    %92 = memref.load %8[%89, %85] : memref<?x?xf64>
    %93 = arith.mulf %91, %92 : f64
    %94 = memref.load %11[%83, %85] : memref<?x?xf64>
    %95 = arith.addf %94, %93 : f64
    memref.store %95, %11[%83, %85] : memref<?x?xf64>
    %96 = arith.addi %89, %c1 : index
    cf.br ^bb32(%96 : index)
  ^bb34:  // pred: ^bb32
    %97 = arith.addi %85, %c1 : index
    cf.br ^bb30(%97 : index)
  ^bb35:  // pred: ^bb30
    %98 = arith.addi %83, %c1 : index
    cf.br ^bb29(%98 : index)
  ^bb36:  // pred: ^bb29
    %99 = llvm.mlir.addressof @stderr : !llvm.ptr
    %100 = llvm.load %99 : !llvm.ptr -> !llvm.ptr
    %101 = llvm.mlir.addressof @str0 : !llvm.ptr
    %102 = llvm.getelementptr %101[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<23 x i8>
    %103 = llvm.call @fprintf(%100, %102) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
    %104 = llvm.load %99 : !llvm.ptr -> !llvm.ptr
    %105 = llvm.mlir.addressof @str1 : !llvm.ptr
    %106 = llvm.getelementptr %105[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<15 x i8>
    %107 = llvm.mlir.addressof @str2 : !llvm.ptr
    %108 = llvm.getelementptr %107[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i8>
    %109 = llvm.call @fprintf(%104, %106, %108) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    %110 = llvm.mlir.addressof @str4 : !llvm.ptr
    %111 = llvm.getelementptr %110[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x i8>
    %112 = llvm.mlir.addressof @str3 : !llvm.ptr
    %113 = llvm.getelementptr %112[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i8>
    cf.br ^bb37(%c0 : index)
  ^bb37(%114: index):  // 2 preds: ^bb36, ^bb43
    %115 = arith.cmpi slt, %114, %c800 : index
    cf.cond_br %115, ^bb38, ^bb44
  ^bb38:  // pred: ^bb37
    %116 = arith.muli %114, %c800 : index
    cf.br ^bb39(%c0 : index)
  ^bb39(%117: index):  // 2 preds: ^bb38, ^bb42
    %118 = arith.cmpi slt, %117, %c1200 : index
    cf.cond_br %118, ^bb40, ^bb43
  ^bb40:  // pred: ^bb39
    %119 = arith.addi %117, %116 : index
    %120 = arith.remsi %119, %c20 : index
    %121 = arith.cmpi slt, %120, %c0 : index
    %122 = arith.addi %120, %c20 : index
    %123 = arith.select %121, %122, %120 : index
    %124 = arith.cmpi eq, %123, %c0 : index
    cf.cond_br %124, ^bb41, ^bb42
  ^bb41:  // pred: ^bb40
    %125 = llvm.load %99 : !llvm.ptr -> !llvm.ptr
    %126 = llvm.call @fprintf(%125, %113) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
    cf.br ^bb42
  ^bb42:  // 2 preds: ^bb40, ^bb41
    %127 = llvm.load %99 : !llvm.ptr -> !llvm.ptr
    %128 = memref.load %11[%114, %117] : memref<?x?xf64>
    %129 = llvm.call @fprintf(%127, %111, %128) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr, f64) -> i32
    %130 = arith.addi %117, %c1 : index
    cf.br ^bb39(%130 : index)
  ^bb43:  // pred: ^bb39
    %131 = arith.addi %114, %c1 : index
    cf.br ^bb37(%131 : index)
  ^bb44:  // pred: ^bb37
    %132 = llvm.load %99 : !llvm.ptr -> !llvm.ptr
    %133 = llvm.mlir.addressof @str5 : !llvm.ptr
    %134 = llvm.getelementptr %133[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<17 x i8>
    %135 = llvm.call @fprintf(%132, %134, %108) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    %136 = llvm.load %99 : !llvm.ptr -> !llvm.ptr
    %137 = llvm.mlir.addressof @str6 : !llvm.ptr
    %138 = llvm.getelementptr %137[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<23 x i8>
    %139 = llvm.call @fprintf(%136, %138) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
    memref.dealloc %alloc : memref<800x900xf64>
    memref.dealloc %alloc_6 : memref<800x1100xf64>
    memref.dealloc %alloc_7 : memref<1100x900xf64>
    memref.dealloc %alloc_8 : memref<900x1200xf64>
    memref.dealloc %alloc_9 : memref<800x1200xf64>
    return %c0_i32 : i32
  }
}

