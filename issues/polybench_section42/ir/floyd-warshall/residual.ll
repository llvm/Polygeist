; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @malloc(i64)

declare void @free(ptr)

declare void @memrefCopy(i64, ptr, ptr)

define void @kernel_floyd_warshall_impl(i32 %0, ptr %1, ptr %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7) {
  %9 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %1, 0
  %10 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %9, ptr %2, 1
  %11 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %10, i64 %3, 2
  %12 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %11, i64 %4, 3, 0
  %13 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, i64 %6, 4, 0
  %14 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, i64 %5, 3, 1
  %15 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %14, i64 %7, 4, 1
  %16 = sext i32 %0 to i64
  %17 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 0
  %18 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 1
  %19 = insertvalue { ptr, ptr, i64 } undef, ptr %17, 0
  %20 = insertvalue { ptr, ptr, i64 } %19, ptr %18, 1
  %21 = insertvalue { ptr, ptr, i64 } %20, i64 0, 2
  %22 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 2
  %23 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 3, 0
  %24 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 3, 1
  %25 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 4, 0
  %26 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 4, 1
  %27 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %17, 0
  %28 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, ptr %18, 1
  %29 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, i64 0, 2
  %30 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, i64 %16, 3, 0
  %31 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, i64 %25, 4, 0
  %32 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, i64 %16, 3, 1
  %33 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, i64 1, 4, 1
  %34 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 0
  %35 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 1
  %36 = insertvalue { ptr, ptr, i64 } undef, ptr %34, 0
  %37 = insertvalue { ptr, ptr, i64 } %36, ptr %35, 1
  %38 = insertvalue { ptr, ptr, i64 } %37, i64 0, 2
  %39 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 2
  %40 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 3, 0
  %41 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 3, 1
  %42 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 4, 0
  %43 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 4, 1
  %44 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %34, 0
  %45 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, ptr %35, 1
  %46 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %45, i64 0, 2
  %47 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, i64 %16, 3, 0
  %48 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %47, i64 %42, 4, 0
  %49 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %48, i64 %16, 3, 1
  %50 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %49, i64 1, 4, 1
  %51 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 0
  %52 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 1
  %53 = insertvalue { ptr, ptr, i64 } undef, ptr %51, 0
  %54 = insertvalue { ptr, ptr, i64 } %53, ptr %52, 1
  %55 = insertvalue { ptr, ptr, i64 } %54, i64 0, 2
  %56 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 2
  %57 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 3, 0
  %58 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 3, 1
  %59 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 4, 0
  %60 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 4, 1
  %61 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %51, 0
  %62 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %61, ptr %52, 1
  %63 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %62, i64 0, 2
  %64 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %63, i64 %16, 3, 0
  %65 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %64, i64 %59, 4, 0
  %66 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %65, i64 %16, 3, 1
  %67 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %66, i64 1, 4, 1
  %68 = mul i64 %16, %16
  %69 = getelementptr double, ptr null, i64 %68
  %70 = ptrtoint ptr %69 to i64
  %71 = add i64 %70, 64
  %72 = call ptr @malloc(i64 %71)
  %73 = ptrtoint ptr %72 to i64
  %74 = add i64 %73, 63
  %75 = urem i64 %74, 64
  %76 = sub i64 %74, %75
  %77 = inttoptr i64 %76 to ptr
  %78 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %72, 0
  %79 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %78, ptr %77, 1
  %80 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %79, i64 0, 2
  %81 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %80, i64 %16, 3, 0
  %82 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %81, i64 %16, 3, 1
  %83 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %82, i64 %16, 4, 0
  %84 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %83, i64 1, 4, 1
  %85 = call ptr @llvm.stacksave.p0()
  %86 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %67, ptr %86, align 8
  %87 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %86, 1
  %88 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %84, ptr %88, align 8
  %89 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %88, 1
  %90 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %87, ptr %90, align 8
  %91 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %89, ptr %91, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), ptr %90, ptr %91)
  call void @llvm.stackrestore.p0(ptr %85)
  br label %92

92:                                               ; preds = %127, %8
  %93 = phi i64 [ %128, %127 ], [ 0, %8 ]
  %94 = icmp slt i64 %93, %16
  br i1 %94, label %95, label %129

95:                                               ; preds = %92
  br label %96

96:                                               ; preds = %125, %95
  %97 = phi i64 [ %126, %125 ], [ 0, %95 ]
  %98 = icmp slt i64 %97, %16
  br i1 %98, label %99, label %127

99:                                               ; preds = %96
  br label %100

100:                                              ; preds = %103, %99
  %101 = phi i64 [ %124, %103 ], [ 0, %99 ]
  %102 = icmp slt i64 %101, %16
  br i1 %102, label %103, label %125

103:                                              ; preds = %100
  %104 = getelementptr double, ptr %18, i64 0
  %105 = mul i64 %97, %25
  %106 = add i64 %105, %93
  %107 = getelementptr double, ptr %104, i64 %106
  %108 = load double, ptr %107, align 8
  %109 = getelementptr double, ptr %35, i64 0
  %110 = mul i64 %93, %42
  %111 = add i64 %110, %101
  %112 = getelementptr double, ptr %109, i64 %111
  %113 = load double, ptr %112, align 8
  %114 = mul i64 %97, %16
  %115 = add i64 %114, %101
  %116 = getelementptr double, ptr %77, i64 %115
  %117 = load double, ptr %116, align 8
  %118 = fadd double %108, %113
  %119 = fcmp olt double %117, %118
  %120 = select i1 %119, double %117, double %118
  %121 = mul i64 %97, %16
  %122 = add i64 %121, %101
  %123 = getelementptr double, ptr %77, i64 %122
  store double %120, ptr %123, align 8
  %124 = add i64 %101, 1
  br label %100

125:                                              ; preds = %100
  %126 = add i64 %97, 1
  br label %96

127:                                              ; preds = %96
  %128 = add i64 %93, 1
  br label %92

129:                                              ; preds = %92
  %130 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 3
  %131 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %130, ptr %131, align 8
  %132 = getelementptr [2 x i64], ptr %131, i32 0, i32 0
  %133 = load i64, ptr %132, align 8
  %134 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 3
  %135 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %134, ptr %135, align 8
  %136 = getelementptr [2 x i64], ptr %135, i32 0, i32 1
  %137 = load i64, ptr %136, align 8
  %138 = mul i64 %137, %133
  %139 = getelementptr double, ptr null, i64 %138
  %140 = ptrtoint ptr %139 to i64
  %141 = add i64 %140, 64
  %142 = call ptr @malloc(i64 %141)
  %143 = ptrtoint ptr %142 to i64
  %144 = add i64 %143, 63
  %145 = urem i64 %144, 64
  %146 = sub i64 %144, %145
  %147 = inttoptr i64 %146 to ptr
  %148 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %142, 0
  %149 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %148, ptr %147, 1
  %150 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %149, i64 0, 2
  %151 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %150, i64 %133, 3, 0
  %152 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %151, i64 %137, 3, 1
  %153 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %152, i64 %137, 4, 0
  %154 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %153, i64 1, 4, 1
  %155 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 3, 0
  %156 = mul i64 %155, 1
  %157 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 3, 1
  %158 = mul i64 %156, %157
  %159 = mul i64 %158, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %160 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 1
  %161 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 2
  %162 = getelementptr double, ptr %160, i64 %161
  %163 = getelementptr double, ptr %147, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %163, ptr %162, i64 %159, i1 false)
  %164 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %142, 0
  %165 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %164, ptr %147, 1
  %166 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %165, i64 0, 2
  %167 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %166, i64 %16, 3, 0
  %168 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %167, i64 %137, 4, 0
  %169 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %168, i64 %16, 3, 1
  %170 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %169, i64 1, 4, 1
  %171 = call ptr @llvm.stacksave.p0()
  %172 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %84, ptr %172, align 8
  %173 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %172, 1
  %174 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %170, ptr %174, align 8
  %175 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %174, 1
  %176 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %173, ptr %176, align 8
  %177 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %175, ptr %177, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), ptr %176, ptr %177)
  call void @llvm.stackrestore.p0(ptr %171)
  %178 = mul i64 %133, 1
  %179 = mul i64 %178, %137
  %180 = mul i64 %179, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %181 = getelementptr double, ptr %147, i64 0
  %182 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 1
  %183 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 2
  %184 = getelementptr double, ptr %182, i64 %183
  call void @llvm.memcpy.p0.p0.i64(ptr %184, ptr %181, i64 %180, i1 false)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare ptr @llvm.stacksave.p0() #0

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.stackrestore.p0(ptr) #0

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #1

attributes #0 = { nocallback nofree nosync nounwind willreturn }
attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
