; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @malloc(i64)

declare void @free(ptr)

declare void @memrefCopy(i64, ptr, ptr)

define void @kernel_syrk_impl(i32 %0, i32 %1, double %2, double %3, ptr %4, ptr %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, ptr %11, ptr %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17) {
  %19 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %4, 0
  %20 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %19, ptr %5, 1
  %21 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %20, i64 %6, 2
  %22 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, i64 %7, 3, 0
  %23 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, i64 %9, 4, 0
  %24 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, i64 %8, 3, 1
  %25 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, i64 %10, 4, 1
  %26 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %11, 0
  %27 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %26, ptr %12, 1
  %28 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, i64 %13, 2
  %29 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, i64 %14, 3, 0
  %30 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, i64 %16, 4, 0
  %31 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, i64 %15, 3, 1
  %32 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, i64 %17, 4, 1
  %33 = sext i32 %1 to i64
  %34 = sext i32 %0 to i64
  %35 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, 3
  %36 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %35, ptr %36, align 8
  %37 = getelementptr [2 x i64], ptr %36, i32 0, i32 0
  %38 = load i64, ptr %37, align 8
  %39 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, 3
  %40 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %39, ptr %40, align 8
  %41 = getelementptr [2 x i64], ptr %40, i32 0, i32 1
  %42 = load i64, ptr %41, align 8
  %43 = mul i64 %42, %38
  %44 = getelementptr double, ptr null, i64 %43
  %45 = ptrtoint ptr %44 to i64
  %46 = add i64 %45, 64
  %47 = call ptr @malloc(i64 %46)
  %48 = ptrtoint ptr %47 to i64
  %49 = add i64 %48, 63
  %50 = urem i64 %49, 64
  %51 = sub i64 %49, %50
  %52 = inttoptr i64 %51 to ptr
  %53 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %47, 0
  %54 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %53, ptr %52, 1
  %55 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %54, i64 0, 2
  %56 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, i64 %38, 3, 0
  %57 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %56, i64 %42, 3, 1
  %58 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %57, i64 %42, 4, 0
  %59 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %58, i64 1, 4, 1
  %60 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, 3, 0
  %61 = mul i64 %60, 1
  %62 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, 3, 1
  %63 = mul i64 %61, %62
  %64 = mul i64 %63, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %65 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, 1
  %66 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, 2
  %67 = getelementptr double, ptr %65, i64 %66
  %68 = getelementptr double, ptr %52, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %68, ptr %67, i64 %64, i1 false)
  br label %69

69:                                               ; preds = %89, %18
  %70 = phi i64 [ %90, %89 ], [ 0, %18 ]
  %71 = icmp slt i64 %70, %38
  br i1 %71, label %72, label %91

72:                                               ; preds = %69
  br label %73

73:                                               ; preds = %76, %72
  %74 = phi i64 [ %88, %76 ], [ 0, %72 ]
  %75 = icmp slt i64 %74, %42
  br i1 %75, label %76, label %89

76:                                               ; preds = %73
  %77 = mul i64 %70, %42
  %78 = add i64 %77, %74
  %79 = getelementptr double, ptr %52, i64 %78
  %80 = load double, ptr %79, align 8
  %81 = fmul double %80, %3
  %82 = add i64 %70, 1
  %83 = icmp slt i64 %74, %82
  %84 = select i1 %83, double %81, double %80
  %85 = mul i64 %70, %42
  %86 = add i64 %85, %74
  %87 = getelementptr double, ptr %52, i64 %86
  store double %84, ptr %87, align 8
  %88 = add i64 %74, 1
  br label %73

89:                                               ; preds = %73
  %90 = add i64 %70, 1
  br label %69

91:                                               ; preds = %69
  %92 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, 0
  %93 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, 1
  %94 = insertvalue { ptr, ptr, i64 } undef, ptr %92, 0
  %95 = insertvalue { ptr, ptr, i64 } %94, ptr %93, 1
  %96 = insertvalue { ptr, ptr, i64 } %95, i64 0, 2
  %97 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, 2
  %98 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, 3, 0
  %99 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, 3, 1
  %100 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, 4, 0
  %101 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, 4, 1
  %102 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %92, 0
  %103 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %102, ptr %93, 1
  %104 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %103, i64 0, 2
  %105 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %104, i64 %34, 3, 0
  %106 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %105, i64 %100, 4, 0
  %107 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %106, i64 %33, 3, 1
  %108 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %107, i64 1, 4, 1
  %109 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, 0
  %110 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, 1
  %111 = insertvalue { ptr, ptr, i64 } undef, ptr %109, 0
  %112 = insertvalue { ptr, ptr, i64 } %111, ptr %110, 1
  %113 = insertvalue { ptr, ptr, i64 } %112, i64 0, 2
  %114 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, 2
  %115 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, 3, 0
  %116 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, 3, 1
  %117 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, 4, 0
  %118 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, 4, 1
  %119 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %109, 0
  %120 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %119, ptr %110, 1
  %121 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %120, i64 0, 2
  %122 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %121, i64 %34, 3, 0
  %123 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %122, i64 %117, 4, 0
  %124 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %123, i64 %33, 3, 1
  %125 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %124, i64 1, 4, 1
  %126 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %47, 0
  %127 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %126, ptr %52, 1
  %128 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %127, i64 0, 2
  %129 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %128, i64 %34, 3, 0
  %130 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %129, i64 %42, 4, 0
  %131 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %130, i64 %34, 3, 1
  %132 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %131, i64 1, 4, 1
  br label %133

133:                                              ; preds = %173, %91
  %134 = phi i64 [ %174, %173 ], [ 0, %91 ]
  %135 = icmp slt i64 %134, %34
  br i1 %135, label %136, label %175

136:                                              ; preds = %133
  br label %137

137:                                              ; preds = %171, %136
  %138 = phi i64 [ %172, %171 ], [ 0, %136 ]
  %139 = icmp slt i64 %138, %33
  br i1 %139, label %140, label %173

140:                                              ; preds = %137
  br label %141

141:                                              ; preds = %144, %140
  %142 = phi i64 [ %170, %144 ], [ 0, %140 ]
  %143 = icmp slt i64 %142, %34
  br i1 %143, label %144, label %171

144:                                              ; preds = %141
  %145 = getelementptr double, ptr %93, i64 0
  %146 = mul i64 %134, %100
  %147 = add i64 %146, %138
  %148 = getelementptr double, ptr %145, i64 %147
  %149 = load double, ptr %148, align 8
  %150 = getelementptr double, ptr %110, i64 0
  %151 = mul i64 %142, %117
  %152 = add i64 %151, %138
  %153 = getelementptr double, ptr %150, i64 %152
  %154 = load double, ptr %153, align 8
  %155 = getelementptr double, ptr %52, i64 0
  %156 = mul i64 %134, %42
  %157 = add i64 %156, %142
  %158 = getelementptr double, ptr %155, i64 %157
  %159 = load double, ptr %158, align 8
  %160 = fmul double %2, %149
  %161 = fmul double %160, %154
  %162 = fadd double %159, %161
  %163 = add i64 %134, 1
  %164 = icmp slt i64 %142, %163
  %165 = select i1 %164, double %162, double %159
  %166 = getelementptr double, ptr %52, i64 0
  %167 = mul i64 %134, %42
  %168 = add i64 %167, %142
  %169 = getelementptr double, ptr %166, i64 %168
  store double %165, ptr %169, align 8
  %170 = add i64 %142, 1
  br label %141

171:                                              ; preds = %141
  %172 = add i64 %138, 1
  br label %137

173:                                              ; preds = %137
  %174 = add i64 %134, 1
  br label %133

175:                                              ; preds = %133
  %176 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %47, 0
  %177 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %176, ptr %52, 1
  %178 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %177, i64 0, 2
  %179 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %178, i64 %34, 3, 0
  %180 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %179, i64 %42, 4, 0
  %181 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %180, i64 %34, 3, 1
  %182 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %181, i64 1, 4, 1
  %183 = call ptr @llvm.stacksave.p0()
  %184 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %132, ptr %184, align 8
  %185 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %184, 1
  %186 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %182, ptr %186, align 8
  %187 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %186, 1
  %188 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %185, ptr %188, align 8
  %189 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %187, ptr %189, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), ptr %188, ptr %189)
  call void @llvm.stackrestore.p0(ptr %183)
  %190 = mul i64 %38, 1
  %191 = mul i64 %190, %42
  %192 = mul i64 %191, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %193 = getelementptr double, ptr %52, i64 0
  %194 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, 1
  %195 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, 2
  %196 = getelementptr double, ptr %194, i64 %195
  call void @llvm.memcpy.p0.p0.i64(ptr %196, ptr %193, i64 %192, i1 false)
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare ptr @llvm.stacksave.p0() #1

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.stackrestore.p0(ptr) #1

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #1 = { nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
