; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @malloc(i64)

declare void @free(ptr)

declare void @memrefCopy(i64, ptr, ptr)

define void @kernel_gemm_impl(i32 %0, i32 %1, i32 %2, double %3, double %4, ptr %5, ptr %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11, ptr %12, ptr %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, ptr %19, ptr %20, i64 %21, i64 %22, i64 %23, i64 %24, i64 %25) {
  %27 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %5, 0
  %28 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, ptr %6, 1
  %29 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, i64 %7, 2
  %30 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, i64 %8, 3, 0
  %31 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, i64 %10, 4, 0
  %32 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, i64 %9, 3, 1
  %33 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, i64 %11, 4, 1
  %34 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %12, 0
  %35 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, ptr %13, 1
  %36 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, i64 %14, 2
  %37 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, i64 %15, 3, 0
  %38 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, i64 %17, 4, 0
  %39 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, i64 %16, 3, 1
  %40 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, i64 %18, 4, 1
  %41 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %19, 0
  %42 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, ptr %20, 1
  %43 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, i64 %21, 2
  %44 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %43, i64 %22, 3, 0
  %45 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, i64 %24, 4, 0
  %46 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %45, i64 %23, 3, 1
  %47 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, i64 %25, 4, 1
  %48 = sext i32 %1 to i64
  %49 = sext i32 %2 to i64
  %50 = sext i32 %0 to i64
  %51 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %33, 3
  %52 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %51, ptr %52, align 8
  %53 = getelementptr [2 x i64], ptr %52, i32 0, i32 0
  %54 = load i64, ptr %53, align 8
  %55 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %33, 3
  %56 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %55, ptr %56, align 8
  %57 = getelementptr [2 x i64], ptr %56, i32 0, i32 1
  %58 = load i64, ptr %57, align 8
  %59 = mul i64 %58, %54
  %60 = getelementptr double, ptr null, i64 %59
  %61 = ptrtoint ptr %60 to i64
  %62 = add i64 %61, 64
  %63 = call ptr @malloc(i64 %62)
  %64 = ptrtoint ptr %63 to i64
  %65 = add i64 %64, 63
  %66 = urem i64 %65, 64
  %67 = sub i64 %65, %66
  %68 = inttoptr i64 %67 to ptr
  %69 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %63, 0
  %70 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %69, ptr %68, 1
  %71 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %70, i64 0, 2
  %72 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %71, i64 %54, 3, 0
  %73 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %72, i64 %58, 3, 1
  %74 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %73, i64 %58, 4, 0
  %75 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %74, i64 1, 4, 1
  %76 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %33, 3, 0
  %77 = mul i64 %76, 1
  %78 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %33, 3, 1
  %79 = mul i64 %77, %78
  %80 = mul i64 %79, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %81 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %33, 1
  %82 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %33, 2
  %83 = getelementptr double, ptr %81, i64 %82
  %84 = getelementptr double, ptr %68, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %84, ptr %83, i64 %80, i1 false)
  br label %85

85:                                               ; preds = %102, %26
  %86 = phi i64 [ %103, %102 ], [ 0, %26 ]
  %87 = icmp slt i64 %86, %54
  br i1 %87, label %88, label %104

88:                                               ; preds = %85
  br label %89

89:                                               ; preds = %92, %88
  %90 = phi i64 [ %101, %92 ], [ 0, %88 ]
  %91 = icmp slt i64 %90, %58
  br i1 %91, label %92, label %102

92:                                               ; preds = %89
  %93 = mul i64 %86, %58
  %94 = add i64 %93, %90
  %95 = getelementptr double, ptr %68, i64 %94
  %96 = load double, ptr %95, align 8
  %97 = fmul double %96, %4
  %98 = mul i64 %86, %58
  %99 = add i64 %98, %90
  %100 = getelementptr double, ptr %68, i64 %99
  store double %97, ptr %100, align 8
  %101 = add i64 %90, 1
  br label %89

102:                                              ; preds = %89
  %103 = add i64 %86, 1
  br label %85

104:                                              ; preds = %85
  %105 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, 0
  %106 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, 1
  %107 = insertvalue { ptr, ptr, i64 } undef, ptr %105, 0
  %108 = insertvalue { ptr, ptr, i64 } %107, ptr %106, 1
  %109 = insertvalue { ptr, ptr, i64 } %108, i64 0, 2
  %110 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, 2
  %111 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, 3, 0
  %112 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, 3, 1
  %113 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, 4, 0
  %114 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, 4, 1
  %115 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %105, 0
  %116 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %115, ptr %106, 1
  %117 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %116, i64 0, 2
  %118 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %117, i64 %50, 3, 0
  %119 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %118, i64 %113, 4, 0
  %120 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %119, i64 %49, 3, 1
  %121 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %120, i64 1, 4, 1
  %122 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %47, 0
  %123 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %47, 1
  %124 = insertvalue { ptr, ptr, i64 } undef, ptr %122, 0
  %125 = insertvalue { ptr, ptr, i64 } %124, ptr %123, 1
  %126 = insertvalue { ptr, ptr, i64 } %125, i64 0, 2
  %127 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %47, 2
  %128 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %47, 3, 0
  %129 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %47, 3, 1
  %130 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %47, 4, 0
  %131 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %47, 4, 1
  %132 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %122, 0
  %133 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %132, ptr %123, 1
  %134 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %133, i64 0, 2
  %135 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %134, i64 %49, 3, 0
  %136 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %135, i64 %130, 4, 0
  %137 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %136, i64 %48, 3, 1
  %138 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %137, i64 1, 4, 1
  %139 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %63, 0
  %140 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %139, ptr %68, 1
  %141 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %140, i64 0, 2
  %142 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %141, i64 %50, 3, 0
  %143 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %142, i64 %58, 4, 0
  %144 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %143, i64 %48, 3, 1
  %145 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %144, i64 1, 4, 1
  br label %146

146:                                              ; preds = %183, %104
  %147 = phi i64 [ %184, %183 ], [ 0, %104 ]
  %148 = icmp slt i64 %147, %50
  br i1 %148, label %149, label %185

149:                                              ; preds = %146
  br label %150

150:                                              ; preds = %181, %149
  %151 = phi i64 [ %182, %181 ], [ 0, %149 ]
  %152 = icmp slt i64 %151, %49
  br i1 %152, label %153, label %183

153:                                              ; preds = %150
  br label %154

154:                                              ; preds = %157, %153
  %155 = phi i64 [ %180, %157 ], [ 0, %153 ]
  %156 = icmp slt i64 %155, %48
  br i1 %156, label %157, label %181

157:                                              ; preds = %154
  %158 = getelementptr double, ptr %106, i64 0
  %159 = mul i64 %147, %113
  %160 = add i64 %159, %151
  %161 = getelementptr double, ptr %158, i64 %160
  %162 = load double, ptr %161, align 8
  %163 = getelementptr double, ptr %123, i64 0
  %164 = mul i64 %151, %130
  %165 = add i64 %164, %155
  %166 = getelementptr double, ptr %163, i64 %165
  %167 = load double, ptr %166, align 8
  %168 = getelementptr double, ptr %68, i64 0
  %169 = mul i64 %147, %58
  %170 = add i64 %169, %155
  %171 = getelementptr double, ptr %168, i64 %170
  %172 = load double, ptr %171, align 8
  %173 = fmul double %3, %162
  %174 = fmul double %173, %167
  %175 = fadd double %172, %174
  %176 = getelementptr double, ptr %68, i64 0
  %177 = mul i64 %147, %58
  %178 = add i64 %177, %155
  %179 = getelementptr double, ptr %176, i64 %178
  store double %175, ptr %179, align 8
  %180 = add i64 %155, 1
  br label %154

181:                                              ; preds = %154
  %182 = add i64 %151, 1
  br label %150

183:                                              ; preds = %150
  %184 = add i64 %147, 1
  br label %146

185:                                              ; preds = %146
  %186 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %63, 0
  %187 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %186, ptr %68, 1
  %188 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %187, i64 0, 2
  %189 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %188, i64 %50, 3, 0
  %190 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %189, i64 %58, 4, 0
  %191 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %190, i64 %48, 3, 1
  %192 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %191, i64 1, 4, 1
  %193 = call ptr @llvm.stacksave.p0()
  %194 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %145, ptr %194, align 8
  %195 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %194, 1
  %196 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %192, ptr %196, align 8
  %197 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %196, 1
  %198 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %195, ptr %198, align 8
  %199 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %197, ptr %199, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), ptr %198, ptr %199)
  call void @llvm.stackrestore.p0(ptr %193)
  %200 = mul i64 %54, 1
  %201 = mul i64 %200, %58
  %202 = mul i64 %201, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %203 = getelementptr double, ptr %68, i64 0
  %204 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %33, 1
  %205 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %33, 2
  %206 = getelementptr double, ptr %204, i64 %205
  call void @llvm.memcpy.p0.p0.i64(ptr %206, ptr %203, i64 %202, i1 false)
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
