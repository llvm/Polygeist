; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @malloc(i64)

declare void @free(ptr)

declare void @memrefCopy(i64, ptr, ptr)

define void @kernel_trmm_impl(i32 %0, i32 %1, double %2, ptr %3, ptr %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, ptr %10, ptr %11, i64 %12, i64 %13, i64 %14, i64 %15, i64 %16) {
  %18 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %3, 0
  %19 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %18, ptr %4, 1
  %20 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %19, i64 %5, 2
  %21 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %20, i64 %6, 3, 0
  %22 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, i64 %8, 4, 0
  %23 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, i64 %7, 3, 1
  %24 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, i64 %9, 4, 1
  %25 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %10, 0
  %26 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, ptr %11, 1
  %27 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %26, i64 %12, 2
  %28 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, i64 %13, 3, 0
  %29 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, i64 %15, 4, 0
  %30 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, i64 %14, 3, 1
  %31 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, i64 %16, 4, 1
  %32 = sext i32 %1 to i64
  %33 = sext i32 %0 to i64
  %34 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, 3
  %35 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %34, ptr %35, align 8
  %36 = getelementptr [2 x i64], ptr %35, i32 0, i32 0
  %37 = load i64, ptr %36, align 8
  %38 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, 3
  %39 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %38, ptr %39, align 8
  %40 = getelementptr [2 x i64], ptr %39, i32 0, i32 1
  %41 = load i64, ptr %40, align 8
  %42 = mul i64 %41, %37
  %43 = getelementptr double, ptr null, i64 %42
  %44 = ptrtoint ptr %43 to i64
  %45 = add i64 %44, 64
  %46 = call ptr @malloc(i64 %45)
  %47 = ptrtoint ptr %46 to i64
  %48 = add i64 %47, 63
  %49 = urem i64 %48, 64
  %50 = sub i64 %48, %49
  %51 = inttoptr i64 %50 to ptr
  %52 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %46, 0
  %53 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %52, ptr %51, 1
  %54 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %53, i64 0, 2
  %55 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %54, i64 %37, 3, 0
  %56 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, i64 %41, 3, 1
  %57 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %56, i64 %41, 4, 0
  %58 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %57, i64 1, 4, 1
  %59 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, 3, 0
  %60 = mul i64 %59, 1
  %61 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, 3, 1
  %62 = mul i64 %60, %61
  %63 = mul i64 %62, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %64 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, 1
  %65 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, 2
  %66 = getelementptr double, ptr %64, i64 %65
  %67 = getelementptr double, ptr %51, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %67, ptr %66, i64 %63, i1 false)
  br label %68

68:                                               ; preds = %186, %17
  %69 = phi i64 [ %210, %186 ], [ 0, %17 ]
  %70 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %70, %186 ], [ %58, %17 ]
  %71 = icmp slt i64 %69, %33
  br i1 %71, label %72, label %211

72:                                               ; preds = %68
  %73 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, 0
  %74 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, 1
  %75 = insertvalue { ptr, ptr, i64 } undef, ptr %73, 0
  %76 = insertvalue { ptr, ptr, i64 } %75, ptr %74, 1
  %77 = insertvalue { ptr, ptr, i64 } %76, i64 0, 2
  %78 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, 2
  %79 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, 3, 0
  %80 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, 3, 1
  %81 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, 4, 0
  %82 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, 4, 1
  %83 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %73, 0
  %84 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %83, ptr %74, 1
  %85 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %84, i64 %69, 2
  %86 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %85, i64 %33, 3, 0
  %87 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %86, i64 %81, 4, 0
  %88 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %70, 0
  %89 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %70, 1
  %90 = insertvalue { ptr, ptr, i64 } undef, ptr %88, 0
  %91 = insertvalue { ptr, ptr, i64 } %90, ptr %89, 1
  %92 = insertvalue { ptr, ptr, i64 } %91, i64 0, 2
  %93 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %70, 2
  %94 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %70, 3, 0
  %95 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %70, 3, 1
  %96 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %70, 4, 0
  %97 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %70, 4, 1
  %98 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %88, 0
  %99 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %98, ptr %89, 1
  %100 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %99, i64 0, 2
  %101 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %100, i64 %33, 3, 0
  %102 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %101, i64 %96, 4, 0
  %103 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %102, i64 %32, 3, 1
  %104 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %103, i64 1, 4, 1
  %105 = mul i64 %32, %33
  %106 = getelementptr double, ptr null, i64 %105
  %107 = ptrtoint ptr %106 to i64
  %108 = add i64 %107, 64
  %109 = call ptr @malloc(i64 %108)
  %110 = ptrtoint ptr %109 to i64
  %111 = add i64 %110, 63
  %112 = urem i64 %111, 64
  %113 = sub i64 %111, %112
  %114 = inttoptr i64 %113 to ptr
  %115 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %109, 0
  %116 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %115, ptr %114, 1
  %117 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %116, i64 0, 2
  %118 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %117, i64 %33, 3, 0
  %119 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %118, i64 %32, 3, 1
  %120 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %119, i64 %32, 4, 0
  %121 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %120, i64 1, 4, 1
  %122 = call ptr @llvm.stacksave.p0()
  %123 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %104, ptr %123, align 8
  %124 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %123, 1
  %125 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %121, ptr %125, align 8
  %126 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %125, 1
  %127 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %124, ptr %127, align 8
  %128 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %126, ptr %128, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), ptr %127, ptr %128)
  call void @llvm.stackrestore.p0(ptr %122)
  %129 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %70, 0
  %130 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %70, 1
  %131 = insertvalue { ptr, ptr, i64 } undef, ptr %129, 0
  %132 = insertvalue { ptr, ptr, i64 } %131, ptr %130, 1
  %133 = insertvalue { ptr, ptr, i64 } %132, i64 0, 2
  %134 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %70, 2
  %135 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %70, 3, 0
  %136 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %70, 3, 1
  %137 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %70, 4, 0
  %138 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %70, 4, 1
  %139 = mul i64 %69, %137
  %140 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %129, 0
  %141 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %140, ptr %130, 1
  %142 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %141, i64 %139, 2
  %143 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %142, i64 %32, 3, 0
  %144 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %143, i64 1, 4, 0
  br label %145

145:                                              ; preds = %172, %72
  %146 = phi i64 [ %173, %172 ], [ 0, %72 ]
  %147 = icmp slt i64 %146, %32
  br i1 %147, label %148, label %174

148:                                              ; preds = %145
  br label %149

149:                                              ; preds = %152, %148
  %150 = phi i64 [ %171, %152 ], [ 0, %148 ]
  %151 = icmp slt i64 %150, %33
  br i1 %151, label %152, label %172

152:                                              ; preds = %149
  %153 = getelementptr double, ptr %74, i64 %69
  %154 = mul i64 %150, %81
  %155 = getelementptr double, ptr %153, i64 %154
  %156 = load double, ptr %155, align 8
  %157 = mul i64 %150, %32
  %158 = add i64 %157, %146
  %159 = getelementptr double, ptr %114, i64 %158
  %160 = load double, ptr %159, align 8
  %161 = getelementptr double, ptr %130, i64 %139
  %162 = getelementptr double, ptr %161, i64 %146
  %163 = load double, ptr %162, align 8
  %164 = fmul double %156, %160
  %165 = fadd double %163, %164
  %166 = add i64 %69, 1
  %167 = icmp sge i64 %150, %166
  %168 = select i1 %167, double %165, double %163
  %169 = getelementptr double, ptr %130, i64 %139
  %170 = getelementptr double, ptr %169, i64 %146
  store double %168, ptr %170, align 8
  %171 = add i64 %150, 1
  br label %149

172:                                              ; preds = %149
  %173 = add i64 %146, 1
  br label %145

174:                                              ; preds = %145
  br label %175

175:                                              ; preds = %178, %174
  %176 = phi i64 [ %185, %178 ], [ 0, %174 ]
  %177 = icmp slt i64 %176, %32
  br i1 %177, label %178, label %186

178:                                              ; preds = %175
  %179 = getelementptr double, ptr %130, i64 %139
  %180 = getelementptr double, ptr %179, i64 %176
  %181 = load double, ptr %180, align 8
  %182 = fmul double %2, %181
  %183 = getelementptr double, ptr %130, i64 %139
  %184 = getelementptr double, ptr %183, i64 %176
  store double %182, ptr %184, align 8
  %185 = add i64 %176, 1
  br label %175

186:                                              ; preds = %175
  %187 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %70, 0
  %188 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %70, 1
  %189 = insertvalue { ptr, ptr, i64 } undef, ptr %187, 0
  %190 = insertvalue { ptr, ptr, i64 } %189, ptr %188, 1
  %191 = insertvalue { ptr, ptr, i64 } %190, i64 0, 2
  %192 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %70, 2
  %193 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %70, 3, 0
  %194 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %70, 3, 1
  %195 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %70, 4, 0
  %196 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %70, 4, 1
  %197 = mul i64 %69, %195
  %198 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %187, 0
  %199 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %198, ptr %188, 1
  %200 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %199, i64 %197, 2
  %201 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %200, i64 %32, 3, 0
  %202 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %201, i64 1, 4, 0
  %203 = call ptr @llvm.stacksave.p0()
  %204 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %144, ptr %204, align 8
  %205 = insertvalue { i64, ptr } { i64 1, ptr undef }, ptr %204, 1
  %206 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %202, ptr %206, align 8
  %207 = insertvalue { i64, ptr } { i64 1, ptr undef }, ptr %206, 1
  %208 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %205, ptr %208, align 8
  %209 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %207, ptr %209, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), ptr %208, ptr %209)
  call void @llvm.stackrestore.p0(ptr %203)
  %210 = add i64 %69, 1
  br label %68

211:                                              ; preds = %68
  %212 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %70, 3, 0
  %213 = mul i64 %212, 1
  %214 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %70, 3, 1
  %215 = mul i64 %213, %214
  %216 = mul i64 %215, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %217 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %70, 1
  %218 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %70, 2
  %219 = getelementptr double, ptr %217, i64 %218
  %220 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, 1
  %221 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, 2
  %222 = getelementptr double, ptr %220, i64 %221
  call void @llvm.memcpy.p0.p0.i64(ptr %222, ptr %219, i64 %216, i1 false)
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
