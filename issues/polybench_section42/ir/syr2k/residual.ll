; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @malloc(i64)

declare void @free(ptr)

declare void @memrefCopy(i64, ptr, ptr)

define void @kernel_syr2k_impl(i32 %0, i32 %1, double %2, double %3, ptr %4, ptr %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, ptr %11, ptr %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, ptr %18, ptr %19, i64 %20, i64 %21, i64 %22, i64 %23, i64 %24) {
  %26 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %4, 0
  %27 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %26, ptr %5, 1
  %28 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, i64 %6, 2
  %29 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, i64 %7, 3, 0
  %30 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, i64 %9, 4, 0
  %31 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, i64 %8, 3, 1
  %32 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, i64 %10, 4, 1
  %33 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %11, 0
  %34 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %33, ptr %12, 1
  %35 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, i64 %13, 2
  %36 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, i64 %14, 3, 0
  %37 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, i64 %16, 4, 0
  %38 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, i64 %15, 3, 1
  %39 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, i64 %17, 4, 1
  %40 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %18, 0
  %41 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, ptr %19, 1
  %42 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, i64 %20, 2
  %43 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, i64 %21, 3, 0
  %44 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %43, i64 %23, 4, 0
  %45 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, i64 %22, 3, 1
  %46 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %45, i64 %24, 4, 1
  %47 = sext i32 %1 to i64
  %48 = sext i32 %0 to i64
  %49 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, 3
  %50 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %49, ptr %50, align 8
  %51 = getelementptr [2 x i64], ptr %50, i32 0, i32 0
  %52 = load i64, ptr %51, align 8
  %53 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, 3
  %54 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %53, ptr %54, align 8
  %55 = getelementptr [2 x i64], ptr %54, i32 0, i32 1
  %56 = load i64, ptr %55, align 8
  %57 = mul i64 %56, %52
  %58 = getelementptr double, ptr null, i64 %57
  %59 = ptrtoint ptr %58 to i64
  %60 = add i64 %59, 64
  %61 = call ptr @malloc(i64 %60)
  %62 = ptrtoint ptr %61 to i64
  %63 = add i64 %62, 63
  %64 = urem i64 %63, 64
  %65 = sub i64 %63, %64
  %66 = inttoptr i64 %65 to ptr
  %67 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %61, 0
  %68 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %67, ptr %66, 1
  %69 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %68, i64 0, 2
  %70 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %69, i64 %52, 3, 0
  %71 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %70, i64 %56, 3, 1
  %72 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %71, i64 %56, 4, 0
  %73 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %72, i64 1, 4, 1
  %74 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, 3, 0
  %75 = mul i64 %74, 1
  %76 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, 3, 1
  %77 = mul i64 %75, %76
  %78 = mul i64 %77, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %79 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, 1
  %80 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, 2
  %81 = getelementptr double, ptr %79, i64 %80
  %82 = getelementptr double, ptr %66, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %82, ptr %81, i64 %78, i1 false)
  br label %83

83:                                               ; preds = %103, %25
  %84 = phi i64 [ %104, %103 ], [ 0, %25 ]
  %85 = icmp slt i64 %84, %52
  br i1 %85, label %86, label %105

86:                                               ; preds = %83
  br label %87

87:                                               ; preds = %90, %86
  %88 = phi i64 [ %102, %90 ], [ 0, %86 ]
  %89 = icmp slt i64 %88, %56
  br i1 %89, label %90, label %103

90:                                               ; preds = %87
  %91 = mul i64 %84, %56
  %92 = add i64 %91, %88
  %93 = getelementptr double, ptr %66, i64 %92
  %94 = load double, ptr %93, align 8
  %95 = fmul double %94, %3
  %96 = add i64 %84, 1
  %97 = icmp slt i64 %88, %96
  %98 = select i1 %97, double %95, double %94
  %99 = mul i64 %84, %56
  %100 = add i64 %99, %88
  %101 = getelementptr double, ptr %66, i64 %100
  store double %98, ptr %101, align 8
  %102 = add i64 %88, 1
  br label %87

103:                                              ; preds = %87
  %104 = add i64 %84, 1
  br label %83

105:                                              ; preds = %83
  %106 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, 0
  %107 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, 1
  %108 = insertvalue { ptr, ptr, i64 } undef, ptr %106, 0
  %109 = insertvalue { ptr, ptr, i64 } %108, ptr %107, 1
  %110 = insertvalue { ptr, ptr, i64 } %109, i64 0, 2
  %111 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, 2
  %112 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, 3, 0
  %113 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, 3, 1
  %114 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, 4, 0
  %115 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, 4, 1
  %116 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %106, 0
  %117 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %116, ptr %107, 1
  %118 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %117, i64 0, 2
  %119 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %118, i64 %48, 3, 0
  %120 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %119, i64 %114, 4, 0
  %121 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %120, i64 %47, 3, 1
  %122 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %121, i64 1, 4, 1
  %123 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 0
  %124 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 1
  %125 = insertvalue { ptr, ptr, i64 } undef, ptr %123, 0
  %126 = insertvalue { ptr, ptr, i64 } %125, ptr %124, 1
  %127 = insertvalue { ptr, ptr, i64 } %126, i64 0, 2
  %128 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 2
  %129 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 3, 0
  %130 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 3, 1
  %131 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 4, 0
  %132 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 4, 1
  %133 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %123, 0
  %134 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %133, ptr %124, 1
  %135 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %134, i64 0, 2
  %136 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %135, i64 %48, 3, 0
  %137 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %136, i64 %131, 4, 0
  %138 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %137, i64 %47, 3, 1
  %139 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %138, i64 1, 4, 1
  %140 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 0
  %141 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 1
  %142 = insertvalue { ptr, ptr, i64 } undef, ptr %140, 0
  %143 = insertvalue { ptr, ptr, i64 } %142, ptr %141, 1
  %144 = insertvalue { ptr, ptr, i64 } %143, i64 0, 2
  %145 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 2
  %146 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 3, 0
  %147 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 3, 1
  %148 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 4, 0
  %149 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 4, 1
  %150 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %140, 0
  %151 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %150, ptr %141, 1
  %152 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %151, i64 0, 2
  %153 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %152, i64 %48, 3, 0
  %154 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %153, i64 %148, 4, 0
  %155 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %154, i64 %47, 3, 1
  %156 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %155, i64 1, 4, 1
  %157 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, 0
  %158 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, 1
  %159 = insertvalue { ptr, ptr, i64 } undef, ptr %157, 0
  %160 = insertvalue { ptr, ptr, i64 } %159, ptr %158, 1
  %161 = insertvalue { ptr, ptr, i64 } %160, i64 0, 2
  %162 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, 2
  %163 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, 3, 0
  %164 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, 3, 1
  %165 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, 4, 0
  %166 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, 4, 1
  %167 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %157, 0
  %168 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %167, ptr %158, 1
  %169 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %168, i64 0, 2
  %170 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %169, i64 %48, 3, 0
  %171 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %170, i64 %165, 4, 0
  %172 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %171, i64 %47, 3, 1
  %173 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %172, i64 1, 4, 1
  %174 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %61, 0
  %175 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %174, ptr %66, 1
  %176 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %175, i64 0, 2
  %177 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %176, i64 %48, 3, 0
  %178 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %177, i64 %56, 4, 0
  %179 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %178, i64 %48, 3, 1
  %180 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %179, i64 1, 4, 1
  br label %181

181:                                              ; preds = %234, %105
  %182 = phi i64 [ %235, %234 ], [ 0, %105 ]
  %183 = icmp slt i64 %182, %48
  br i1 %183, label %184, label %236

184:                                              ; preds = %181
  br label %185

185:                                              ; preds = %232, %184
  %186 = phi i64 [ %233, %232 ], [ 0, %184 ]
  %187 = icmp slt i64 %186, %47
  br i1 %187, label %188, label %234

188:                                              ; preds = %185
  br label %189

189:                                              ; preds = %192, %188
  %190 = phi i64 [ %231, %192 ], [ 0, %188 ]
  %191 = icmp slt i64 %190, %48
  br i1 %191, label %192, label %232

192:                                              ; preds = %189
  %193 = getelementptr double, ptr %107, i64 0
  %194 = mul i64 %190, %114
  %195 = add i64 %194, %186
  %196 = getelementptr double, ptr %193, i64 %195
  %197 = load double, ptr %196, align 8
  %198 = getelementptr double, ptr %124, i64 0
  %199 = mul i64 %182, %131
  %200 = add i64 %199, %186
  %201 = getelementptr double, ptr %198, i64 %200
  %202 = load double, ptr %201, align 8
  %203 = getelementptr double, ptr %141, i64 0
  %204 = mul i64 %190, %148
  %205 = add i64 %204, %186
  %206 = getelementptr double, ptr %203, i64 %205
  %207 = load double, ptr %206, align 8
  %208 = getelementptr double, ptr %158, i64 0
  %209 = mul i64 %182, %165
  %210 = add i64 %209, %186
  %211 = getelementptr double, ptr %208, i64 %210
  %212 = load double, ptr %211, align 8
  %213 = getelementptr double, ptr %66, i64 0
  %214 = mul i64 %182, %56
  %215 = add i64 %214, %190
  %216 = getelementptr double, ptr %213, i64 %215
  %217 = load double, ptr %216, align 8
  %218 = fmul double %197, %2
  %219 = fmul double %218, %202
  %220 = fmul double %207, %2
  %221 = fmul double %220, %212
  %222 = fadd double %219, %221
  %223 = fadd double %217, %222
  %224 = add i64 %182, 1
  %225 = icmp slt i64 %190, %224
  %226 = select i1 %225, double %223, double %217
  %227 = getelementptr double, ptr %66, i64 0
  %228 = mul i64 %182, %56
  %229 = add i64 %228, %190
  %230 = getelementptr double, ptr %227, i64 %229
  store double %226, ptr %230, align 8
  %231 = add i64 %190, 1
  br label %189

232:                                              ; preds = %189
  %233 = add i64 %186, 1
  br label %185

234:                                              ; preds = %185
  %235 = add i64 %182, 1
  br label %181

236:                                              ; preds = %181
  %237 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %61, 0
  %238 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %237, ptr %66, 1
  %239 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %238, i64 0, 2
  %240 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %239, i64 %48, 3, 0
  %241 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %240, i64 %56, 4, 0
  %242 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %241, i64 %48, 3, 1
  %243 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %242, i64 1, 4, 1
  %244 = call ptr @llvm.stacksave.p0()
  %245 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %180, ptr %245, align 8
  %246 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %245, 1
  %247 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %243, ptr %247, align 8
  %248 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %247, 1
  %249 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %246, ptr %249, align 8
  %250 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %248, ptr %250, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), ptr %249, ptr %250)
  call void @llvm.stackrestore.p0(ptr %244)
  %251 = mul i64 %52, 1
  %252 = mul i64 %251, %56
  %253 = mul i64 %252, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %254 = getelementptr double, ptr %66, i64 0
  %255 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, 1
  %256 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, 2
  %257 = getelementptr double, ptr %255, i64 %256
  call void @llvm.memcpy.p0.p0.i64(ptr %257, ptr %254, i64 %253, i1 false)
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
