; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @malloc(i64)

declare void @free(ptr)

declare void @memrefCopy(i64, ptr, ptr)

define void @kernel_deriche_impl(i32 %0, i32 %1, double %2, ptr %3, ptr %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, ptr %10, ptr %11, i64 %12, i64 %13, i64 %14, i64 %15, i64 %16, ptr %17, ptr %18, i64 %19, i64 %20, i64 %21, i64 %22, i64 %23, ptr %24, ptr %25, i64 %26, i64 %27, i64 %28, i64 %29, i64 %30) {
  %32 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %3, 0
  %33 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, ptr %4, 1
  %34 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %33, i64 %5, 2
  %35 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, i64 %6, 3, 0
  %36 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, i64 %8, 4, 0
  %37 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, i64 %7, 3, 1
  %38 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, i64 %9, 4, 1
  %39 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %17, 0
  %40 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, ptr %18, 1
  %41 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, i64 %19, 2
  %42 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, i64 %20, 3, 0
  %43 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, i64 %22, 4, 0
  %44 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %43, i64 %21, 3, 1
  %45 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, i64 %23, 4, 1
  %46 = sext i32 %1 to i64
  %47 = fneg double %2
  %48 = call double @llvm.exp.f64(double %47)
  %49 = fsub double 1.000000e+00, %48
  %50 = fmul double %49, %49
  %51 = fmul double %2, 2.000000e+00
  %52 = fmul double %51, %48
  %53 = fadd double %52, 1.000000e+00
  %54 = call double @llvm.exp.f64(double %51)
  %55 = fsub double %53, %54
  %56 = fdiv double %50, %55
  %57 = fmul double %56, %48
  %58 = fsub double %2, 1.000000e+00
  %59 = fmul double %57, %58
  %60 = call double @llvm.pow.f64(double 2.000000e+00, double %47)
  %61 = fmul double %2, -2.000000e+00
  %62 = call double @llvm.exp.f64(double %61)
  %63 = fneg double %62
  %64 = sext i32 %0 to i64
  %65 = getelementptr double, ptr null, i64 %64
  %66 = ptrtoint ptr %65 to i64
  %67 = add i64 %66, 64
  %68 = call ptr @malloc(i64 %67)
  %69 = ptrtoint ptr %68 to i64
  %70 = add i64 %69, 63
  %71 = urem i64 %70, 64
  %72 = sub i64 %70, %71
  %73 = inttoptr i64 %72 to ptr
  %74 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %68, 0
  %75 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %74, ptr %73, 1
  %76 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %75, i64 0, 2
  %77 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %76, i64 %64, 3, 0
  %78 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %77, i64 1, 4, 0
  %79 = getelementptr double, ptr null, i64 %64
  %80 = ptrtoint ptr %79 to i64
  %81 = add i64 %80, 64
  %82 = call ptr @malloc(i64 %81)
  %83 = ptrtoint ptr %82 to i64
  %84 = add i64 %83, 63
  %85 = urem i64 %84, 64
  %86 = sub i64 %84, %85
  %87 = inttoptr i64 %86 to ptr
  %88 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %82, 0
  %89 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %88, ptr %87, 1
  %90 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %89, i64 0, 2
  %91 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %90, i64 %64, 3, 0
  %92 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %91, i64 1, 4, 0
  %93 = getelementptr double, ptr null, i64 %64
  %94 = ptrtoint ptr %93 to i64
  %95 = add i64 %94, 64
  %96 = call ptr @malloc(i64 %95)
  %97 = ptrtoint ptr %96 to i64
  %98 = add i64 %97, 63
  %99 = urem i64 %98, 64
  %100 = sub i64 %98, %99
  %101 = inttoptr i64 %100 to ptr
  %102 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %96, 0
  %103 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %102, ptr %101, 1
  %104 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %103, i64 0, 2
  %105 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %104, i64 %64, 3, 0
  %106 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %105, i64 1, 4, 0
  br label %107

107:                                              ; preds = %110, %31
  %108 = phi i64 [ %112, %110 ], [ 0, %31 ]
  %109 = icmp slt i64 %108, %64
  br i1 %109, label %110, label %113

110:                                              ; preds = %107
  %111 = getelementptr double, ptr %73, i64 %108
  store double 0.000000e+00, ptr %111, align 8
  %112 = add i64 %108, 1
  br label %107

113:                                              ; preds = %107
  br label %114

114:                                              ; preds = %117, %113
  %115 = phi i64 [ %119, %117 ], [ 0, %113 ]
  %116 = icmp slt i64 %115, %64
  br i1 %116, label %117, label %120

117:                                              ; preds = %114
  %118 = getelementptr double, ptr %87, i64 %115
  store double 0.000000e+00, ptr %118, align 8
  %119 = add i64 %115, 1
  br label %114

120:                                              ; preds = %114
  br label %121

121:                                              ; preds = %124, %120
  %122 = phi i64 [ %126, %124 ], [ 0, %120 ]
  %123 = icmp slt i64 %122, %64
  br i1 %123, label %124, label %127

124:                                              ; preds = %121
  %125 = getelementptr double, ptr %101, i64 %122
  store double 0.000000e+00, ptr %125, align 8
  %126 = add i64 %122, 1
  br label %121

127:                                              ; preds = %121
  %128 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, 0
  %129 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, 1
  %130 = insertvalue { ptr, ptr, i64 } undef, ptr %128, 0
  %131 = insertvalue { ptr, ptr, i64 } %130, ptr %129, 1
  %132 = insertvalue { ptr, ptr, i64 } %131, i64 0, 2
  %133 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, 2
  %134 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, 3, 0
  %135 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, 3, 1
  %136 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, 4, 0
  %137 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, 4, 1
  %138 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %128, 0
  %139 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %138, ptr %129, 1
  %140 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %139, i64 0, 2
  %141 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %140, i64 %64, 3, 0
  %142 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %141, i64 %136, 4, 0
  %143 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %142, i64 %46, 3, 1
  %144 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %143, i64 1, 4, 1
  %145 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, 0
  %146 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, 1
  %147 = insertvalue { ptr, ptr, i64 } undef, ptr %145, 0
  %148 = insertvalue { ptr, ptr, i64 } %147, ptr %146, 1
  %149 = insertvalue { ptr, ptr, i64 } %148, i64 0, 2
  %150 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, 2
  %151 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, 3, 0
  %152 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, 3, 1
  %153 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, 4, 0
  %154 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, 4, 1
  %155 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %145, 0
  %156 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %155, ptr %146, 1
  %157 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %156, i64 0, 2
  %158 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %157, i64 %64, 3, 0
  %159 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %158, i64 %153, 4, 0
  %160 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %159, i64 %46, 3, 1
  %161 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %160, i64 1, 4, 1
  %162 = mul i64 %46, %64
  %163 = getelementptr double, ptr null, i64 %162
  %164 = ptrtoint ptr %163 to i64
  %165 = add i64 %164, 64
  %166 = call ptr @malloc(i64 %165)
  %167 = ptrtoint ptr %166 to i64
  %168 = add i64 %167, 63
  %169 = urem i64 %168, 64
  %170 = sub i64 %168, %169
  %171 = inttoptr i64 %170 to ptr
  %172 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %166, 0
  %173 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %172, ptr %171, 1
  %174 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %173, i64 0, 2
  %175 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %174, i64 %64, 3, 0
  %176 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %175, i64 %46, 3, 1
  %177 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %176, i64 %46, 4, 0
  %178 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %177, i64 1, 4, 1
  %179 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %96, 0
  %180 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %179, ptr %101, 1
  %181 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %180, i64 0, 2
  %182 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %181, i64 %64, 3, 0
  %183 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %182, i64 1, 4, 0
  %184 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %82, 0
  %185 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %184, ptr %87, 1
  %186 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %185, i64 0, 2
  %187 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %186, i64 %64, 3, 0
  %188 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %187, i64 1, 4, 0
  %189 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %68, 0
  %190 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %189, ptr %73, 1
  %191 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %190, i64 0, 2
  %192 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %191, i64 %64, 3, 0
  %193 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %192, i64 1, 4, 0
  br label %194

194:                                              ; preds = %232, %127
  %195 = phi i64 [ %233, %232 ], [ 0, %127 ]
  %196 = icmp slt i64 %195, %64
  br i1 %196, label %197, label %234

197:                                              ; preds = %194
  br label %198

198:                                              ; preds = %201, %197
  %199 = phi i64 [ %231, %201 ], [ 0, %197 ]
  %200 = icmp slt i64 %199, %46
  br i1 %200, label %201, label %232

201:                                              ; preds = %198
  %202 = getelementptr double, ptr %129, i64 0
  %203 = mul i64 %195, %136
  %204 = add i64 %203, %199
  %205 = getelementptr double, ptr %202, i64 %204
  %206 = load double, ptr %205, align 8
  %207 = getelementptr double, ptr %146, i64 0
  %208 = mul i64 %195, %153
  %209 = add i64 %208, %199
  %210 = getelementptr double, ptr %207, i64 %209
  %211 = load double, ptr %210, align 8
  %212 = getelementptr double, ptr %101, i64 %195
  %213 = load double, ptr %212, align 8
  %214 = getelementptr double, ptr %87, i64 %195
  %215 = load double, ptr %214, align 8
  %216 = getelementptr double, ptr %73, i64 %195
  %217 = load double, ptr %216, align 8
  %218 = fmul double %56, %206
  %219 = fmul double %59, %213
  %220 = fadd double %218, %219
  %221 = fmul double %60, %217
  %222 = fadd double %220, %221
  %223 = fmul double %63, %215
  %224 = fadd double %222, %223
  %225 = mul i64 %195, %46
  %226 = add i64 %225, %199
  %227 = getelementptr double, ptr %171, i64 %226
  store double %224, ptr %227, align 8
  %228 = getelementptr double, ptr %101, i64 %195
  store double %211, ptr %228, align 8
  %229 = getelementptr double, ptr %87, i64 %195
  store double %217, ptr %229, align 8
  %230 = getelementptr double, ptr %73, i64 %195
  store double %224, ptr %230, align 8
  %231 = add i64 %199, 1
  br label %198

232:                                              ; preds = %198
  %233 = add i64 %195, 1
  br label %194

234:                                              ; preds = %194
  %235 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %45, 3
  %236 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %235, ptr %236, align 8
  %237 = getelementptr [2 x i64], ptr %236, i32 0, i32 0
  %238 = load i64, ptr %237, align 8
  %239 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %45, 3
  %240 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %239, ptr %240, align 8
  %241 = getelementptr [2 x i64], ptr %240, i32 0, i32 1
  %242 = load i64, ptr %241, align 8
  %243 = mul i64 %242, %238
  %244 = getelementptr double, ptr null, i64 %243
  %245 = ptrtoint ptr %244 to i64
  %246 = add i64 %245, 64
  %247 = call ptr @malloc(i64 %246)
  %248 = ptrtoint ptr %247 to i64
  %249 = add i64 %248, 63
  %250 = urem i64 %249, 64
  %251 = sub i64 %249, %250
  %252 = inttoptr i64 %251 to ptr
  %253 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %247, 0
  %254 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %253, ptr %252, 1
  %255 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %254, i64 0, 2
  %256 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %255, i64 %238, 3, 0
  %257 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %256, i64 %242, 3, 1
  %258 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %257, i64 %242, 4, 0
  %259 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %258, i64 1, 4, 1
  %260 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %45, 3, 0
  %261 = mul i64 %260, 1
  %262 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %45, 3, 1
  %263 = mul i64 %261, %262
  %264 = mul i64 %263, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %265 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %45, 1
  %266 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %45, 2
  %267 = getelementptr double, ptr %265, i64 %266
  %268 = getelementptr double, ptr %252, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %268, ptr %267, i64 %264, i1 false)
  %269 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %247, 0
  %270 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %269, ptr %252, 1
  %271 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %270, i64 0, 2
  %272 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %271, i64 %64, 3, 0
  %273 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %272, i64 %242, 4, 0
  %274 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %273, i64 %46, 3, 1
  %275 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %274, i64 1, 4, 1
  %276 = call ptr @llvm.stacksave.p0()
  %277 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %178, ptr %277, align 8
  %278 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %277, 1
  %279 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %275, ptr %279, align 8
  %280 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %279, 1
  %281 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %278, ptr %281, align 8
  %282 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %280, ptr %282, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), ptr %281, ptr %282)
  call void @llvm.stackrestore.p0(ptr %276)
  %283 = mul i64 %238, 1
  %284 = mul i64 %283, %242
  %285 = mul i64 %284, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %286 = getelementptr double, ptr %252, i64 0
  %287 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %45, 1
  %288 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %45, 2
  %289 = getelementptr double, ptr %287, i64 %288
  call void @llvm.memcpy.p0.p0.i64(ptr %289, ptr %286, i64 %285, i1 false)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.exp.f64(double) #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.pow.f64(double, double) #0

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare ptr @llvm.stacksave.p0() #2

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.stackrestore.p0(ptr) #2

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
