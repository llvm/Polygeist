; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @malloc(i64)

declare void @free(ptr)

declare void @memrefCopy(i64, ptr, ptr)

define void @kernel_2mm_impl(i32 %0, i32 %1, i32 %2, i32 %3, double %4, double %5, ptr %6, ptr %7, i64 %8, i64 %9, i64 %10, i64 %11, i64 %12, ptr %13, ptr %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19, ptr %20, ptr %21, i64 %22, i64 %23, i64 %24, i64 %25, i64 %26, ptr %27, ptr %28, i64 %29, i64 %30, i64 %31, i64 %32, i64 %33, ptr %34, ptr %35, i64 %36, i64 %37, i64 %38, i64 %39, i64 %40) {
  %42 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %6, 0
  %43 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, ptr %7, 1
  %44 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %43, i64 %8, 2
  %45 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, i64 %9, 3, 0
  %46 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %45, i64 %11, 4, 0
  %47 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, i64 %10, 3, 1
  %48 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %47, i64 %12, 4, 1
  %49 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %13, 0
  %50 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %49, ptr %14, 1
  %51 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %50, i64 %15, 2
  %52 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %51, i64 %16, 3, 0
  %53 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %52, i64 %18, 4, 0
  %54 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %53, i64 %17, 3, 1
  %55 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %54, i64 %19, 4, 1
  %56 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %20, 0
  %57 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %56, ptr %21, 1
  %58 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %57, i64 %22, 2
  %59 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %58, i64 %23, 3, 0
  %60 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %59, i64 %25, 4, 0
  %61 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, i64 %24, 3, 1
  %62 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %61, i64 %26, 4, 1
  %63 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %27, 0
  %64 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %63, ptr %28, 1
  %65 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %64, i64 %29, 2
  %66 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %65, i64 %30, 3, 0
  %67 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %66, i64 %32, 4, 0
  %68 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %67, i64 %31, 3, 1
  %69 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %68, i64 %33, 4, 1
  %70 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %34, 0
  %71 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %70, ptr %35, 1
  %72 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %71, i64 %36, 2
  %73 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %72, i64 %37, 3, 0
  %74 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %73, i64 %39, 4, 0
  %75 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %74, i64 %38, 3, 1
  %76 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %75, i64 %40, 4, 1
  %77 = sext i32 %2 to i64
  %78 = sext i32 %3 to i64
  %79 = sext i32 %1 to i64
  %80 = sext i32 %0 to i64
  %81 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %48, 3
  %82 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %81, ptr %82, align 8
  %83 = getelementptr [2 x i64], ptr %82, i32 0, i32 0
  %84 = load i64, ptr %83, align 8
  %85 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %48, 3
  %86 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %85, ptr %86, align 8
  %87 = getelementptr [2 x i64], ptr %86, i32 0, i32 1
  %88 = load i64, ptr %87, align 8
  %89 = mul i64 %88, %84
  %90 = getelementptr double, ptr null, i64 %89
  %91 = ptrtoint ptr %90 to i64
  %92 = add i64 %91, 64
  %93 = call ptr @malloc(i64 %92)
  %94 = ptrtoint ptr %93 to i64
  %95 = add i64 %94, 63
  %96 = urem i64 %95, 64
  %97 = sub i64 %95, %96
  %98 = inttoptr i64 %97 to ptr
  %99 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %93, 0
  %100 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %99, ptr %98, 1
  %101 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %100, i64 0, 2
  %102 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %101, i64 %84, 3, 0
  %103 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %102, i64 %88, 3, 1
  %104 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %103, i64 %88, 4, 0
  %105 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %104, i64 1, 4, 1
  br label %106

106:                                              ; preds = %118, %41
  %107 = phi i64 [ %119, %118 ], [ 0, %41 ]
  %108 = icmp slt i64 %107, %84
  br i1 %108, label %109, label %120

109:                                              ; preds = %106
  br label %110

110:                                              ; preds = %113, %109
  %111 = phi i64 [ %117, %113 ], [ 0, %109 ]
  %112 = icmp slt i64 %111, %88
  br i1 %112, label %113, label %118

113:                                              ; preds = %110
  %114 = mul i64 %107, %88
  %115 = add i64 %114, %111
  %116 = getelementptr double, ptr %98, i64 %115
  store double 0.000000e+00, ptr %116, align 8
  %117 = add i64 %111, 1
  br label %110

118:                                              ; preds = %110
  %119 = add i64 %107, 1
  br label %106

120:                                              ; preds = %106
  %121 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, 0
  %122 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, 1
  %123 = insertvalue { ptr, ptr, i64 } undef, ptr %121, 0
  %124 = insertvalue { ptr, ptr, i64 } %123, ptr %122, 1
  %125 = insertvalue { ptr, ptr, i64 } %124, i64 0, 2
  %126 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, 2
  %127 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, 3, 0
  %128 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, 3, 1
  %129 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, 4, 0
  %130 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, 4, 1
  %131 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %121, 0
  %132 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %131, ptr %122, 1
  %133 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %132, i64 0, 2
  %134 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %133, i64 %80, 3, 0
  %135 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %134, i64 %129, 4, 0
  %136 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %135, i64 %77, 3, 1
  %137 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %136, i64 1, 4, 1
  %138 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %62, 0
  %139 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %62, 1
  %140 = insertvalue { ptr, ptr, i64 } undef, ptr %138, 0
  %141 = insertvalue { ptr, ptr, i64 } %140, ptr %139, 1
  %142 = insertvalue { ptr, ptr, i64 } %141, i64 0, 2
  %143 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %62, 2
  %144 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %62, 3, 0
  %145 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %62, 3, 1
  %146 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %62, 4, 0
  %147 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %62, 4, 1
  %148 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %138, 0
  %149 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %148, ptr %139, 1
  %150 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %149, i64 0, 2
  %151 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %150, i64 %77, 3, 0
  %152 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %151, i64 %146, 4, 0
  %153 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %152, i64 %79, 3, 1
  %154 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %153, i64 1, 4, 1
  %155 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %93, 0
  %156 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %155, ptr %98, 1
  %157 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %156, i64 0, 2
  %158 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %157, i64 %80, 3, 0
  %159 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %158, i64 %88, 4, 0
  %160 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %159, i64 %79, 3, 1
  %161 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %160, i64 1, 4, 1
  %162 = mul i64 %79, %80
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
  %175 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %174, i64 %80, 3, 0
  %176 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %175, i64 %79, 3, 1
  %177 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %176, i64 %79, 4, 0
  %178 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %177, i64 1, 4, 1
  %179 = call ptr @llvm.stacksave.p0()
  %180 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %161, ptr %180, align 8
  %181 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %180, 1
  %182 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %178, ptr %182, align 8
  %183 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %182, 1
  %184 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %181, ptr %184, align 8
  %185 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %183, ptr %185, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), ptr %184, ptr %185)
  call void @llvm.stackrestore.p0(ptr %179)
  br label %186

186:                                              ; preds = %221, %120
  %187 = phi i64 [ %222, %221 ], [ 0, %120 ]
  %188 = icmp slt i64 %187, %80
  br i1 %188, label %189, label %223

189:                                              ; preds = %186
  br label %190

190:                                              ; preds = %219, %189
  %191 = phi i64 [ %220, %219 ], [ 0, %189 ]
  %192 = icmp slt i64 %191, %79
  br i1 %192, label %193, label %221

193:                                              ; preds = %190
  br label %194

194:                                              ; preds = %197, %193
  %195 = phi i64 [ %218, %197 ], [ 0, %193 ]
  %196 = icmp slt i64 %195, %77
  br i1 %196, label %197, label %219

197:                                              ; preds = %194
  %198 = getelementptr double, ptr %122, i64 0
  %199 = mul i64 %187, %129
  %200 = add i64 %199, %195
  %201 = getelementptr double, ptr %198, i64 %200
  %202 = load double, ptr %201, align 8
  %203 = getelementptr double, ptr %139, i64 0
  %204 = mul i64 %195, %146
  %205 = add i64 %204, %191
  %206 = getelementptr double, ptr %203, i64 %205
  %207 = load double, ptr %206, align 8
  %208 = mul i64 %187, %79
  %209 = add i64 %208, %191
  %210 = getelementptr double, ptr %171, i64 %209
  %211 = load double, ptr %210, align 8
  %212 = fmul double %4, %202
  %213 = fmul double %212, %207
  %214 = fadd double %211, %213
  %215 = mul i64 %187, %79
  %216 = add i64 %215, %191
  %217 = getelementptr double, ptr %171, i64 %216
  store double %214, ptr %217, align 8
  %218 = add i64 %195, 1
  br label %194

219:                                              ; preds = %194
  %220 = add i64 %191, 1
  br label %190

221:                                              ; preds = %190
  %222 = add i64 %187, 1
  br label %186

223:                                              ; preds = %186
  %224 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %93, 0
  %225 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %224, ptr %98, 1
  %226 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %225, i64 0, 2
  %227 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %226, i64 %80, 3, 0
  %228 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %227, i64 %88, 4, 0
  %229 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %228, i64 %79, 3, 1
  %230 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %229, i64 1, 4, 1
  %231 = call ptr @llvm.stacksave.p0()
  %232 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %178, ptr %232, align 8
  %233 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %232, 1
  %234 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %230, ptr %234, align 8
  %235 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %234, 1
  %236 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %233, ptr %236, align 8
  %237 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %235, ptr %237, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), ptr %236, ptr %237)
  call void @llvm.stackrestore.p0(ptr %231)
  %238 = mul i64 %84, 1
  %239 = mul i64 %238, %88
  %240 = mul i64 %239, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %241 = getelementptr double, ptr %98, i64 0
  %242 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %48, 1
  %243 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %48, 2
  %244 = getelementptr double, ptr %242, i64 %243
  call void @llvm.memcpy.p0.p0.i64(ptr %244, ptr %241, i64 %240, i1 false)
  %245 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %76, 3
  %246 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %245, ptr %246, align 8
  %247 = getelementptr [2 x i64], ptr %246, i32 0, i32 0
  %248 = load i64, ptr %247, align 8
  %249 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %76, 3
  %250 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %249, ptr %250, align 8
  %251 = getelementptr [2 x i64], ptr %250, i32 0, i32 1
  %252 = load i64, ptr %251, align 8
  %253 = mul i64 %252, %248
  %254 = getelementptr double, ptr null, i64 %253
  %255 = ptrtoint ptr %254 to i64
  %256 = add i64 %255, 64
  %257 = call ptr @malloc(i64 %256)
  %258 = ptrtoint ptr %257 to i64
  %259 = add i64 %258, 63
  %260 = urem i64 %259, 64
  %261 = sub i64 %259, %260
  %262 = inttoptr i64 %261 to ptr
  %263 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %257, 0
  %264 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %263, ptr %262, 1
  %265 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %264, i64 0, 2
  %266 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %265, i64 %248, 3, 0
  %267 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %266, i64 %252, 3, 1
  %268 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %267, i64 %252, 4, 0
  %269 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %268, i64 1, 4, 1
  %270 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %76, 3, 0
  %271 = mul i64 %270, 1
  %272 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %76, 3, 1
  %273 = mul i64 %271, %272
  %274 = mul i64 %273, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %275 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %76, 1
  %276 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %76, 2
  %277 = getelementptr double, ptr %275, i64 %276
  %278 = getelementptr double, ptr %262, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %278, ptr %277, i64 %274, i1 false)
  br label %279

279:                                              ; preds = %296, %223
  %280 = phi i64 [ %297, %296 ], [ 0, %223 ]
  %281 = icmp slt i64 %280, %248
  br i1 %281, label %282, label %298

282:                                              ; preds = %279
  br label %283

283:                                              ; preds = %286, %282
  %284 = phi i64 [ %295, %286 ], [ 0, %282 ]
  %285 = icmp slt i64 %284, %252
  br i1 %285, label %286, label %296

286:                                              ; preds = %283
  %287 = mul i64 %280, %252
  %288 = add i64 %287, %284
  %289 = getelementptr double, ptr %262, i64 %288
  %290 = load double, ptr %289, align 8
  %291 = fmul double %290, %5
  %292 = mul i64 %280, %252
  %293 = add i64 %292, %284
  %294 = getelementptr double, ptr %262, i64 %293
  store double %291, ptr %294, align 8
  %295 = add i64 %284, 1
  br label %283

296:                                              ; preds = %283
  %297 = add i64 %280, 1
  br label %279

298:                                              ; preds = %279
  %299 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %69, 0
  %300 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %69, 1
  %301 = insertvalue { ptr, ptr, i64 } undef, ptr %299, 0
  %302 = insertvalue { ptr, ptr, i64 } %301, ptr %300, 1
  %303 = insertvalue { ptr, ptr, i64 } %302, i64 0, 2
  %304 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %69, 2
  %305 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %69, 3, 0
  %306 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %69, 3, 1
  %307 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %69, 4, 0
  %308 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %69, 4, 1
  %309 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %299, 0
  %310 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %309, ptr %300, 1
  %311 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %310, i64 0, 2
  %312 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %311, i64 %79, 3, 0
  %313 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %312, i64 %307, 4, 0
  %314 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %313, i64 %78, 3, 1
  %315 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %314, i64 1, 4, 1
  %316 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %257, 0
  %317 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %316, ptr %262, 1
  %318 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %317, i64 0, 2
  %319 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %318, i64 %80, 3, 0
  %320 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %319, i64 %252, 4, 0
  %321 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %320, i64 %78, 3, 1
  %322 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %321, i64 1, 4, 1
  br label %323

323:                                              ; preds = %358, %298
  %324 = phi i64 [ %359, %358 ], [ 0, %298 ]
  %325 = icmp slt i64 %324, %80
  br i1 %325, label %326, label %360

326:                                              ; preds = %323
  br label %327

327:                                              ; preds = %356, %326
  %328 = phi i64 [ %357, %356 ], [ 0, %326 ]
  %329 = icmp slt i64 %328, %78
  br i1 %329, label %330, label %358

330:                                              ; preds = %327
  br label %331

331:                                              ; preds = %334, %330
  %332 = phi i64 [ %355, %334 ], [ 0, %330 ]
  %333 = icmp slt i64 %332, %79
  br i1 %333, label %334, label %356

334:                                              ; preds = %331
  %335 = mul i64 %324, %79
  %336 = add i64 %335, %332
  %337 = getelementptr double, ptr %171, i64 %336
  %338 = load double, ptr %337, align 8
  %339 = getelementptr double, ptr %300, i64 0
  %340 = mul i64 %332, %307
  %341 = add i64 %340, %328
  %342 = getelementptr double, ptr %339, i64 %341
  %343 = load double, ptr %342, align 8
  %344 = getelementptr double, ptr %262, i64 0
  %345 = mul i64 %324, %252
  %346 = add i64 %345, %328
  %347 = getelementptr double, ptr %344, i64 %346
  %348 = load double, ptr %347, align 8
  %349 = fmul double %338, %343
  %350 = fadd double %348, %349
  %351 = getelementptr double, ptr %262, i64 0
  %352 = mul i64 %324, %252
  %353 = add i64 %352, %328
  %354 = getelementptr double, ptr %351, i64 %353
  store double %350, ptr %354, align 8
  %355 = add i64 %332, 1
  br label %331

356:                                              ; preds = %331
  %357 = add i64 %328, 1
  br label %327

358:                                              ; preds = %327
  %359 = add i64 %324, 1
  br label %323

360:                                              ; preds = %323
  %361 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %257, 0
  %362 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %361, ptr %262, 1
  %363 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %362, i64 0, 2
  %364 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %363, i64 %80, 3, 0
  %365 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %364, i64 %252, 4, 0
  %366 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %365, i64 %78, 3, 1
  %367 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %366, i64 1, 4, 1
  %368 = call ptr @llvm.stacksave.p0()
  %369 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %322, ptr %369, align 8
  %370 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %369, 1
  %371 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %367, ptr %371, align 8
  %372 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %371, 1
  %373 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %370, ptr %373, align 8
  %374 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %372, ptr %374, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), ptr %373, ptr %374)
  call void @llvm.stackrestore.p0(ptr %368)
  %375 = mul i64 %248, 1
  %376 = mul i64 %375, %252
  %377 = mul i64 %376, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %378 = getelementptr double, ptr %262, i64 0
  %379 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %76, 1
  %380 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %76, 2
  %381 = getelementptr double, ptr %379, i64 %380
  call void @llvm.memcpy.p0.p0.i64(ptr %381, ptr %378, i64 %377, i1 false)
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
