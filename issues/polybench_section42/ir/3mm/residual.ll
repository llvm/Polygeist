; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @malloc(i64)

declare void @free(ptr)

declare void @memrefCopy(i64, ptr, ptr)

define void @kernel_3mm_impl(i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, ptr %5, ptr %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11, ptr %12, ptr %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, ptr %19, ptr %20, i64 %21, i64 %22, i64 %23, i64 %24, i64 %25, ptr %26, ptr %27, i64 %28, i64 %29, i64 %30, i64 %31, i64 %32, ptr %33, ptr %34, i64 %35, i64 %36, i64 %37, i64 %38, i64 %39, ptr %40, ptr %41, i64 %42, i64 %43, i64 %44, i64 %45, i64 %46, ptr %47, ptr %48, i64 %49, i64 %50, i64 %51, i64 %52, i64 %53) {
  %55 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %5, 0
  %56 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, ptr %6, 1
  %57 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %56, i64 %7, 2
  %58 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %57, i64 %8, 3, 0
  %59 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %58, i64 %10, 4, 0
  %60 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %59, i64 %9, 3, 1
  %61 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, i64 %11, 4, 1
  %62 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %12, 0
  %63 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %62, ptr %13, 1
  %64 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %63, i64 %14, 2
  %65 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %64, i64 %15, 3, 0
  %66 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %65, i64 %17, 4, 0
  %67 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %66, i64 %16, 3, 1
  %68 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %67, i64 %18, 4, 1
  %69 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %19, 0
  %70 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %69, ptr %20, 1
  %71 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %70, i64 %21, 2
  %72 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %71, i64 %22, 3, 0
  %73 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %72, i64 %24, 4, 0
  %74 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %73, i64 %23, 3, 1
  %75 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %74, i64 %25, 4, 1
  %76 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %26, 0
  %77 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %76, ptr %27, 1
  %78 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %77, i64 %28, 2
  %79 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %78, i64 %29, 3, 0
  %80 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %79, i64 %31, 4, 0
  %81 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %80, i64 %30, 3, 1
  %82 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %81, i64 %32, 4, 1
  %83 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %33, 0
  %84 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %83, ptr %34, 1
  %85 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %84, i64 %35, 2
  %86 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %85, i64 %36, 3, 0
  %87 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %86, i64 %38, 4, 0
  %88 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %87, i64 %37, 3, 1
  %89 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %88, i64 %39, 4, 1
  %90 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %40, 0
  %91 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %90, ptr %41, 1
  %92 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %91, i64 %42, 2
  %93 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %92, i64 %43, 3, 0
  %94 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %93, i64 %45, 4, 0
  %95 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %94, i64 %44, 3, 1
  %96 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %95, i64 %46, 4, 1
  %97 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %47, 0
  %98 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %97, ptr %48, 1
  %99 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %98, i64 %49, 2
  %100 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %99, i64 %50, 3, 0
  %101 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %100, i64 %52, 4, 0
  %102 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %101, i64 %51, 3, 1
  %103 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %102, i64 %53, 4, 1
  %104 = sext i32 %1 to i64
  %105 = sext i32 %2 to i64
  %106 = sext i32 %4 to i64
  %107 = sext i32 %3 to i64
  %108 = sext i32 %0 to i64
  %109 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %61, 3
  %110 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %109, ptr %110, align 8
  %111 = getelementptr [2 x i64], ptr %110, i32 0, i32 0
  %112 = load i64, ptr %111, align 8
  %113 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %61, 3
  %114 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %113, ptr %114, align 8
  %115 = getelementptr [2 x i64], ptr %114, i32 0, i32 1
  %116 = load i64, ptr %115, align 8
  %117 = mul i64 %116, %112
  %118 = getelementptr double, ptr null, i64 %117
  %119 = ptrtoint ptr %118 to i64
  %120 = add i64 %119, 64
  %121 = call ptr @malloc(i64 %120)
  %122 = ptrtoint ptr %121 to i64
  %123 = add i64 %122, 63
  %124 = urem i64 %123, 64
  %125 = sub i64 %123, %124
  %126 = inttoptr i64 %125 to ptr
  %127 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %121, 0
  %128 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %127, ptr %126, 1
  %129 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %128, i64 0, 2
  %130 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %129, i64 %112, 3, 0
  %131 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %130, i64 %116, 3, 1
  %132 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %131, i64 %116, 4, 0
  %133 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %132, i64 1, 4, 1
  br label %134

134:                                              ; preds = %146, %54
  %135 = phi i64 [ %147, %146 ], [ 0, %54 ]
  %136 = icmp slt i64 %135, %112
  br i1 %136, label %137, label %148

137:                                              ; preds = %134
  br label %138

138:                                              ; preds = %141, %137
  %139 = phi i64 [ %145, %141 ], [ 0, %137 ]
  %140 = icmp slt i64 %139, %116
  br i1 %140, label %141, label %146

141:                                              ; preds = %138
  %142 = mul i64 %135, %116
  %143 = add i64 %142, %139
  %144 = getelementptr double, ptr %126, i64 %143
  store double 0.000000e+00, ptr %144, align 8
  %145 = add i64 %139, 1
  br label %138

146:                                              ; preds = %138
  %147 = add i64 %135, 1
  br label %134

148:                                              ; preds = %134
  %149 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %68, 0
  %150 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %68, 1
  %151 = insertvalue { ptr, ptr, i64 } undef, ptr %149, 0
  %152 = insertvalue { ptr, ptr, i64 } %151, ptr %150, 1
  %153 = insertvalue { ptr, ptr, i64 } %152, i64 0, 2
  %154 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %68, 2
  %155 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %68, 3, 0
  %156 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %68, 3, 1
  %157 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %68, 4, 0
  %158 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %68, 4, 1
  %159 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %149, 0
  %160 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %159, ptr %150, 1
  %161 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %160, i64 0, 2
  %162 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %161, i64 %108, 3, 0
  %163 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %162, i64 %157, 4, 0
  %164 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %163, i64 %105, 3, 1
  %165 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %164, i64 1, 4, 1
  %166 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %75, 0
  %167 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %75, 1
  %168 = insertvalue { ptr, ptr, i64 } undef, ptr %166, 0
  %169 = insertvalue { ptr, ptr, i64 } %168, ptr %167, 1
  %170 = insertvalue { ptr, ptr, i64 } %169, i64 0, 2
  %171 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %75, 2
  %172 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %75, 3, 0
  %173 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %75, 3, 1
  %174 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %75, 4, 0
  %175 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %75, 4, 1
  %176 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %166, 0
  %177 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %176, ptr %167, 1
  %178 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %177, i64 0, 2
  %179 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %178, i64 %105, 3, 0
  %180 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %179, i64 %174, 4, 0
  %181 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %180, i64 %104, 3, 1
  %182 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %181, i64 1, 4, 1
  %183 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %121, 0
  %184 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %183, ptr %126, 1
  %185 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %184, i64 0, 2
  %186 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %185, i64 %108, 3, 0
  %187 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %186, i64 %116, 4, 0
  %188 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %187, i64 %104, 3, 1
  %189 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %188, i64 1, 4, 1
  %190 = mul i64 %104, %108
  %191 = getelementptr double, ptr null, i64 %190
  %192 = ptrtoint ptr %191 to i64
  %193 = add i64 %192, 64
  %194 = call ptr @malloc(i64 %193)
  %195 = ptrtoint ptr %194 to i64
  %196 = add i64 %195, 63
  %197 = urem i64 %196, 64
  %198 = sub i64 %196, %197
  %199 = inttoptr i64 %198 to ptr
  %200 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %194, 0
  %201 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %200, ptr %199, 1
  %202 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %201, i64 0, 2
  %203 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %202, i64 %108, 3, 0
  %204 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %203, i64 %104, 3, 1
  %205 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %204, i64 %104, 4, 0
  %206 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %205, i64 1, 4, 1
  %207 = call ptr @llvm.stacksave.p0()
  %208 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %189, ptr %208, align 8
  %209 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %208, 1
  %210 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %206, ptr %210, align 8
  %211 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %210, 1
  %212 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %209, ptr %212, align 8
  %213 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %211, ptr %213, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), ptr %212, ptr %213)
  call void @llvm.stackrestore.p0(ptr %207)
  br label %214

214:                                              ; preds = %248, %148
  %215 = phi i64 [ %249, %248 ], [ 0, %148 ]
  %216 = icmp slt i64 %215, %108
  br i1 %216, label %217, label %250

217:                                              ; preds = %214
  br label %218

218:                                              ; preds = %246, %217
  %219 = phi i64 [ %247, %246 ], [ 0, %217 ]
  %220 = icmp slt i64 %219, %104
  br i1 %220, label %221, label %248

221:                                              ; preds = %218
  br label %222

222:                                              ; preds = %225, %221
  %223 = phi i64 [ %245, %225 ], [ 0, %221 ]
  %224 = icmp slt i64 %223, %105
  br i1 %224, label %225, label %246

225:                                              ; preds = %222
  %226 = getelementptr double, ptr %150, i64 0
  %227 = mul i64 %215, %157
  %228 = add i64 %227, %223
  %229 = getelementptr double, ptr %226, i64 %228
  %230 = load double, ptr %229, align 8
  %231 = getelementptr double, ptr %167, i64 0
  %232 = mul i64 %223, %174
  %233 = add i64 %232, %219
  %234 = getelementptr double, ptr %231, i64 %233
  %235 = load double, ptr %234, align 8
  %236 = mul i64 %215, %104
  %237 = add i64 %236, %219
  %238 = getelementptr double, ptr %199, i64 %237
  %239 = load double, ptr %238, align 8
  %240 = fmul double %230, %235
  %241 = fadd double %239, %240
  %242 = mul i64 %215, %104
  %243 = add i64 %242, %219
  %244 = getelementptr double, ptr %199, i64 %243
  store double %241, ptr %244, align 8
  %245 = add i64 %223, 1
  br label %222

246:                                              ; preds = %222
  %247 = add i64 %219, 1
  br label %218

248:                                              ; preds = %218
  %249 = add i64 %215, 1
  br label %214

250:                                              ; preds = %214
  %251 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %121, 0
  %252 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %251, ptr %126, 1
  %253 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %252, i64 0, 2
  %254 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %253, i64 %108, 3, 0
  %255 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %254, i64 %116, 4, 0
  %256 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %255, i64 %104, 3, 1
  %257 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %256, i64 1, 4, 1
  %258 = call ptr @llvm.stacksave.p0()
  %259 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %206, ptr %259, align 8
  %260 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %259, 1
  %261 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %257, ptr %261, align 8
  %262 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %261, 1
  %263 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %260, ptr %263, align 8
  %264 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %262, ptr %264, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), ptr %263, ptr %264)
  call void @llvm.stackrestore.p0(ptr %258)
  %265 = mul i64 %112, 1
  %266 = mul i64 %265, %116
  %267 = mul i64 %266, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %268 = getelementptr double, ptr %126, i64 0
  %269 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %61, 1
  %270 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %61, 2
  %271 = getelementptr double, ptr %269, i64 %270
  call void @llvm.memcpy.p0.p0.i64(ptr %271, ptr %268, i64 %267, i1 false)
  %272 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %82, 3
  %273 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %272, ptr %273, align 8
  %274 = getelementptr [2 x i64], ptr %273, i32 0, i32 0
  %275 = load i64, ptr %274, align 8
  %276 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %82, 3
  %277 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %276, ptr %277, align 8
  %278 = getelementptr [2 x i64], ptr %277, i32 0, i32 1
  %279 = load i64, ptr %278, align 8
  %280 = mul i64 %279, %275
  %281 = getelementptr double, ptr null, i64 %280
  %282 = ptrtoint ptr %281 to i64
  %283 = add i64 %282, 64
  %284 = call ptr @malloc(i64 %283)
  %285 = ptrtoint ptr %284 to i64
  %286 = add i64 %285, 63
  %287 = urem i64 %286, 64
  %288 = sub i64 %286, %287
  %289 = inttoptr i64 %288 to ptr
  %290 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %284, 0
  %291 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %290, ptr %289, 1
  %292 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %291, i64 0, 2
  %293 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %292, i64 %275, 3, 0
  %294 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %293, i64 %279, 3, 1
  %295 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %294, i64 %279, 4, 0
  %296 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %295, i64 1, 4, 1
  br label %297

297:                                              ; preds = %309, %250
  %298 = phi i64 [ %310, %309 ], [ 0, %250 ]
  %299 = icmp slt i64 %298, %275
  br i1 %299, label %300, label %311

300:                                              ; preds = %297
  br label %301

301:                                              ; preds = %304, %300
  %302 = phi i64 [ %308, %304 ], [ 0, %300 ]
  %303 = icmp slt i64 %302, %279
  br i1 %303, label %304, label %309

304:                                              ; preds = %301
  %305 = mul i64 %298, %279
  %306 = add i64 %305, %302
  %307 = getelementptr double, ptr %289, i64 %306
  store double 0.000000e+00, ptr %307, align 8
  %308 = add i64 %302, 1
  br label %301

309:                                              ; preds = %301
  %310 = add i64 %298, 1
  br label %297

311:                                              ; preds = %297
  %312 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %89, 0
  %313 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %89, 1
  %314 = insertvalue { ptr, ptr, i64 } undef, ptr %312, 0
  %315 = insertvalue { ptr, ptr, i64 } %314, ptr %313, 1
  %316 = insertvalue { ptr, ptr, i64 } %315, i64 0, 2
  %317 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %89, 2
  %318 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %89, 3, 0
  %319 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %89, 3, 1
  %320 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %89, 4, 0
  %321 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %89, 4, 1
  %322 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %312, 0
  %323 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %322, ptr %313, 1
  %324 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %323, i64 0, 2
  %325 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %324, i64 %104, 3, 0
  %326 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %325, i64 %320, 4, 0
  %327 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %326, i64 %106, 3, 1
  %328 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %327, i64 1, 4, 1
  %329 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %96, 0
  %330 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %96, 1
  %331 = insertvalue { ptr, ptr, i64 } undef, ptr %329, 0
  %332 = insertvalue { ptr, ptr, i64 } %331, ptr %330, 1
  %333 = insertvalue { ptr, ptr, i64 } %332, i64 0, 2
  %334 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %96, 2
  %335 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %96, 3, 0
  %336 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %96, 3, 1
  %337 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %96, 4, 0
  %338 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %96, 4, 1
  %339 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %329, 0
  %340 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %339, ptr %330, 1
  %341 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %340, i64 0, 2
  %342 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %341, i64 %106, 3, 0
  %343 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %342, i64 %337, 4, 0
  %344 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %343, i64 %107, 3, 1
  %345 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %344, i64 1, 4, 1
  %346 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %284, 0
  %347 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %346, ptr %289, 1
  %348 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %347, i64 0, 2
  %349 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %348, i64 %104, 3, 0
  %350 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %349, i64 %279, 4, 0
  %351 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %350, i64 %107, 3, 1
  %352 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %351, i64 1, 4, 1
  %353 = mul i64 %107, %104
  %354 = getelementptr double, ptr null, i64 %353
  %355 = ptrtoint ptr %354 to i64
  %356 = add i64 %355, 64
  %357 = call ptr @malloc(i64 %356)
  %358 = ptrtoint ptr %357 to i64
  %359 = add i64 %358, 63
  %360 = urem i64 %359, 64
  %361 = sub i64 %359, %360
  %362 = inttoptr i64 %361 to ptr
  %363 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %357, 0
  %364 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %363, ptr %362, 1
  %365 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %364, i64 0, 2
  %366 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %365, i64 %104, 3, 0
  %367 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %366, i64 %107, 3, 1
  %368 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %367, i64 %107, 4, 0
  %369 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %368, i64 1, 4, 1
  %370 = call ptr @llvm.stacksave.p0()
  %371 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %352, ptr %371, align 8
  %372 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %371, 1
  %373 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %369, ptr %373, align 8
  %374 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %373, 1
  %375 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %372, ptr %375, align 8
  %376 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %374, ptr %376, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), ptr %375, ptr %376)
  call void @llvm.stackrestore.p0(ptr %370)
  br label %377

377:                                              ; preds = %411, %311
  %378 = phi i64 [ %412, %411 ], [ 0, %311 ]
  %379 = icmp slt i64 %378, %104
  br i1 %379, label %380, label %413

380:                                              ; preds = %377
  br label %381

381:                                              ; preds = %409, %380
  %382 = phi i64 [ %410, %409 ], [ 0, %380 ]
  %383 = icmp slt i64 %382, %107
  br i1 %383, label %384, label %411

384:                                              ; preds = %381
  br label %385

385:                                              ; preds = %388, %384
  %386 = phi i64 [ %408, %388 ], [ 0, %384 ]
  %387 = icmp slt i64 %386, %106
  br i1 %387, label %388, label %409

388:                                              ; preds = %385
  %389 = getelementptr double, ptr %313, i64 0
  %390 = mul i64 %378, %320
  %391 = add i64 %390, %386
  %392 = getelementptr double, ptr %389, i64 %391
  %393 = load double, ptr %392, align 8
  %394 = getelementptr double, ptr %330, i64 0
  %395 = mul i64 %386, %337
  %396 = add i64 %395, %382
  %397 = getelementptr double, ptr %394, i64 %396
  %398 = load double, ptr %397, align 8
  %399 = mul i64 %378, %107
  %400 = add i64 %399, %382
  %401 = getelementptr double, ptr %362, i64 %400
  %402 = load double, ptr %401, align 8
  %403 = fmul double %393, %398
  %404 = fadd double %402, %403
  %405 = mul i64 %378, %107
  %406 = add i64 %405, %382
  %407 = getelementptr double, ptr %362, i64 %406
  store double %404, ptr %407, align 8
  %408 = add i64 %386, 1
  br label %385

409:                                              ; preds = %385
  %410 = add i64 %382, 1
  br label %381

411:                                              ; preds = %381
  %412 = add i64 %378, 1
  br label %377

413:                                              ; preds = %377
  %414 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %284, 0
  %415 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %414, ptr %289, 1
  %416 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %415, i64 0, 2
  %417 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %416, i64 %104, 3, 0
  %418 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %417, i64 %279, 4, 0
  %419 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %418, i64 %107, 3, 1
  %420 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %419, i64 1, 4, 1
  %421 = call ptr @llvm.stacksave.p0()
  %422 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %369, ptr %422, align 8
  %423 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %422, 1
  %424 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %420, ptr %424, align 8
  %425 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %424, 1
  %426 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %423, ptr %426, align 8
  %427 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %425, ptr %427, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), ptr %426, ptr %427)
  call void @llvm.stackrestore.p0(ptr %421)
  %428 = mul i64 %275, 1
  %429 = mul i64 %428, %279
  %430 = mul i64 %429, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %431 = getelementptr double, ptr %289, i64 0
  %432 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %82, 1
  %433 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %82, 2
  %434 = getelementptr double, ptr %432, i64 %433
  call void @llvm.memcpy.p0.p0.i64(ptr %434, ptr %431, i64 %430, i1 false)
  %435 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %103, 3
  %436 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %435, ptr %436, align 8
  %437 = getelementptr [2 x i64], ptr %436, i32 0, i32 0
  %438 = load i64, ptr %437, align 8
  %439 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %103, 3
  %440 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %439, ptr %440, align 8
  %441 = getelementptr [2 x i64], ptr %440, i32 0, i32 1
  %442 = load i64, ptr %441, align 8
  %443 = mul i64 %442, %438
  %444 = getelementptr double, ptr null, i64 %443
  %445 = ptrtoint ptr %444 to i64
  %446 = add i64 %445, 64
  %447 = call ptr @malloc(i64 %446)
  %448 = ptrtoint ptr %447 to i64
  %449 = add i64 %448, 63
  %450 = urem i64 %449, 64
  %451 = sub i64 %449, %450
  %452 = inttoptr i64 %451 to ptr
  %453 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %447, 0
  %454 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %453, ptr %452, 1
  %455 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %454, i64 0, 2
  %456 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %455, i64 %438, 3, 0
  %457 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %456, i64 %442, 3, 1
  %458 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %457, i64 %442, 4, 0
  %459 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %458, i64 1, 4, 1
  br label %460

460:                                              ; preds = %472, %413
  %461 = phi i64 [ %473, %472 ], [ 0, %413 ]
  %462 = icmp slt i64 %461, %438
  br i1 %462, label %463, label %474

463:                                              ; preds = %460
  br label %464

464:                                              ; preds = %467, %463
  %465 = phi i64 [ %471, %467 ], [ 0, %463 ]
  %466 = icmp slt i64 %465, %442
  br i1 %466, label %467, label %472

467:                                              ; preds = %464
  %468 = mul i64 %461, %442
  %469 = add i64 %468, %465
  %470 = getelementptr double, ptr %452, i64 %469
  store double 0.000000e+00, ptr %470, align 8
  %471 = add i64 %465, 1
  br label %464

472:                                              ; preds = %464
  %473 = add i64 %461, 1
  br label %460

474:                                              ; preds = %460
  %475 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %447, 0
  %476 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %475, ptr %452, 1
  %477 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %476, i64 0, 2
  %478 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %477, i64 %108, 3, 0
  %479 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %478, i64 %442, 4, 0
  %480 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %479, i64 %107, 3, 1
  %481 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %480, i64 1, 4, 1
  br label %482

482:                                              ; preds = %516, %474
  %483 = phi i64 [ %517, %516 ], [ 0, %474 ]
  %484 = icmp slt i64 %483, %108
  br i1 %484, label %485, label %518

485:                                              ; preds = %482
  br label %486

486:                                              ; preds = %514, %485
  %487 = phi i64 [ %515, %514 ], [ 0, %485 ]
  %488 = icmp slt i64 %487, %107
  br i1 %488, label %489, label %516

489:                                              ; preds = %486
  br label %490

490:                                              ; preds = %493, %489
  %491 = phi i64 [ %513, %493 ], [ 0, %489 ]
  %492 = icmp slt i64 %491, %104
  br i1 %492, label %493, label %514

493:                                              ; preds = %490
  %494 = mul i64 %483, %104
  %495 = add i64 %494, %491
  %496 = getelementptr double, ptr %199, i64 %495
  %497 = load double, ptr %496, align 8
  %498 = mul i64 %491, %107
  %499 = add i64 %498, %487
  %500 = getelementptr double, ptr %362, i64 %499
  %501 = load double, ptr %500, align 8
  %502 = getelementptr double, ptr %452, i64 0
  %503 = mul i64 %483, %442
  %504 = add i64 %503, %487
  %505 = getelementptr double, ptr %502, i64 %504
  %506 = load double, ptr %505, align 8
  %507 = fmul double %497, %501
  %508 = fadd double %506, %507
  %509 = getelementptr double, ptr %452, i64 0
  %510 = mul i64 %483, %442
  %511 = add i64 %510, %487
  %512 = getelementptr double, ptr %509, i64 %511
  store double %508, ptr %512, align 8
  %513 = add i64 %491, 1
  br label %490

514:                                              ; preds = %490
  %515 = add i64 %487, 1
  br label %486

516:                                              ; preds = %486
  %517 = add i64 %483, 1
  br label %482

518:                                              ; preds = %482
  %519 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %447, 0
  %520 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %519, ptr %452, 1
  %521 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %520, i64 0, 2
  %522 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %521, i64 %108, 3, 0
  %523 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %522, i64 %442, 4, 0
  %524 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %523, i64 %107, 3, 1
  %525 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %524, i64 1, 4, 1
  %526 = call ptr @llvm.stacksave.p0()
  %527 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %481, ptr %527, align 8
  %528 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %527, 1
  %529 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %525, ptr %529, align 8
  %530 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %529, 1
  %531 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %528, ptr %531, align 8
  %532 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %530, ptr %532, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), ptr %531, ptr %532)
  call void @llvm.stackrestore.p0(ptr %526)
  %533 = mul i64 %438, 1
  %534 = mul i64 %533, %442
  %535 = mul i64 %534, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %536 = getelementptr double, ptr %452, i64 0
  %537 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %103, 1
  %538 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %103, 2
  %539 = getelementptr double, ptr %537, i64 %538
  call void @llvm.memcpy.p0.p0.i64(ptr %539, ptr %536, i64 %535, i1 false)
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
