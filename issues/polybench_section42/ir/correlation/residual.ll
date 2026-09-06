; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @malloc(i64)

declare void @free(ptr)

declare void @memrefCopy(i64, ptr, ptr)

define void @kernel_correlation_impl(i32 %0, i32 %1, double %2, ptr %3, ptr %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, ptr %10, ptr %11, i64 %12, i64 %13, i64 %14, i64 %15, i64 %16, ptr %17, ptr %18, i64 %19, i64 %20, i64 %21, ptr %22, ptr %23, i64 %24, i64 %25, i64 %26) {
  %28 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %3, 0
  %29 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, ptr %4, 1
  %30 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, i64 %5, 2
  %31 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, i64 %6, 3, 0
  %32 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, i64 %8, 4, 0
  %33 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, i64 %7, 3, 1
  %34 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %33, i64 %9, 4, 1
  %35 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %10, 0
  %36 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, ptr %11, 1
  %37 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, i64 %12, 2
  %38 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, i64 %13, 3, 0
  %39 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, i64 %15, 4, 0
  %40 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, i64 %14, 3, 1
  %41 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, i64 %16, 4, 1
  %42 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %17, 0
  %43 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %42, ptr %18, 1
  %44 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %43, i64 %19, 2
  %45 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %44, i64 %20, 3, 0
  %46 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %45, i64 %21, 4, 0
  %47 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %22, 0
  %48 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %47, ptr %23, 1
  %49 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %48, i64 %24, 2
  %50 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %49, i64 %25, 3, 0
  %51 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %50, i64 %26, 4, 0
  %52 = sext i32 %1 to i64
  %53 = sext i32 %0 to i64
  %54 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %46, 3
  %55 = alloca [1 x i64], i64 1, align 8
  store [1 x i64] %54, ptr %55, align 8
  %56 = getelementptr [1 x i64], ptr %55, i32 0, i32 0
  %57 = load i64, ptr %56, align 8
  %58 = getelementptr double, ptr null, i64 %57
  %59 = ptrtoint ptr %58 to i64
  %60 = add i64 %59, 64
  %61 = call ptr @malloc(i64 %60)
  %62 = ptrtoint ptr %61 to i64
  %63 = add i64 %62, 63
  %64 = urem i64 %63, 64
  %65 = sub i64 %63, %64
  %66 = inttoptr i64 %65 to ptr
  %67 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %61, 0
  %68 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %67, ptr %66, 1
  %69 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %68, i64 0, 2
  %70 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %69, i64 %57, 3, 0
  %71 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %70, i64 1, 4, 0
  br label %72

72:                                               ; preds = %75, %27
  %73 = phi i64 [ %77, %75 ], [ 0, %27 ]
  %74 = icmp slt i64 %73, %57
  br i1 %74, label %75, label %78

75:                                               ; preds = %72
  %76 = getelementptr double, ptr %66, i64 %73
  store double 0.000000e+00, ptr %76, align 8
  %77 = add i64 %73, 1
  br label %72

78:                                               ; preds = %72
  %79 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, 3
  %80 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %79, ptr %80, align 8
  %81 = getelementptr [2 x i64], ptr %80, i32 0, i32 0
  %82 = load i64, ptr %81, align 8
  %83 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, 3
  %84 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %83, ptr %84, align 8
  %85 = getelementptr [2 x i64], ptr %84, i32 0, i32 1
  %86 = load i64, ptr %85, align 8
  br label %87

87:                                               ; preds = %106, %78
  %88 = phi i64 [ %107, %106 ], [ 0, %78 ]
  %89 = icmp slt i64 %88, %86
  br i1 %89, label %90, label %108

90:                                               ; preds = %87
  br label %91

91:                                               ; preds = %94, %90
  %92 = phi i64 [ %105, %94 ], [ 0, %90 ]
  %93 = icmp slt i64 %92, %82
  br i1 %93, label %94, label %106

94:                                               ; preds = %91
  %95 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, 1
  %96 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, 4, 0
  %97 = mul i64 %92, %96
  %98 = add i64 %97, %88
  %99 = getelementptr double, ptr %95, i64 %98
  %100 = load double, ptr %99, align 8
  %101 = getelementptr double, ptr %66, i64 %88
  %102 = load double, ptr %101, align 8
  %103 = fadd double %102, %100
  %104 = getelementptr double, ptr %66, i64 %88
  store double %103, ptr %104, align 8
  %105 = add i64 %92, 1
  br label %91

106:                                              ; preds = %91
  %107 = add i64 %88, 1
  br label %87

108:                                              ; preds = %87
  br label %109

109:                                              ; preds = %112, %108
  %110 = phi i64 [ %117, %112 ], [ 0, %108 ]
  %111 = icmp slt i64 %110, %57
  br i1 %111, label %112, label %118

112:                                              ; preds = %109
  %113 = getelementptr double, ptr %66, i64 %110
  %114 = load double, ptr %113, align 8
  %115 = fdiv double %114, %2
  %116 = getelementptr double, ptr %66, i64 %110
  store double %115, ptr %116, align 8
  %117 = add i64 %110, 1
  br label %109

118:                                              ; preds = %109
  %119 = getelementptr double, ptr null, i64 %57
  %120 = ptrtoint ptr %119 to i64
  %121 = add i64 %120, 64
  %122 = call ptr @malloc(i64 %121)
  %123 = ptrtoint ptr %122 to i64
  %124 = add i64 %123, 63
  %125 = urem i64 %124, 64
  %126 = sub i64 %124, %125
  %127 = inttoptr i64 %126 to ptr
  %128 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %122, 0
  %129 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %128, ptr %127, 1
  %130 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %129, i64 0, 2
  %131 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %130, i64 %57, 3, 0
  %132 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %131, i64 1, 4, 0
  %133 = mul i64 %57, 1
  %134 = mul i64 %133, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %135 = getelementptr double, ptr %66, i64 0
  %136 = getelementptr double, ptr %127, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %136, ptr %135, i64 %134, i1 false)
  %137 = mul i64 %57, 1
  %138 = mul i64 %137, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %139 = getelementptr double, ptr %127, i64 0
  %140 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %46, 1
  %141 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %46, 2
  %142 = getelementptr double, ptr %140, i64 %141
  call void @llvm.memcpy.p0.p0.i64(ptr %142, ptr %139, i64 %138, i1 false)
  %143 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %51, 3
  %144 = alloca [1 x i64], i64 1, align 8
  store [1 x i64] %143, ptr %144, align 8
  %145 = getelementptr [1 x i64], ptr %144, i32 0, i32 0
  %146 = load i64, ptr %145, align 8
  %147 = getelementptr double, ptr null, i64 %146
  %148 = ptrtoint ptr %147 to i64
  %149 = add i64 %148, 64
  %150 = call ptr @malloc(i64 %149)
  %151 = ptrtoint ptr %150 to i64
  %152 = add i64 %151, 63
  %153 = urem i64 %152, 64
  %154 = sub i64 %152, %153
  %155 = inttoptr i64 %154 to ptr
  %156 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %150, 0
  %157 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %156, ptr %155, 1
  %158 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %157, i64 0, 2
  %159 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %158, i64 %146, 3, 0
  %160 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %159, i64 1, 4, 0
  br label %161

161:                                              ; preds = %164, %118
  %162 = phi i64 [ %166, %164 ], [ 0, %118 ]
  %163 = icmp slt i64 %162, %146
  br i1 %163, label %164, label %167

164:                                              ; preds = %161
  %165 = getelementptr double, ptr %155, i64 %162
  store double 0.000000e+00, ptr %165, align 8
  %166 = add i64 %162, 1
  br label %161

167:                                              ; preds = %161
  %168 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, 0
  %169 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, 1
  %170 = insertvalue { ptr, ptr, i64 } undef, ptr %168, 0
  %171 = insertvalue { ptr, ptr, i64 } %170, ptr %169, 1
  %172 = insertvalue { ptr, ptr, i64 } %171, i64 0, 2
  %173 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, 2
  %174 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, 3, 0
  %175 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, 3, 1
  %176 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, 4, 0
  %177 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, 4, 1
  %178 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %168, 0
  %179 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %178, ptr %169, 1
  %180 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %179, i64 0, 2
  %181 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %180, i64 %52, 3, 0
  %182 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %181, i64 %176, 4, 0
  %183 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %182, i64 %53, 3, 1
  %184 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %183, i64 1, 4, 1
  %185 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %61, 0
  %186 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %185, ptr %66, 1
  %187 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %186, i64 0, 2
  %188 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %187, i64 %53, 3, 0
  %189 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %188, i64 1, 4, 0
  %190 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %150, 0
  %191 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %190, ptr %155, 1
  %192 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %191, i64 0, 2
  %193 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %192, i64 %53, 3, 0
  %194 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %193, i64 1, 4, 0
  br label %195

195:                                              ; preds = %217, %167
  %196 = phi i64 [ %218, %217 ], [ 0, %167 ]
  %197 = icmp slt i64 %196, %53
  br i1 %197, label %198, label %219

198:                                              ; preds = %195
  br label %199

199:                                              ; preds = %202, %198
  %200 = phi i64 [ %216, %202 ], [ 0, %198 ]
  %201 = icmp slt i64 %200, %52
  br i1 %201, label %202, label %217

202:                                              ; preds = %199
  %203 = getelementptr double, ptr %169, i64 0
  %204 = mul i64 %200, %176
  %205 = add i64 %204, %196
  %206 = getelementptr double, ptr %203, i64 %205
  %207 = load double, ptr %206, align 8
  %208 = getelementptr double, ptr %66, i64 %196
  %209 = load double, ptr %208, align 8
  %210 = getelementptr double, ptr %155, i64 %196
  %211 = load double, ptr %210, align 8
  %212 = fsub double %207, %209
  %213 = fmul double %212, %212
  %214 = fadd double %211, %213
  %215 = getelementptr double, ptr %155, i64 %196
  store double %214, ptr %215, align 8
  %216 = add i64 %200, 1
  br label %199

217:                                              ; preds = %199
  %218 = add i64 %196, 1
  br label %195

219:                                              ; preds = %195
  %220 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %150, 0
  %221 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %220, ptr %155, 1
  %222 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %221, i64 0, 2
  %223 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %222, i64 %53, 3, 0
  %224 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %223, i64 1, 4, 0
  %225 = mul i64 %53, 1
  %226 = mul i64 %225, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %227 = getelementptr double, ptr %155, i64 0
  %228 = getelementptr double, ptr %155, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %228, ptr %227, i64 %226, i1 false)
  br label %229

229:                                              ; preds = %232, %219
  %230 = phi i64 [ %240, %232 ], [ 0, %219 ]
  %231 = icmp slt i64 %230, %146
  br i1 %231, label %232, label %241

232:                                              ; preds = %229
  %233 = getelementptr double, ptr %155, i64 %230
  %234 = load double, ptr %233, align 8
  %235 = fdiv double %234, %2
  %236 = call double @llvm.sqrt.f64(double %235)
  %237 = fcmp ole double %236, 1.000000e-01
  %238 = select i1 %237, double 1.000000e+00, double %236
  %239 = getelementptr double, ptr %155, i64 %230
  store double %238, ptr %239, align 8
  %240 = add i64 %230, 1
  br label %229

241:                                              ; preds = %229
  %242 = getelementptr double, ptr null, i64 %146
  %243 = ptrtoint ptr %242 to i64
  %244 = add i64 %243, 64
  %245 = call ptr @malloc(i64 %244)
  %246 = ptrtoint ptr %245 to i64
  %247 = add i64 %246, 63
  %248 = urem i64 %247, 64
  %249 = sub i64 %247, %248
  %250 = inttoptr i64 %249 to ptr
  %251 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %245, 0
  %252 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %251, ptr %250, 1
  %253 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %252, i64 0, 2
  %254 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %253, i64 %146, 3, 0
  %255 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %254, i64 1, 4, 0
  %256 = mul i64 %146, 1
  %257 = mul i64 %256, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %258 = getelementptr double, ptr %155, i64 0
  %259 = getelementptr double, ptr %250, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %259, ptr %258, i64 %257, i1 false)
  %260 = mul i64 %146, 1
  %261 = mul i64 %260, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %262 = getelementptr double, ptr %250, i64 0
  %263 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %51, 1
  %264 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %51, 2
  %265 = getelementptr double, ptr %263, i64 %264
  call void @llvm.memcpy.p0.p0.i64(ptr %265, ptr %262, i64 %261, i1 false)
  %266 = call double @llvm.sqrt.f64(double %2)
  %267 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, 3
  %268 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %267, ptr %268, align 8
  %269 = getelementptr [2 x i64], ptr %268, i32 0, i32 0
  %270 = load i64, ptr %269, align 8
  %271 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, 3
  %272 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %271, ptr %272, align 8
  %273 = getelementptr [2 x i64], ptr %272, i32 0, i32 1
  %274 = load i64, ptr %273, align 8
  %275 = mul i64 %274, %270
  %276 = getelementptr double, ptr null, i64 %275
  %277 = ptrtoint ptr %276 to i64
  %278 = add i64 %277, 64
  %279 = call ptr @malloc(i64 %278)
  %280 = ptrtoint ptr %279 to i64
  %281 = add i64 %280, 63
  %282 = urem i64 %281, 64
  %283 = sub i64 %281, %282
  %284 = inttoptr i64 %283 to ptr
  %285 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %279, 0
  %286 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %285, ptr %284, 1
  %287 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %286, i64 0, 2
  %288 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %287, i64 %270, 3, 0
  %289 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %288, i64 %274, 3, 1
  %290 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %289, i64 %274, 4, 0
  %291 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %290, i64 1, 4, 1
  %292 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, 3, 0
  %293 = mul i64 %292, 1
  %294 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, 3, 1
  %295 = mul i64 %293, %294
  %296 = mul i64 %295, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %297 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, 1
  %298 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, 2
  %299 = getelementptr double, ptr %297, i64 %298
  %300 = getelementptr double, ptr %284, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %300, ptr %299, i64 %296, i1 false)
  br label %301

301:                                              ; preds = %335, %241
  %302 = phi i64 [ %336, %335 ], [ 0, %241 ]
  %303 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %308, %335 ], [ %291, %241 ]
  %304 = icmp slt i64 %302, %52
  br i1 %304, label %305, label %337

305:                                              ; preds = %301
  br label %306

306:                                              ; preds = %310, %305
  %307 = phi i64 [ %334, %310 ], [ 0, %305 ]
  %308 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %308, %310 ], [ %303, %305 ]
  %309 = icmp slt i64 %307, %53
  br i1 %309, label %310, label %335

310:                                              ; preds = %306
  %311 = getelementptr double, ptr %66, i64 %307
  %312 = load double, ptr %311, align 8
  %313 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %308, 1
  %314 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %308, 4, 0
  %315 = mul i64 %302, %314
  %316 = add i64 %315, %307
  %317 = getelementptr double, ptr %313, i64 %316
  %318 = load double, ptr %317, align 8
  %319 = fsub double %318, %312
  %320 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %308, 1
  %321 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %308, 4, 0
  %322 = mul i64 %302, %321
  %323 = add i64 %322, %307
  %324 = getelementptr double, ptr %320, i64 %323
  store double %319, ptr %324, align 8
  %325 = getelementptr double, ptr %155, i64 %307
  %326 = load double, ptr %325, align 8
  %327 = fmul double %266, %326
  %328 = fdiv double %319, %327
  %329 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %308, 1
  %330 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %308, 4, 0
  %331 = mul i64 %302, %330
  %332 = add i64 %331, %307
  %333 = getelementptr double, ptr %329, i64 %332
  store double %328, ptr %333, align 8
  %334 = add i64 %307, 1
  br label %306

335:                                              ; preds = %306
  %336 = add i64 %302, 1
  br label %301

337:                                              ; preds = %301
  %338 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %303, 3
  %339 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %338, ptr %339, align 8
  %340 = getelementptr [2 x i64], ptr %339, i32 0, i32 0
  %341 = load i64, ptr %340, align 8
  %342 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %303, 3
  %343 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %342, ptr %343, align 8
  %344 = getelementptr [2 x i64], ptr %343, i32 0, i32 1
  %345 = load i64, ptr %344, align 8
  %346 = mul i64 %345, %341
  %347 = getelementptr double, ptr null, i64 %346
  %348 = ptrtoint ptr %347 to i64
  %349 = add i64 %348, 64
  %350 = call ptr @malloc(i64 %349)
  %351 = ptrtoint ptr %350 to i64
  %352 = add i64 %351, 63
  %353 = urem i64 %352, 64
  %354 = sub i64 %352, %353
  %355 = inttoptr i64 %354 to ptr
  %356 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %350, 0
  %357 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %356, ptr %355, 1
  %358 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %357, i64 0, 2
  %359 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %358, i64 %341, 3, 0
  %360 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %359, i64 %345, 3, 1
  %361 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %360, i64 %345, 4, 0
  %362 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %361, i64 1, 4, 1
  %363 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %303, 3, 0
  %364 = mul i64 %363, 1
  %365 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %303, 3, 1
  %366 = mul i64 %364, %365
  %367 = mul i64 %366, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %368 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %303, 1
  %369 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %303, 2
  %370 = getelementptr double, ptr %368, i64 %369
  %371 = getelementptr double, ptr %355, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %371, ptr %370, i64 %367, i1 false)
  %372 = mul i64 %341, 1
  %373 = mul i64 %372, %345
  %374 = mul i64 %373, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %375 = getelementptr double, ptr %355, i64 0
  %376 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, 1
  %377 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, 2
  %378 = getelementptr double, ptr %376, i64 %377
  call void @llvm.memcpy.p0.p0.i64(ptr %378, ptr %375, i64 %374, i1 false)
  %379 = add i64 %53, -1
  %380 = mul i64 %379, %379
  %381 = getelementptr double, ptr null, i64 %380
  %382 = ptrtoint ptr %381 to i64
  %383 = add i64 %382, 64
  %384 = call ptr @malloc(i64 %383)
  %385 = ptrtoint ptr %384 to i64
  %386 = add i64 %385, 63
  %387 = urem i64 %386, 64
  %388 = sub i64 %386, %387
  %389 = inttoptr i64 %388 to ptr
  %390 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %384, 0
  %391 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %390, ptr %389, 1
  %392 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %391, i64 0, 2
  %393 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %392, i64 %379, 3, 0
  %394 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %393, i64 %379, 3, 1
  %395 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %394, i64 %379, 4, 0
  %396 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %395, i64 1, 4, 1
  br label %397

397:                                              ; preds = %400, %337
  %398 = phi i64 [ %404, %400 ], [ 0, %337 ]
  %399 = icmp slt i64 %398, %379
  br i1 %399, label %400, label %405

400:                                              ; preds = %397
  %401 = mul i64 %398, %379
  %402 = add i64 %401, %398
  %403 = getelementptr double, ptr %389, i64 %402
  store double 1.000000e+00, ptr %403, align 8
  %404 = add i64 %398, 1
  br label %397

405:                                              ; preds = %397
  %406 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, 3
  %407 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %406, ptr %407, align 8
  %408 = getelementptr [2 x i64], ptr %407, i32 0, i32 0
  %409 = load i64, ptr %408, align 8
  %410 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, 3
  %411 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %410, ptr %411, align 8
  %412 = getelementptr [2 x i64], ptr %411, i32 0, i32 1
  %413 = load i64, ptr %412, align 8
  %414 = mul i64 %413, %409
  %415 = getelementptr double, ptr null, i64 %414
  %416 = ptrtoint ptr %415 to i64
  %417 = add i64 %416, 64
  %418 = call ptr @malloc(i64 %417)
  %419 = ptrtoint ptr %418 to i64
  %420 = add i64 %419, 63
  %421 = urem i64 %420, 64
  %422 = sub i64 %420, %421
  %423 = inttoptr i64 %422 to ptr
  %424 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %418, 0
  %425 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %424, ptr %423, 1
  %426 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %425, i64 0, 2
  %427 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %426, i64 %409, 3, 0
  %428 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %427, i64 %413, 3, 1
  %429 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %428, i64 %413, 4, 0
  %430 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %429, i64 1, 4, 1
  %431 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, 3, 0
  %432 = mul i64 %431, 1
  %433 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, 3, 1
  %434 = mul i64 %432, %433
  %435 = mul i64 %434, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %436 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, 1
  %437 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, 2
  %438 = getelementptr double, ptr %436, i64 %437
  %439 = getelementptr double, ptr %423, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %439, ptr %438, i64 %435, i1 false)
  %440 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %418, 0
  %441 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %440, ptr %423, 1
  %442 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %441, i64 0, 2
  %443 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %442, i64 %379, 3, 0
  %444 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %443, i64 %413, 4, 0
  %445 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %444, i64 %379, 3, 1
  %446 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %445, i64 1, 4, 1
  %447 = call ptr @llvm.stacksave.p0()
  %448 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %396, ptr %448, align 8
  %449 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %448, 1
  %450 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %446, ptr %450, align 8
  %451 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %450, 1
  %452 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %449, ptr %452, align 8
  %453 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %451, ptr %453, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), ptr %452, ptr %453)
  call void @llvm.stackrestore.p0(ptr %447)
  br label %454

454:                                              ; preds = %473, %405
  %455 = phi i64 [ %474, %473 ], [ 0, %405 ]
  %456 = icmp slt i64 %455, %409
  br i1 %456, label %457, label %475

457:                                              ; preds = %454
  br label %458

458:                                              ; preds = %461, %457
  %459 = phi i64 [ %472, %461 ], [ 0, %457 ]
  %460 = icmp slt i64 %459, %413
  br i1 %460, label %461, label %473

461:                                              ; preds = %458
  %462 = mul i64 %455, %413
  %463 = add i64 %462, %459
  %464 = getelementptr double, ptr %423, i64 %463
  %465 = load double, ptr %464, align 8
  %466 = add i64 %455, 1
  %467 = icmp sge i64 %459, %466
  %468 = select i1 %467, double 0.000000e+00, double %465
  %469 = mul i64 %455, %413
  %470 = add i64 %469, %459
  %471 = getelementptr double, ptr %423, i64 %470
  store double %468, ptr %471, align 8
  %472 = add i64 %459, 1
  br label %458

473:                                              ; preds = %458
  %474 = add i64 %455, 1
  br label %454

475:                                              ; preds = %454
  %476 = add i64 %53, -1
  %477 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %303, 0
  %478 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %303, 1
  %479 = insertvalue { ptr, ptr, i64 } undef, ptr %477, 0
  %480 = insertvalue { ptr, ptr, i64 } %479, ptr %478, 1
  %481 = insertvalue { ptr, ptr, i64 } %480, i64 0, 2
  %482 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %303, 2
  %483 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %303, 3, 0
  %484 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %303, 3, 1
  %485 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %303, 4, 0
  %486 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %303, 4, 1
  %487 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %477, 0
  %488 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %487, ptr %478, 1
  %489 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %488, i64 0, 2
  %490 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %489, i64 %52, 3, 0
  %491 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %490, i64 %485, 4, 0
  %492 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %491, i64 %476, 3, 1
  %493 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %492, i64 1, 4, 1
  %494 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %303, 0
  %495 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %303, 1
  %496 = insertvalue { ptr, ptr, i64 } undef, ptr %494, 0
  %497 = insertvalue { ptr, ptr, i64 } %496, ptr %495, 1
  %498 = insertvalue { ptr, ptr, i64 } %497, i64 0, 2
  %499 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %303, 2
  %500 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %303, 3, 0
  %501 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %303, 3, 1
  %502 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %303, 4, 0
  %503 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %303, 4, 1
  %504 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %494, 0
  %505 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %504, ptr %495, 1
  %506 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %505, i64 0, 2
  %507 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %506, i64 %52, 3, 0
  %508 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %507, i64 %502, 4, 0
  %509 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %508, i64 %53, 3, 1
  %510 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %509, i64 1, 4, 1
  %511 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %418, 0
  %512 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %511, ptr %423, 1
  %513 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %512, i64 0, 2
  %514 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %513, i64 %476, 3, 0
  %515 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %514, i64 %413, 4, 0
  %516 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %515, i64 %53, 3, 1
  %517 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %516, i64 1, 4, 1
  br label %518

518:                                              ; preds = %554, %475
  %519 = phi i64 [ %555, %554 ], [ 0, %475 ]
  %520 = icmp slt i64 %519, %476
  br i1 %520, label %521, label %556

521:                                              ; preds = %518
  br label %522

522:                                              ; preds = %552, %521
  %523 = phi i64 [ %553, %552 ], [ 0, %521 ]
  %524 = icmp slt i64 %523, %53
  br i1 %524, label %525, label %554

525:                                              ; preds = %522
  br label %526

526:                                              ; preds = %529, %525
  %527 = phi i64 [ %551, %529 ], [ 0, %525 ]
  %528 = icmp slt i64 %527, %52
  br i1 %528, label %529, label %552

529:                                              ; preds = %526
  %530 = getelementptr double, ptr %478, i64 0
  %531 = mul i64 %527, %485
  %532 = add i64 %531, %519
  %533 = getelementptr double, ptr %530, i64 %532
  %534 = load double, ptr %533, align 8
  %535 = getelementptr double, ptr %495, i64 0
  %536 = mul i64 %527, %502
  %537 = add i64 %536, %523
  %538 = getelementptr double, ptr %535, i64 %537
  %539 = load double, ptr %538, align 8
  %540 = getelementptr double, ptr %423, i64 0
  %541 = mul i64 %519, %413
  %542 = add i64 %541, %523
  %543 = getelementptr double, ptr %540, i64 %542
  %544 = load double, ptr %543, align 8
  %545 = fmul double %534, %539
  %546 = fadd double %544, %545
  %547 = getelementptr double, ptr %423, i64 0
  %548 = mul i64 %519, %413
  %549 = add i64 %548, %523
  %550 = getelementptr double, ptr %547, i64 %549
  store double %546, ptr %550, align 8
  %551 = add i64 %527, 1
  br label %526

552:                                              ; preds = %526
  %553 = add i64 %523, 1
  br label %522

554:                                              ; preds = %522
  %555 = add i64 %519, 1
  br label %518

556:                                              ; preds = %518
  %557 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %418, 0
  %558 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %557, ptr %423, 1
  %559 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %558, i64 0, 2
  %560 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %559, i64 %476, 3, 0
  %561 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %560, i64 %413, 4, 0
  %562 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %561, i64 %53, 3, 1
  %563 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %562, i64 1, 4, 1
  %564 = call ptr @llvm.stacksave.p0()
  %565 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %517, ptr %565, align 8
  %566 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %565, 1
  %567 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %563, ptr %567, align 8
  %568 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %567, 1
  %569 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %566, ptr %569, align 8
  %570 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %568, ptr %570, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), ptr %569, ptr %570)
  call void @llvm.stackrestore.p0(ptr %564)
  %571 = mul i64 %413, %409
  %572 = getelementptr double, ptr null, i64 %571
  %573 = ptrtoint ptr %572 to i64
  %574 = add i64 %573, 64
  %575 = call ptr @malloc(i64 %574)
  %576 = ptrtoint ptr %575 to i64
  %577 = add i64 %576, 63
  %578 = urem i64 %577, 64
  %579 = sub i64 %577, %578
  %580 = inttoptr i64 %579 to ptr
  %581 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %575, 0
  %582 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %581, ptr %580, 1
  %583 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %582, i64 0, 2
  %584 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %583, i64 %409, 3, 0
  %585 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %584, i64 %413, 3, 1
  %586 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %585, i64 %413, 4, 0
  %587 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %586, i64 1, 4, 1
  %588 = mul i64 %409, 1
  %589 = mul i64 %588, %413
  %590 = mul i64 %589, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %591 = getelementptr double, ptr %423, i64 0
  %592 = getelementptr double, ptr %580, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %592, ptr %591, i64 %590, i1 false)
  br label %593

593:                                              ; preds = %616, %556
  %594 = phi i64 [ %617, %616 ], [ 0, %556 ]
  %595 = icmp slt i64 %594, %409
  br i1 %595, label %596, label %618

596:                                              ; preds = %593
  br label %597

597:                                              ; preds = %600, %596
  %598 = phi i64 [ %615, %600 ], [ 0, %596 ]
  %599 = icmp slt i64 %598, %413
  br i1 %599, label %600, label %616

600:                                              ; preds = %597
  %601 = mul i64 %594, %413
  %602 = add i64 %601, %598
  %603 = getelementptr double, ptr %423, i64 %602
  %604 = load double, ptr %603, align 8
  %605 = mul i64 %598, %413
  %606 = add i64 %605, %594
  %607 = getelementptr double, ptr %580, i64 %606
  %608 = load double, ptr %607, align 8
  %609 = add i64 %594, 1
  %610 = icmp sge i64 %598, %609
  %611 = select i1 %610, double %604, double %608
  %612 = mul i64 %598, %413
  %613 = add i64 %612, %594
  %614 = getelementptr double, ptr %580, i64 %613
  store double %611, ptr %614, align 8
  %615 = add i64 %598, 1
  br label %597

616:                                              ; preds = %597
  %617 = add i64 %594, 1
  br label %593

618:                                              ; preds = %593
  %619 = add i64 %53, -1
  %620 = add i64 %53, -1
  %621 = mul i64 %619, %413
  %622 = add i64 %621, %620
  %623 = getelementptr double, ptr %580, i64 %622
  store double 1.000000e+00, ptr %623, align 8
  %624 = mul i64 %409, 1
  %625 = mul i64 %624, %413
  %626 = mul i64 %625, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %627 = getelementptr double, ptr %580, i64 0
  %628 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, 1
  %629 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, 2
  %630 = getelementptr double, ptr %628, i64 %629
  call void @llvm.memcpy.p0.p0.i64(ptr %630, ptr %627, i64 %626, i1 false)
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.sqrt.f64(double) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare ptr @llvm.stacksave.p0() #2

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.stackrestore.p0(ptr) #2

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
