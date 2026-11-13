; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @malloc(i64)

declare void @free(ptr)

define void @kernel_gesummv_impl(i32 %0, double %1, double %2, ptr %3, ptr %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, ptr %10, ptr %11, i64 %12, i64 %13, i64 %14, i64 %15, i64 %16, ptr %17, ptr %18, i64 %19, i64 %20, i64 %21, ptr %22, ptr %23, i64 %24, i64 %25, i64 %26, ptr %27, ptr %28, i64 %29, i64 %30, i64 %31) {
  %33 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %3, 0
  %34 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %33, ptr %4, 1
  %35 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, i64 %5, 2
  %36 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, i64 %6, 3, 0
  %37 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, i64 %8, 4, 0
  %38 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, i64 %7, 3, 1
  %39 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, i64 %9, 4, 1
  %40 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %10, 0
  %41 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, ptr %11, 1
  %42 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, i64 %12, 2
  %43 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, i64 %13, 3, 0
  %44 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %43, i64 %15, 4, 0
  %45 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, i64 %14, 3, 1
  %46 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %45, i64 %16, 4, 1
  %47 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %17, 0
  %48 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %47, ptr %18, 1
  %49 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %48, i64 %19, 2
  %50 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %49, i64 %20, 3, 0
  %51 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %50, i64 %21, 4, 0
  %52 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %22, 0
  %53 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %52, ptr %23, 1
  %54 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %53, i64 %24, 2
  %55 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %54, i64 %25, 3, 0
  %56 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %55, i64 %26, 4, 0
  %57 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %27, 0
  %58 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %57, ptr %28, 1
  %59 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %58, i64 %29, 2
  %60 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %59, i64 %30, 3, 0
  %61 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %60, i64 %31, 4, 0
  %62 = sext i32 %0 to i64
  %63 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %51, 3
  %64 = alloca [1 x i64], i64 1, align 8
  store [1 x i64] %63, ptr %64, align 8
  %65 = getelementptr [1 x i64], ptr %64, i32 0, i32 0
  %66 = load i64, ptr %65, align 8
  %67 = getelementptr double, ptr null, i64 %66
  %68 = ptrtoint ptr %67 to i64
  %69 = add i64 %68, 64
  %70 = call ptr @malloc(i64 %69)
  %71 = ptrtoint ptr %70 to i64
  %72 = add i64 %71, 63
  %73 = urem i64 %72, 64
  %74 = sub i64 %72, %73
  %75 = inttoptr i64 %74 to ptr
  %76 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %70, 0
  %77 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %76, ptr %75, 1
  %78 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %77, i64 0, 2
  %79 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %78, i64 %66, 3, 0
  %80 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %79, i64 1, 4, 0
  br label %81

81:                                               ; preds = %84, %32
  %82 = phi i64 [ %86, %84 ], [ 0, %32 ]
  %83 = icmp slt i64 %82, %66
  br i1 %83, label %84, label %87

84:                                               ; preds = %81
  %85 = getelementptr double, ptr %75, i64 %82
  store double 0.000000e+00, ptr %85, align 8
  %86 = add i64 %82, 1
  br label %81

87:                                               ; preds = %81
  %88 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %61, 3
  %89 = alloca [1 x i64], i64 1, align 8
  store [1 x i64] %88, ptr %89, align 8
  %90 = getelementptr [1 x i64], ptr %89, i32 0, i32 0
  %91 = load i64, ptr %90, align 8
  %92 = getelementptr double, ptr null, i64 %91
  %93 = ptrtoint ptr %92 to i64
  %94 = add i64 %93, 64
  %95 = call ptr @malloc(i64 %94)
  %96 = ptrtoint ptr %95 to i64
  %97 = add i64 %96, 63
  %98 = urem i64 %97, 64
  %99 = sub i64 %97, %98
  %100 = inttoptr i64 %99 to ptr
  %101 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %95, 0
  %102 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %101, ptr %100, 1
  %103 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %102, i64 0, 2
  %104 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %103, i64 %91, 3, 0
  %105 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %104, i64 1, 4, 0
  br label %106

106:                                              ; preds = %109, %87
  %107 = phi i64 [ %111, %109 ], [ 0, %87 ]
  %108 = icmp slt i64 %107, %91
  br i1 %108, label %109, label %112

109:                                              ; preds = %106
  %110 = getelementptr double, ptr %100, i64 %107
  store double 0.000000e+00, ptr %110, align 8
  %111 = add i64 %107, 1
  br label %106

112:                                              ; preds = %106
  %113 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, 0
  %114 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, 1
  %115 = insertvalue { ptr, ptr, i64 } undef, ptr %113, 0
  %116 = insertvalue { ptr, ptr, i64 } %115, ptr %114, 1
  %117 = insertvalue { ptr, ptr, i64 } %116, i64 0, 2
  %118 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, 2
  %119 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, 3, 0
  %120 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, 3, 1
  %121 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, 4, 0
  %122 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, 4, 1
  %123 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %113, 0
  %124 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %123, ptr %114, 1
  %125 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %124, i64 0, 2
  %126 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %125, i64 %62, 3, 0
  %127 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %126, i64 %121, 4, 0
  %128 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %127, i64 %62, 3, 1
  %129 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %128, i64 1, 4, 1
  %130 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %56, 0
  %131 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %56, 1
  %132 = insertvalue { ptr, ptr, i64 } undef, ptr %130, 0
  %133 = insertvalue { ptr, ptr, i64 } %132, ptr %131, 1
  %134 = insertvalue { ptr, ptr, i64 } %133, i64 0, 2
  %135 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %56, 2
  %136 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %56, 3, 0
  %137 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %56, 4, 0
  %138 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %130, 0
  %139 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %138, ptr %131, 1
  %140 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %139, i64 0, 2
  %141 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %140, i64 %62, 3, 0
  %142 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %141, i64 1, 4, 0
  %143 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %70, 0
  %144 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %143, ptr %75, 1
  %145 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %144, i64 0, 2
  %146 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %145, i64 %62, 3, 0
  %147 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %146, i64 1, 4, 0
  br label %148

148:                                              ; preds = %169, %112
  %149 = phi i64 [ %170, %169 ], [ 0, %112 ]
  %150 = icmp slt i64 %149, %62
  br i1 %150, label %151, label %171

151:                                              ; preds = %148
  br label %152

152:                                              ; preds = %155, %151
  %153 = phi i64 [ %168, %155 ], [ 0, %151 ]
  %154 = icmp slt i64 %153, %62
  br i1 %154, label %155, label %169

155:                                              ; preds = %152
  %156 = getelementptr double, ptr %114, i64 0
  %157 = mul i64 %149, %121
  %158 = add i64 %157, %153
  %159 = getelementptr double, ptr %156, i64 %158
  %160 = load double, ptr %159, align 8
  %161 = getelementptr double, ptr %131, i64 %153
  %162 = load double, ptr %161, align 8
  %163 = getelementptr double, ptr %75, i64 %149
  %164 = load double, ptr %163, align 8
  %165 = fmul double %160, %162
  %166 = fadd double %165, %164
  %167 = getelementptr double, ptr %75, i64 %149
  store double %166, ptr %167, align 8
  %168 = add i64 %153, 1
  br label %152

169:                                              ; preds = %152
  %170 = add i64 %149, 1
  br label %148

171:                                              ; preds = %148
  %172 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %70, 0
  %173 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %172, ptr %75, 1
  %174 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %173, i64 0, 2
  %175 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %174, i64 %62, 3, 0
  %176 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %175, i64 1, 4, 0
  %177 = mul i64 %62, 1
  %178 = mul i64 %177, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %179 = getelementptr double, ptr %75, i64 0
  %180 = getelementptr double, ptr %75, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %180, ptr %179, i64 %178, i1 false)
  %181 = getelementptr double, ptr null, i64 %66
  %182 = ptrtoint ptr %181 to i64
  %183 = add i64 %182, 64
  %184 = call ptr @malloc(i64 %183)
  %185 = ptrtoint ptr %184 to i64
  %186 = add i64 %185, 63
  %187 = urem i64 %186, 64
  %188 = sub i64 %186, %187
  %189 = inttoptr i64 %188 to ptr
  %190 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %184, 0
  %191 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %190, ptr %189, 1
  %192 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %191, i64 0, 2
  %193 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %192, i64 %66, 3, 0
  %194 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %193, i64 1, 4, 0
  %195 = mul i64 %66, 1
  %196 = mul i64 %195, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %197 = getelementptr double, ptr %75, i64 0
  %198 = getelementptr double, ptr %189, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %198, ptr %197, i64 %196, i1 false)
  %199 = mul i64 %66, 1
  %200 = mul i64 %199, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %201 = getelementptr double, ptr %189, i64 0
  %202 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %51, 1
  %203 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %51, 2
  %204 = getelementptr double, ptr %202, i64 %203
  call void @llvm.memcpy.p0.p0.i64(ptr %204, ptr %201, i64 %200, i1 false)
  %205 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 0
  %206 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 1
  %207 = insertvalue { ptr, ptr, i64 } undef, ptr %205, 0
  %208 = insertvalue { ptr, ptr, i64 } %207, ptr %206, 1
  %209 = insertvalue { ptr, ptr, i64 } %208, i64 0, 2
  %210 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 2
  %211 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 3, 0
  %212 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 3, 1
  %213 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 4, 0
  %214 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 4, 1
  %215 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %205, 0
  %216 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %215, ptr %206, 1
  %217 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %216, i64 0, 2
  %218 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %217, i64 %62, 3, 0
  %219 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %218, i64 %213, 4, 0
  %220 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %219, i64 %62, 3, 1
  %221 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %220, i64 1, 4, 1
  %222 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %56, 0
  %223 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %56, 1
  %224 = insertvalue { ptr, ptr, i64 } undef, ptr %222, 0
  %225 = insertvalue { ptr, ptr, i64 } %224, ptr %223, 1
  %226 = insertvalue { ptr, ptr, i64 } %225, i64 0, 2
  %227 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %56, 2
  %228 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %56, 3, 0
  %229 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %56, 4, 0
  %230 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %222, 0
  %231 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %230, ptr %223, 1
  %232 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %231, i64 0, 2
  %233 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %232, i64 %62, 3, 0
  %234 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %233, i64 1, 4, 0
  %235 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %95, 0
  %236 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %235, ptr %100, 1
  %237 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %236, i64 0, 2
  %238 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %237, i64 %62, 3, 0
  %239 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %238, i64 1, 4, 0
  br label %240

240:                                              ; preds = %261, %171
  %241 = phi i64 [ %262, %261 ], [ 0, %171 ]
  %242 = icmp slt i64 %241, %62
  br i1 %242, label %243, label %263

243:                                              ; preds = %240
  br label %244

244:                                              ; preds = %247, %243
  %245 = phi i64 [ %260, %247 ], [ 0, %243 ]
  %246 = icmp slt i64 %245, %62
  br i1 %246, label %247, label %261

247:                                              ; preds = %244
  %248 = getelementptr double, ptr %206, i64 0
  %249 = mul i64 %241, %213
  %250 = add i64 %249, %245
  %251 = getelementptr double, ptr %248, i64 %250
  %252 = load double, ptr %251, align 8
  %253 = getelementptr double, ptr %223, i64 %245
  %254 = load double, ptr %253, align 8
  %255 = getelementptr double, ptr %100, i64 %241
  %256 = load double, ptr %255, align 8
  %257 = fmul double %252, %254
  %258 = fadd double %257, %256
  %259 = getelementptr double, ptr %100, i64 %241
  store double %258, ptr %259, align 8
  %260 = add i64 %245, 1
  br label %244

261:                                              ; preds = %244
  %262 = add i64 %241, 1
  br label %240

263:                                              ; preds = %240
  %264 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %95, 0
  %265 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %264, ptr %100, 1
  %266 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %265, i64 0, 2
  %267 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %266, i64 %62, 3, 0
  %268 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %267, i64 1, 4, 0
  %269 = mul i64 %62, 1
  %270 = mul i64 %269, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %271 = getelementptr double, ptr %100, i64 0
  %272 = getelementptr double, ptr %100, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %272, ptr %271, i64 %270, i1 false)
  br label %273

273:                                              ; preds = %276, %263
  %274 = phi i64 [ %285, %276 ], [ 0, %263 ]
  %275 = icmp slt i64 %274, %66
  br i1 %275, label %276, label %286

276:                                              ; preds = %273
  %277 = getelementptr double, ptr %75, i64 %274
  %278 = load double, ptr %277, align 8
  %279 = getelementptr double, ptr %100, i64 %274
  %280 = load double, ptr %279, align 8
  %281 = fmul double %1, %278
  %282 = fmul double %2, %280
  %283 = fadd double %281, %282
  %284 = getelementptr double, ptr %100, i64 %274
  store double %283, ptr %284, align 8
  %285 = add i64 %274, 1
  br label %273

286:                                              ; preds = %273
  %287 = mul i64 %91, 1
  %288 = mul i64 %287, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %289 = getelementptr double, ptr %100, i64 0
  %290 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %61, 1
  %291 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %61, 2
  %292 = getelementptr double, ptr %290, i64 %291
  call void @llvm.memcpy.p0.p0.i64(ptr %292, ptr %289, i64 %288, i1 false)
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #0

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
