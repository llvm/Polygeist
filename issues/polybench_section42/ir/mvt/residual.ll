; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @malloc(i64)

declare void @free(ptr)

define void @kernel_mvt_impl(i32 %0, ptr %1, ptr %2, i64 %3, i64 %4, i64 %5, ptr %6, ptr %7, i64 %8, i64 %9, i64 %10, ptr %11, ptr %12, i64 %13, i64 %14, i64 %15, ptr %16, ptr %17, i64 %18, i64 %19, i64 %20, ptr %21, ptr %22, i64 %23, i64 %24, i64 %25, i64 %26, i64 %27) {
  %29 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %1, 0
  %30 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %29, ptr %2, 1
  %31 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %30, i64 %3, 2
  %32 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %31, i64 %4, 3, 0
  %33 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %32, i64 %5, 4, 0
  %34 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %6, 0
  %35 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %34, ptr %7, 1
  %36 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %35, i64 %8, 2
  %37 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %36, i64 %9, 3, 0
  %38 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %37, i64 %10, 4, 0
  %39 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %11, 0
  %40 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %39, ptr %12, 1
  %41 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %40, i64 %13, 2
  %42 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %41, i64 %14, 3, 0
  %43 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %42, i64 %15, 4, 0
  %44 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %16, 0
  %45 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %44, ptr %17, 1
  %46 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %45, i64 %18, 2
  %47 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %46, i64 %19, 3, 0
  %48 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %47, i64 %20, 4, 0
  %49 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %21, 0
  %50 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %49, ptr %22, 1
  %51 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %50, i64 %23, 2
  %52 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %51, i64 %24, 3, 0
  %53 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %52, i64 %26, 4, 0
  %54 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %53, i64 %25, 3, 1
  %55 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %54, i64 %27, 4, 1
  %56 = sext i32 %0 to i64
  %57 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, 0
  %58 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, 1
  %59 = insertvalue { ptr, ptr, i64 } undef, ptr %57, 0
  %60 = insertvalue { ptr, ptr, i64 } %59, ptr %58, 1
  %61 = insertvalue { ptr, ptr, i64 } %60, i64 0, 2
  %62 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, 2
  %63 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, 3, 0
  %64 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, 3, 1
  %65 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, 4, 0
  %66 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, 4, 1
  %67 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %57, 0
  %68 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %67, ptr %58, 1
  %69 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %68, i64 0, 2
  %70 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %69, i64 %56, 3, 0
  %71 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %70, i64 %65, 4, 0
  %72 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %71, i64 %56, 3, 1
  %73 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %72, i64 1, 4, 1
  %74 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %43, 0
  %75 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %43, 1
  %76 = insertvalue { ptr, ptr, i64 } undef, ptr %74, 0
  %77 = insertvalue { ptr, ptr, i64 } %76, ptr %75, 1
  %78 = insertvalue { ptr, ptr, i64 } %77, i64 0, 2
  %79 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %43, 2
  %80 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %43, 3, 0
  %81 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %43, 4, 0
  %82 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %74, 0
  %83 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %82, ptr %75, 1
  %84 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %83, i64 0, 2
  %85 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %84, i64 %56, 3, 0
  %86 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %85, i64 1, 4, 0
  %87 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 0
  %88 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 1
  %89 = insertvalue { ptr, ptr, i64 } undef, ptr %87, 0
  %90 = insertvalue { ptr, ptr, i64 } %89, ptr %88, 1
  %91 = insertvalue { ptr, ptr, i64 } %90, i64 0, 2
  %92 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 2
  %93 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 3, 0
  %94 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 4, 0
  %95 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %87, 0
  %96 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %95, ptr %88, 1
  %97 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %96, i64 0, 2
  %98 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %97, i64 %56, 3, 0
  %99 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %98, i64 1, 4, 0
  %100 = getelementptr double, ptr null, i64 %56
  %101 = ptrtoint ptr %100 to i64
  %102 = add i64 %101, 64
  %103 = call ptr @malloc(i64 %102)
  %104 = ptrtoint ptr %103 to i64
  %105 = add i64 %104, 63
  %106 = urem i64 %105, 64
  %107 = sub i64 %105, %106
  %108 = inttoptr i64 %107 to ptr
  %109 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %103, 0
  %110 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %109, ptr %108, 1
  %111 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %110, i64 0, 2
  %112 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %111, i64 %56, 3, 0
  %113 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %112, i64 1, 4, 0
  %114 = mul i64 %56, 1
  %115 = mul i64 %114, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %116 = getelementptr double, ptr %88, i64 0
  %117 = getelementptr double, ptr %108, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %117, ptr %116, i64 %115, i1 false)
  br label %118

118:                                              ; preds = %139, %28
  %119 = phi i64 [ %140, %139 ], [ 0, %28 ]
  %120 = icmp slt i64 %119, %56
  br i1 %120, label %121, label %141

121:                                              ; preds = %118
  br label %122

122:                                              ; preds = %125, %121
  %123 = phi i64 [ %138, %125 ], [ 0, %121 ]
  %124 = icmp slt i64 %123, %56
  br i1 %124, label %125, label %139

125:                                              ; preds = %122
  %126 = getelementptr double, ptr %58, i64 0
  %127 = mul i64 %119, %65
  %128 = add i64 %127, %123
  %129 = getelementptr double, ptr %126, i64 %128
  %130 = load double, ptr %129, align 8
  %131 = getelementptr double, ptr %75, i64 %123
  %132 = load double, ptr %131, align 8
  %133 = getelementptr double, ptr %108, i64 %119
  %134 = load double, ptr %133, align 8
  %135 = fmul double %130, %132
  %136 = fadd double %134, %135
  %137 = getelementptr double, ptr %108, i64 %119
  store double %136, ptr %137, align 8
  %138 = add i64 %123, 1
  br label %122

139:                                              ; preds = %122
  %140 = add i64 %119, 1
  br label %118

141:                                              ; preds = %118
  %142 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 3
  %143 = alloca [1 x i64], i64 1, align 8
  store [1 x i64] %142, ptr %143, align 8
  %144 = getelementptr [1 x i64], ptr %143, i32 0, i32 0
  %145 = load i64, ptr %144, align 8
  %146 = getelementptr double, ptr null, i64 %145
  %147 = ptrtoint ptr %146 to i64
  %148 = add i64 %147, 64
  %149 = call ptr @malloc(i64 %148)
  %150 = ptrtoint ptr %149 to i64
  %151 = add i64 %150, 63
  %152 = urem i64 %151, 64
  %153 = sub i64 %151, %152
  %154 = inttoptr i64 %153 to ptr
  %155 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %149, 0
  %156 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %155, ptr %154, 1
  %157 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %156, i64 0, 2
  %158 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %157, i64 %145, 3, 0
  %159 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %158, i64 1, 4, 0
  %160 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 3, 0
  %161 = mul i64 %160, 1
  %162 = mul i64 %161, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %163 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 1
  %164 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 2
  %165 = getelementptr double, ptr %163, i64 %164
  %166 = getelementptr double, ptr %154, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %166, ptr %165, i64 %162, i1 false)
  %167 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %149, 0
  %168 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %167, ptr %154, 1
  %169 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %168, i64 0, 2
  %170 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %169, i64 %56, 3, 0
  %171 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %170, i64 1, 4, 0
  %172 = mul i64 %56, 1
  %173 = mul i64 %172, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %174 = getelementptr double, ptr %108, i64 0
  %175 = getelementptr double, ptr %154, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %175, ptr %174, i64 %173, i1 false)
  %176 = mul i64 %145, 1
  %177 = mul i64 %176, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %178 = getelementptr double, ptr %154, i64 0
  %179 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 1
  %180 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 2
  %181 = getelementptr double, ptr %179, i64 %180
  call void @llvm.memcpy.p0.p0.i64(ptr %181, ptr %178, i64 %177, i1 false)
  %182 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, 0
  %183 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, 1
  %184 = insertvalue { ptr, ptr, i64 } undef, ptr %182, 0
  %185 = insertvalue { ptr, ptr, i64 } %184, ptr %183, 1
  %186 = insertvalue { ptr, ptr, i64 } %185, i64 0, 2
  %187 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, 2
  %188 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, 3, 0
  %189 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, 3, 1
  %190 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, 4, 0
  %191 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, 4, 1
  %192 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %182, 0
  %193 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %192, ptr %183, 1
  %194 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %193, i64 0, 2
  %195 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %194, i64 %56, 3, 0
  %196 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %195, i64 %190, 4, 0
  %197 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %196, i64 %56, 3, 1
  %198 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %197, i64 1, 4, 1
  %199 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %48, 0
  %200 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %48, 1
  %201 = insertvalue { ptr, ptr, i64 } undef, ptr %199, 0
  %202 = insertvalue { ptr, ptr, i64 } %201, ptr %200, 1
  %203 = insertvalue { ptr, ptr, i64 } %202, i64 0, 2
  %204 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %48, 2
  %205 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %48, 3, 0
  %206 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %48, 4, 0
  %207 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %199, 0
  %208 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %207, ptr %200, 1
  %209 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %208, i64 0, 2
  %210 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %209, i64 %56, 3, 0
  %211 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %210, i64 1, 4, 0
  %212 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %38, 0
  %213 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %38, 1
  %214 = insertvalue { ptr, ptr, i64 } undef, ptr %212, 0
  %215 = insertvalue { ptr, ptr, i64 } %214, ptr %213, 1
  %216 = insertvalue { ptr, ptr, i64 } %215, i64 0, 2
  %217 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %38, 2
  %218 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %38, 3, 0
  %219 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %38, 4, 0
  %220 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %212, 0
  %221 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %220, ptr %213, 1
  %222 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %221, i64 0, 2
  %223 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %222, i64 %56, 3, 0
  %224 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %223, i64 1, 4, 0
  %225 = getelementptr double, ptr null, i64 %56
  %226 = ptrtoint ptr %225 to i64
  %227 = add i64 %226, 64
  %228 = call ptr @malloc(i64 %227)
  %229 = ptrtoint ptr %228 to i64
  %230 = add i64 %229, 63
  %231 = urem i64 %230, 64
  %232 = sub i64 %230, %231
  %233 = inttoptr i64 %232 to ptr
  %234 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %228, 0
  %235 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %234, ptr %233, 1
  %236 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %235, i64 0, 2
  %237 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %236, i64 %56, 3, 0
  %238 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %237, i64 1, 4, 0
  %239 = mul i64 %56, 1
  %240 = mul i64 %239, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %241 = getelementptr double, ptr %213, i64 0
  %242 = getelementptr double, ptr %233, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %242, ptr %241, i64 %240, i1 false)
  br label %243

243:                                              ; preds = %264, %141
  %244 = phi i64 [ %265, %264 ], [ 0, %141 ]
  %245 = icmp slt i64 %244, %56
  br i1 %245, label %246, label %266

246:                                              ; preds = %243
  br label %247

247:                                              ; preds = %250, %246
  %248 = phi i64 [ %263, %250 ], [ 0, %246 ]
  %249 = icmp slt i64 %248, %56
  br i1 %249, label %250, label %264

250:                                              ; preds = %247
  %251 = getelementptr double, ptr %183, i64 0
  %252 = mul i64 %248, %190
  %253 = add i64 %252, %244
  %254 = getelementptr double, ptr %251, i64 %253
  %255 = load double, ptr %254, align 8
  %256 = getelementptr double, ptr %200, i64 %248
  %257 = load double, ptr %256, align 8
  %258 = getelementptr double, ptr %233, i64 %244
  %259 = load double, ptr %258, align 8
  %260 = fmul double %255, %257
  %261 = fadd double %259, %260
  %262 = getelementptr double, ptr %233, i64 %244
  store double %261, ptr %262, align 8
  %263 = add i64 %248, 1
  br label %247

264:                                              ; preds = %247
  %265 = add i64 %244, 1
  br label %243

266:                                              ; preds = %243
  %267 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %38, 3
  %268 = alloca [1 x i64], i64 1, align 8
  store [1 x i64] %267, ptr %268, align 8
  %269 = getelementptr [1 x i64], ptr %268, i32 0, i32 0
  %270 = load i64, ptr %269, align 8
  %271 = getelementptr double, ptr null, i64 %270
  %272 = ptrtoint ptr %271 to i64
  %273 = add i64 %272, 64
  %274 = call ptr @malloc(i64 %273)
  %275 = ptrtoint ptr %274 to i64
  %276 = add i64 %275, 63
  %277 = urem i64 %276, 64
  %278 = sub i64 %276, %277
  %279 = inttoptr i64 %278 to ptr
  %280 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %274, 0
  %281 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %280, ptr %279, 1
  %282 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %281, i64 0, 2
  %283 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %282, i64 %270, 3, 0
  %284 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %283, i64 1, 4, 0
  %285 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %38, 3, 0
  %286 = mul i64 %285, 1
  %287 = mul i64 %286, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %288 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %38, 1
  %289 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %38, 2
  %290 = getelementptr double, ptr %288, i64 %289
  %291 = getelementptr double, ptr %279, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %291, ptr %290, i64 %287, i1 false)
  %292 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %274, 0
  %293 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %292, ptr %279, 1
  %294 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %293, i64 0, 2
  %295 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %294, i64 %56, 3, 0
  %296 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %295, i64 1, 4, 0
  %297 = mul i64 %56, 1
  %298 = mul i64 %297, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %299 = getelementptr double, ptr %233, i64 0
  %300 = getelementptr double, ptr %279, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %300, ptr %299, i64 %298, i1 false)
  %301 = mul i64 %270, 1
  %302 = mul i64 %301, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %303 = getelementptr double, ptr %279, i64 0
  %304 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %38, 1
  %305 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %38, 2
  %306 = getelementptr double, ptr %304, i64 %305
  call void @llvm.memcpy.p0.p0.i64(ptr %306, ptr %303, i64 %302, i1 false)
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #0

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
