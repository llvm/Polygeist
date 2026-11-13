; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @malloc(i64)

declare void @free(ptr)

declare void @memrefCopy(i64, ptr, ptr)

define void @kernel_gramschmidt_impl(i32 %0, i32 %1, ptr %2, ptr %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr %9, ptr %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, ptr %16, ptr %17, i64 %18, i64 %19, i64 %20, i64 %21, i64 %22) {
  %24 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %2, 0
  %25 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, ptr %3, 1
  %26 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, i64 %4, 2
  %27 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %26, i64 %5, 3, 0
  %28 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, i64 %7, 4, 0
  %29 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, i64 %6, 3, 1
  %30 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, i64 %8, 4, 1
  %31 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %9, 0
  %32 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, ptr %10, 1
  %33 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, i64 %11, 2
  %34 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %33, i64 %12, 3, 0
  %35 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, i64 %14, 4, 0
  %36 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, i64 %13, 3, 1
  %37 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, i64 %15, 4, 1
  %38 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %16, 0
  %39 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, ptr %17, 1
  %40 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, i64 %18, 2
  %41 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, i64 %19, 3, 0
  %42 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, i64 %21, 4, 0
  %43 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, i64 %20, 3, 1
  %44 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %43, i64 %22, 4, 1
  %45 = sext i32 %1 to i64
  %46 = sext i32 %0 to i64
  %47 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, 3
  %48 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %47, ptr %48, align 8
  %49 = getelementptr [2 x i64], ptr %48, i32 0, i32 0
  %50 = load i64, ptr %49, align 8
  %51 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, 3
  %52 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %51, ptr %52, align 8
  %53 = getelementptr [2 x i64], ptr %52, i32 0, i32 1
  %54 = load i64, ptr %53, align 8
  %55 = mul i64 %54, %50
  %56 = getelementptr double, ptr null, i64 %55
  %57 = ptrtoint ptr %56 to i64
  %58 = add i64 %57, 64
  %59 = call ptr @malloc(i64 %58)
  %60 = ptrtoint ptr %59 to i64
  %61 = add i64 %60, 63
  %62 = urem i64 %61, 64
  %63 = sub i64 %61, %62
  %64 = inttoptr i64 %63 to ptr
  %65 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %59, 0
  %66 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %65, ptr %64, 1
  %67 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %66, i64 0, 2
  %68 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %67, i64 %50, 3, 0
  %69 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %68, i64 %54, 3, 1
  %70 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %69, i64 %54, 4, 0
  %71 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %70, i64 1, 4, 1
  %72 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, 3, 0
  %73 = mul i64 %72, 1
  %74 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, 3, 1
  %75 = mul i64 %73, %74
  %76 = mul i64 %75, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %77 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, 1
  %78 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, 2
  %79 = getelementptr double, ptr %77, i64 %78
  %80 = getelementptr double, ptr %64, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %80, ptr %79, i64 %76, i1 false)
  %81 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, 3
  %82 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %81, ptr %82, align 8
  %83 = getelementptr [2 x i64], ptr %82, i32 0, i32 0
  %84 = load i64, ptr %83, align 8
  %85 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, 3
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
  %106 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, 3, 0
  %107 = mul i64 %106, 1
  %108 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, 3, 1
  %109 = mul i64 %107, %108
  %110 = mul i64 %109, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %111 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, 1
  %112 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, 2
  %113 = getelementptr double, ptr %111, i64 %112
  %114 = getelementptr double, ptr %98, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %114, ptr %113, i64 %110, i1 false)
  %115 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, 3
  %116 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %115, ptr %116, align 8
  %117 = getelementptr [2 x i64], ptr %116, i32 0, i32 0
  %118 = load i64, ptr %117, align 8
  %119 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, 3
  %120 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %119, ptr %120, align 8
  %121 = getelementptr [2 x i64], ptr %120, i32 0, i32 1
  %122 = load i64, ptr %121, align 8
  %123 = mul i64 %122, %118
  %124 = getelementptr double, ptr null, i64 %123
  %125 = ptrtoint ptr %124 to i64
  %126 = add i64 %125, 64
  %127 = call ptr @malloc(i64 %126)
  %128 = ptrtoint ptr %127 to i64
  %129 = add i64 %128, 63
  %130 = urem i64 %129, 64
  %131 = sub i64 %129, %130
  %132 = inttoptr i64 %131 to ptr
  %133 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %127, 0
  %134 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %133, ptr %132, 1
  %135 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %134, i64 0, 2
  %136 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %135, i64 %118, 3, 0
  %137 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %136, i64 %122, 3, 1
  %138 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %137, i64 %122, 4, 0
  %139 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %138, i64 1, 4, 1
  %140 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, 3, 0
  %141 = mul i64 %140, 1
  %142 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, 3, 1
  %143 = mul i64 %141, %142
  %144 = mul i64 %143, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %145 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, 1
  %146 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, 2
  %147 = getelementptr double, ptr %145, i64 %146
  %148 = getelementptr double, ptr %132, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %148, ptr %147, i64 %144, i1 false)
  br label %149

149:                                              ; preds = %433, %23
  %150 = phi i64 [ %458, %433 ], [ 0, %23 ]
  %151 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %151, %433 ], [ %71, %23 ]
  %152 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %152, %433 ], [ %105, %23 ]
  %153 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %153, %433 ], [ %139, %23 ]
  %154 = icmp slt i64 %150, %45
  br i1 %154, label %155, label %459

155:                                              ; preds = %149
  %156 = alloca double, i64 1, align 8
  %157 = insertvalue { ptr, ptr, i64 } undef, ptr %156, 0
  %158 = insertvalue { ptr, ptr, i64 } %157, ptr %156, 1
  %159 = insertvalue { ptr, ptr, i64 } %158, i64 0, 2
  %160 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), i64 64))
  %161 = ptrtoint ptr %160 to i64
  %162 = add i64 %161, 63
  %163 = urem i64 %162, 64
  %164 = sub i64 %162, %163
  %165 = inttoptr i64 %164 to ptr
  %166 = insertvalue { ptr, ptr, i64 } undef, ptr %160, 0
  %167 = insertvalue { ptr, ptr, i64 } %166, ptr %165, 1
  %168 = insertvalue { ptr, ptr, i64 } %167, i64 0, 2
  %169 = getelementptr double, ptr %156, i64 0
  %170 = getelementptr double, ptr %165, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %170, ptr %169, i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), i1 false)
  store double 0.000000e+00, ptr %165, align 8
  %171 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %151, 0
  %172 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %151, 1
  %173 = insertvalue { ptr, ptr, i64 } undef, ptr %171, 0
  %174 = insertvalue { ptr, ptr, i64 } %173, ptr %172, 1
  %175 = insertvalue { ptr, ptr, i64 } %174, i64 0, 2
  %176 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %151, 2
  %177 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %151, 3, 0
  %178 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %151, 3, 1
  %179 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %151, 4, 0
  %180 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %151, 4, 1
  %181 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %171, 0
  %182 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %181, ptr %172, 1
  %183 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %182, i64 %150, 2
  %184 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %183, i64 %46, 3, 0
  %185 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %184, i64 %179, 4, 0
  br label %186

186:                                              ; preds = %189, %155
  %187 = phi i64 [ %197, %189 ], [ 0, %155 ]
  %188 = icmp slt i64 %187, %46
  br i1 %188, label %189, label %198

189:                                              ; preds = %186
  %190 = getelementptr double, ptr %172, i64 %150
  %191 = mul i64 %187, %179
  %192 = getelementptr double, ptr %190, i64 %191
  %193 = load double, ptr %192, align 8
  %194 = load double, ptr %165, align 8
  %195 = fmul double %193, %193
  %196 = fadd double %194, %195
  store double %196, ptr %165, align 8
  %197 = add i64 %187, 1
  br label %186

198:                                              ; preds = %186
  %199 = load double, ptr %165, align 8
  %200 = call double @llvm.sqrt.f64(double %199)
  %201 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %152, 1
  %202 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %152, 4, 0
  %203 = mul i64 %150, %202
  %204 = add i64 %203, %150
  %205 = getelementptr double, ptr %201, i64 %204
  store double %200, ptr %205, align 8
  %206 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %153, 0
  %207 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %153, 1
  %208 = insertvalue { ptr, ptr, i64 } undef, ptr %206, 0
  %209 = insertvalue { ptr, ptr, i64 } %208, ptr %207, 1
  %210 = insertvalue { ptr, ptr, i64 } %209, i64 0, 2
  %211 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %153, 2
  %212 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %153, 3, 0
  %213 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %153, 3, 1
  %214 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %153, 4, 0
  %215 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %153, 4, 1
  %216 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %206, 0
  %217 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %216, ptr %207, 1
  %218 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %217, i64 %150, 2
  %219 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %218, i64 %46, 3, 0
  %220 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %219, i64 %214, 4, 0
  %221 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, 0
  %222 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, 1
  %223 = insertvalue { ptr, ptr, i64 } undef, ptr %221, 0
  %224 = insertvalue { ptr, ptr, i64 } %223, ptr %222, 1
  %225 = insertvalue { ptr, ptr, i64 } %224, i64 0, 2
  %226 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, 2
  %227 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, 3, 0
  %228 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, 3, 1
  %229 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, 4, 0
  %230 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, 4, 1
  %231 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %221, 0
  %232 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %231, ptr %222, 1
  %233 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %232, i64 %150, 2
  %234 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %233, i64 %46, 3, 0
  %235 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %234, i64 %229, 4, 0
  %236 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, 0
  %237 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, 1
  %238 = insertvalue { ptr, ptr, i64 } undef, ptr %236, 0
  %239 = insertvalue { ptr, ptr, i64 } %238, ptr %237, 1
  %240 = insertvalue { ptr, ptr, i64 } %239, i64 0, 2
  %241 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, 2
  %242 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, 3, 0
  %243 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, 3, 1
  %244 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, 4, 0
  %245 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, 4, 1
  %246 = mul i64 %150, %244
  %247 = add i64 %150, %246
  %248 = insertvalue { ptr, ptr, i64 } undef, ptr %236, 0
  %249 = insertvalue { ptr, ptr, i64 } %248, ptr %237, 1
  %250 = insertvalue { ptr, ptr, i64 } %249, i64 %247, 2
  br label %251

251:                                              ; preds = %254, %198
  %252 = phi i64 [ %265, %254 ], [ 0, %198 ]
  %253 = icmp slt i64 %252, %46
  br i1 %253, label %254, label %266

254:                                              ; preds = %251
  %255 = getelementptr double, ptr %222, i64 %150
  %256 = mul i64 %252, %229
  %257 = getelementptr double, ptr %255, i64 %256
  %258 = load double, ptr %257, align 8
  %259 = getelementptr double, ptr %237, i64 %247
  %260 = load double, ptr %259, align 8
  %261 = fdiv double %258, %260
  %262 = getelementptr double, ptr %207, i64 %150
  %263 = mul i64 %252, %214
  %264 = getelementptr double, ptr %262, i64 %263
  store double %261, ptr %264, align 8
  %265 = add i64 %252, 1
  br label %251

266:                                              ; preds = %251
  %267 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %153, 0
  %268 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %153, 1
  %269 = insertvalue { ptr, ptr, i64 } undef, ptr %267, 0
  %270 = insertvalue { ptr, ptr, i64 } %269, ptr %268, 1
  %271 = insertvalue { ptr, ptr, i64 } %270, i64 0, 2
  %272 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %153, 2
  %273 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %153, 3, 0
  %274 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %153, 3, 1
  %275 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %153, 4, 0
  %276 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %153, 4, 1
  %277 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %267, 0
  %278 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %277, ptr %268, 1
  %279 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %278, i64 %150, 2
  %280 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %279, i64 %46, 3, 0
  %281 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %280, i64 %275, 4, 0
  %282 = call ptr @llvm.stacksave.p0()
  %283 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %220, ptr %283, align 8
  %284 = insertvalue { i64, ptr } { i64 1, ptr undef }, ptr %283, 1
  %285 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %281, ptr %285, align 8
  %286 = insertvalue { i64, ptr } { i64 1, ptr undef }, ptr %285, 1
  %287 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %284, ptr %287, align 8
  %288 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %286, ptr %288, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), ptr %287, ptr %288)
  call void @llvm.stackrestore.p0(ptr %282)
  %289 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %152, 0
  %290 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %152, 1
  %291 = insertvalue { ptr, ptr, i64 } undef, ptr %289, 0
  %292 = insertvalue { ptr, ptr, i64 } %291, ptr %290, 1
  %293 = insertvalue { ptr, ptr, i64 } %292, i64 0, 2
  %294 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %152, 2
  %295 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %152, 3, 0
  %296 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %152, 3, 1
  %297 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %152, 4, 0
  %298 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %152, 4, 1
  %299 = mul i64 %150, %297
  %300 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %289, 0
  %301 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %300, ptr %290, 1
  %302 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %301, i64 %299, 2
  %303 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %302, i64 %45, 3, 0
  %304 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %303, i64 1, 4, 0
  br label %305

305:                                              ; preds = %308, %266
  %306 = phi i64 [ %317, %308 ], [ 0, %266 ]
  %307 = icmp slt i64 %306, %45
  br i1 %307, label %308, label %318

308:                                              ; preds = %305
  %309 = getelementptr double, ptr %290, i64 %299
  %310 = getelementptr double, ptr %309, i64 %306
  %311 = load double, ptr %310, align 8
  %312 = add i64 %150, 1
  %313 = icmp sge i64 %306, %312
  %314 = select i1 %313, double 0.000000e+00, double %311
  %315 = getelementptr double, ptr %290, i64 %299
  %316 = getelementptr double, ptr %315, i64 %306
  store double %314, ptr %316, align 8
  %317 = add i64 %306, 1
  br label %305

318:                                              ; preds = %305
  %319 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, 0
  %320 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, 1
  %321 = insertvalue { ptr, ptr, i64 } undef, ptr %319, 0
  %322 = insertvalue { ptr, ptr, i64 } %321, ptr %320, 1
  %323 = insertvalue { ptr, ptr, i64 } %322, i64 0, 2
  %324 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, 2
  %325 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, 3, 0
  %326 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, 3, 1
  %327 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, 4, 0
  %328 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, 4, 1
  %329 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %319, 0
  %330 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %329, ptr %320, 1
  %331 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %330, i64 0, 2
  %332 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %331, i64 %46, 3, 0
  %333 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %332, i64 %327, 4, 0
  %334 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %333, i64 %45, 3, 1
  %335 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %334, i64 1, 4, 1
  br label %336

336:                                              ; preds = %361, %318
  %337 = phi i64 [ %362, %361 ], [ 0, %318 ]
  %338 = icmp slt i64 %337, %45
  br i1 %338, label %339, label %363

339:                                              ; preds = %336
  br label %340

340:                                              ; preds = %343, %339
  %341 = phi i64 [ %360, %343 ], [ 0, %339 ]
  %342 = icmp slt i64 %341, %46
  br i1 %342, label %343, label %361

343:                                              ; preds = %340
  %344 = getelementptr double, ptr %207, i64 %150
  %345 = mul i64 %341, %214
  %346 = getelementptr double, ptr %344, i64 %345
  %347 = load double, ptr %346, align 8
  %348 = getelementptr double, ptr %320, i64 0
  %349 = mul i64 %341, %327
  %350 = add i64 %349, %337
  %351 = getelementptr double, ptr %348, i64 %350
  %352 = load double, ptr %351, align 8
  %353 = getelementptr double, ptr %290, i64 %299
  %354 = getelementptr double, ptr %353, i64 %337
  %355 = load double, ptr %354, align 8
  %356 = fmul double %347, %352
  %357 = fadd double %355, %356
  %358 = getelementptr double, ptr %290, i64 %299
  %359 = getelementptr double, ptr %358, i64 %337
  store double %357, ptr %359, align 8
  %360 = add i64 %341, 1
  br label %340

361:                                              ; preds = %340
  %362 = add i64 %337, 1
  br label %336

363:                                              ; preds = %336
  %364 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %152, 0
  %365 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %152, 1
  %366 = insertvalue { ptr, ptr, i64 } undef, ptr %364, 0
  %367 = insertvalue { ptr, ptr, i64 } %366, ptr %365, 1
  %368 = insertvalue { ptr, ptr, i64 } %367, i64 0, 2
  %369 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %152, 2
  %370 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %152, 3, 0
  %371 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %152, 3, 1
  %372 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %152, 4, 0
  %373 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %152, 4, 1
  %374 = mul i64 %150, %372
  %375 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %364, 0
  %376 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %375, ptr %365, 1
  %377 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %376, i64 %374, 2
  %378 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %377, i64 %45, 3, 0
  %379 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %378, i64 1, 4, 0
  %380 = call ptr @llvm.stacksave.p0()
  %381 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %304, ptr %381, align 8
  %382 = insertvalue { i64, ptr } { i64 1, ptr undef }, ptr %381, 1
  %383 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %379, ptr %383, align 8
  %384 = insertvalue { i64, ptr } { i64 1, ptr undef }, ptr %383, 1
  %385 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %382, ptr %385, align 8
  %386 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %384, ptr %386, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), ptr %385, ptr %386)
  call void @llvm.stackrestore.p0(ptr %380)
  %387 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %151, 0
  %388 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %151, 1
  %389 = insertvalue { ptr, ptr, i64 } undef, ptr %387, 0
  %390 = insertvalue { ptr, ptr, i64 } %389, ptr %388, 1
  %391 = insertvalue { ptr, ptr, i64 } %390, i64 0, 2
  %392 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %151, 2
  %393 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %151, 3, 0
  %394 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %151, 3, 1
  %395 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %151, 4, 0
  %396 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %151, 4, 1
  %397 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %387, 0
  %398 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %397, ptr %388, 1
  %399 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %398, i64 0, 2
  %400 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %399, i64 %46, 3, 0
  %401 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %400, i64 %395, 4, 0
  %402 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %401, i64 %45, 3, 1
  %403 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %402, i64 1, 4, 1
  br label %404

404:                                              ; preds = %431, %363
  %405 = phi i64 [ %432, %431 ], [ 0, %363 ]
  %406 = icmp slt i64 %405, %45
  br i1 %406, label %407, label %433

407:                                              ; preds = %404
  br label %408

408:                                              ; preds = %411, %407
  %409 = phi i64 [ %430, %411 ], [ 0, %407 ]
  %410 = icmp slt i64 %409, %46
  br i1 %410, label %411, label %431

411:                                              ; preds = %408
  %412 = getelementptr double, ptr %207, i64 %150
  %413 = mul i64 %409, %214
  %414 = getelementptr double, ptr %412, i64 %413
  %415 = load double, ptr %414, align 8
  %416 = getelementptr double, ptr %290, i64 %299
  %417 = getelementptr double, ptr %416, i64 %405
  %418 = load double, ptr %417, align 8
  %419 = getelementptr double, ptr %388, i64 0
  %420 = mul i64 %409, %395
  %421 = add i64 %420, %405
  %422 = getelementptr double, ptr %419, i64 %421
  %423 = load double, ptr %422, align 8
  %424 = fmul double %415, %418
  %425 = fsub double %423, %424
  %426 = getelementptr double, ptr %388, i64 0
  %427 = mul i64 %409, %395
  %428 = add i64 %427, %405
  %429 = getelementptr double, ptr %426, i64 %428
  store double %425, ptr %429, align 8
  %430 = add i64 %409, 1
  br label %408

431:                                              ; preds = %408
  %432 = add i64 %405, 1
  br label %404

433:                                              ; preds = %404
  %434 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %151, 0
  %435 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %151, 1
  %436 = insertvalue { ptr, ptr, i64 } undef, ptr %434, 0
  %437 = insertvalue { ptr, ptr, i64 } %436, ptr %435, 1
  %438 = insertvalue { ptr, ptr, i64 } %437, i64 0, 2
  %439 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %151, 2
  %440 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %151, 3, 0
  %441 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %151, 3, 1
  %442 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %151, 4, 0
  %443 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %151, 4, 1
  %444 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %434, 0
  %445 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %444, ptr %435, 1
  %446 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %445, i64 0, 2
  %447 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %446, i64 %46, 3, 0
  %448 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %447, i64 %442, 4, 0
  %449 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %448, i64 %45, 3, 1
  %450 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %449, i64 1, 4, 1
  %451 = call ptr @llvm.stacksave.p0()
  %452 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %403, ptr %452, align 8
  %453 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %452, 1
  %454 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %450, ptr %454, align 8
  %455 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %454, 1
  %456 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %453, ptr %456, align 8
  %457 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %455, ptr %457, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), ptr %456, ptr %457)
  call void @llvm.stackrestore.p0(ptr %451)
  %458 = add i64 %150, 1
  br label %149

459:                                              ; preds = %149
  %460 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %153, 3, 0
  %461 = mul i64 %460, 1
  %462 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %153, 3, 1
  %463 = mul i64 %461, %462
  %464 = mul i64 %463, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %465 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %153, 1
  %466 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %153, 2
  %467 = getelementptr double, ptr %465, i64 %466
  %468 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, 1
  %469 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, 2
  %470 = getelementptr double, ptr %468, i64 %469
  call void @llvm.memcpy.p0.p0.i64(ptr %470, ptr %467, i64 %464, i1 false)
  %471 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %152, 3, 0
  %472 = mul i64 %471, 1
  %473 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %152, 3, 1
  %474 = mul i64 %472, %473
  %475 = mul i64 %474, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %476 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %152, 1
  %477 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %152, 2
  %478 = getelementptr double, ptr %476, i64 %477
  %479 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, 1
  %480 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, 2
  %481 = getelementptr double, ptr %479, i64 %480
  call void @llvm.memcpy.p0.p0.i64(ptr %481, ptr %478, i64 %475, i1 false)
  %482 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %151, 3, 0
  %483 = mul i64 %482, 1
  %484 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %151, 3, 1
  %485 = mul i64 %483, %484
  %486 = mul i64 %485, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %487 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %151, 1
  %488 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %151, 2
  %489 = getelementptr double, ptr %487, i64 %488
  %490 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, 1
  %491 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, 2
  %492 = getelementptr double, ptr %490, i64 %491
  call void @llvm.memcpy.p0.p0.i64(ptr %492, ptr %489, i64 %486, i1 false)
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
