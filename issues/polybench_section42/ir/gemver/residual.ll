; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @malloc(i64)

declare void @free(ptr)

declare void @memrefCopy(i64, ptr, ptr)

define void @kernel_gemver_impl(i32 %0, double %1, double %2, ptr %3, ptr %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, ptr %10, ptr %11, i64 %12, i64 %13, i64 %14, ptr %15, ptr %16, i64 %17, i64 %18, i64 %19, ptr %20, ptr %21, i64 %22, i64 %23, i64 %24, ptr %25, ptr %26, i64 %27, i64 %28, i64 %29, ptr %30, ptr %31, i64 %32, i64 %33, i64 %34, ptr %35, ptr %36, i64 %37, i64 %38, i64 %39, ptr %40, ptr %41, i64 %42, i64 %43, i64 %44, ptr %45, ptr %46, i64 %47, i64 %48, i64 %49) {
  %51 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %3, 0
  %52 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %51, ptr %4, 1
  %53 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %52, i64 %5, 2
  %54 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %53, i64 %6, 3, 0
  %55 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %54, i64 %8, 4, 0
  %56 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, i64 %7, 3, 1
  %57 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %56, i64 %9, 4, 1
  %58 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %10, 0
  %59 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %58, ptr %11, 1
  %60 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %59, i64 %12, 2
  %61 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %60, i64 %13, 3, 0
  %62 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %61, i64 %14, 4, 0
  %63 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %15, 0
  %64 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %63, ptr %16, 1
  %65 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %64, i64 %17, 2
  %66 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %65, i64 %18, 3, 0
  %67 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %66, i64 %19, 4, 0
  %68 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %20, 0
  %69 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %68, ptr %21, 1
  %70 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %69, i64 %22, 2
  %71 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %70, i64 %23, 3, 0
  %72 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %71, i64 %24, 4, 0
  %73 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %25, 0
  %74 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %73, ptr %26, 1
  %75 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %74, i64 %27, 2
  %76 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %75, i64 %28, 3, 0
  %77 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %76, i64 %29, 4, 0
  %78 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %30, 0
  %79 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %78, ptr %31, 1
  %80 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %79, i64 %32, 2
  %81 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %80, i64 %33, 3, 0
  %82 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %81, i64 %34, 4, 0
  %83 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %35, 0
  %84 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %83, ptr %36, 1
  %85 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %84, i64 %37, 2
  %86 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %85, i64 %38, 3, 0
  %87 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %86, i64 %39, 4, 0
  %88 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %40, 0
  %89 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %88, ptr %41, 1
  %90 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %89, i64 %42, 2
  %91 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %90, i64 %43, 3, 0
  %92 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %91, i64 %44, 4, 0
  %93 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %45, 0
  %94 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %93, ptr %46, 1
  %95 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %94, i64 %47, 2
  %96 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %95, i64 %48, 3, 0
  %97 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %96, i64 %49, 4, 0
  %98 = sext i32 %0 to i64
  %99 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %62, 0
  %100 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %62, 1
  %101 = insertvalue { ptr, ptr, i64 } undef, ptr %99, 0
  %102 = insertvalue { ptr, ptr, i64 } %101, ptr %100, 1
  %103 = insertvalue { ptr, ptr, i64 } %102, i64 0, 2
  %104 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %62, 2
  %105 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %62, 3, 0
  %106 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %62, 4, 0
  %107 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %99, 0
  %108 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %107, ptr %100, 1
  %109 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %108, i64 0, 2
  %110 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %109, i64 %98, 3, 0
  %111 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %110, i64 1, 4, 0
  %112 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %67, 0
  %113 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %67, 1
  %114 = insertvalue { ptr, ptr, i64 } undef, ptr %112, 0
  %115 = insertvalue { ptr, ptr, i64 } %114, ptr %113, 1
  %116 = insertvalue { ptr, ptr, i64 } %115, i64 0, 2
  %117 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %67, 2
  %118 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %67, 3, 0
  %119 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %67, 4, 0
  %120 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %112, 0
  %121 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %120, ptr %113, 1
  %122 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %121, i64 0, 2
  %123 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %122, i64 %98, 3, 0
  %124 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %123, i64 1, 4, 0
  %125 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %72, 0
  %126 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %72, 1
  %127 = insertvalue { ptr, ptr, i64 } undef, ptr %125, 0
  %128 = insertvalue { ptr, ptr, i64 } %127, ptr %126, 1
  %129 = insertvalue { ptr, ptr, i64 } %128, i64 0, 2
  %130 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %72, 2
  %131 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %72, 3, 0
  %132 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %72, 4, 0
  %133 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %125, 0
  %134 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %133, ptr %126, 1
  %135 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %134, i64 0, 2
  %136 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %135, i64 %98, 3, 0
  %137 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %136, i64 1, 4, 0
  %138 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %77, 0
  %139 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %77, 1
  %140 = insertvalue { ptr, ptr, i64 } undef, ptr %138, 0
  %141 = insertvalue { ptr, ptr, i64 } %140, ptr %139, 1
  %142 = insertvalue { ptr, ptr, i64 } %141, i64 0, 2
  %143 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %77, 2
  %144 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %77, 3, 0
  %145 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %77, 4, 0
  %146 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %138, 0
  %147 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %146, ptr %139, 1
  %148 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %147, i64 0, 2
  %149 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %148, i64 %98, 3, 0
  %150 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %149, i64 1, 4, 0
  %151 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %57, 0
  %152 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %57, 1
  %153 = insertvalue { ptr, ptr, i64 } undef, ptr %151, 0
  %154 = insertvalue { ptr, ptr, i64 } %153, ptr %152, 1
  %155 = insertvalue { ptr, ptr, i64 } %154, i64 0, 2
  %156 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %57, 2
  %157 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %57, 3, 0
  %158 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %57, 3, 1
  %159 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %57, 4, 0
  %160 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %57, 4, 1
  %161 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %151, 0
  %162 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %161, ptr %152, 1
  %163 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %162, i64 0, 2
  %164 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %163, i64 %98, 3, 0
  %165 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %164, i64 %159, 4, 0
  %166 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %165, i64 %98, 3, 1
  %167 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %166, i64 1, 4, 1
  %168 = mul i64 %98, %98
  %169 = getelementptr double, ptr null, i64 %168
  %170 = ptrtoint ptr %169 to i64
  %171 = add i64 %170, 64
  %172 = call ptr @malloc(i64 %171)
  %173 = ptrtoint ptr %172 to i64
  %174 = add i64 %173, 63
  %175 = urem i64 %174, 64
  %176 = sub i64 %174, %175
  %177 = inttoptr i64 %176 to ptr
  %178 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %172, 0
  %179 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %178, ptr %177, 1
  %180 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %179, i64 0, 2
  %181 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %180, i64 %98, 3, 0
  %182 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %181, i64 %98, 3, 1
  %183 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %182, i64 %98, 4, 0
  %184 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %183, i64 1, 4, 1
  %185 = call ptr @llvm.stacksave.p0()
  %186 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %167, ptr %186, align 8
  %187 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %186, 1
  %188 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %184, ptr %188, align 8
  %189 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %188, 1
  %190 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %187, ptr %190, align 8
  %191 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %189, ptr %191, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), ptr %190, ptr %191)
  call void @llvm.stackrestore.p0(ptr %185)
  br label %192

192:                                              ; preds = %220, %50
  %193 = phi i64 [ %221, %220 ], [ 0, %50 ]
  %194 = icmp slt i64 %193, %98
  br i1 %194, label %195, label %222

195:                                              ; preds = %192
  br label %196

196:                                              ; preds = %199, %195
  %197 = phi i64 [ %219, %199 ], [ 0, %195 ]
  %198 = icmp slt i64 %197, %98
  br i1 %198, label %199, label %220

199:                                              ; preds = %196
  %200 = getelementptr double, ptr %100, i64 %193
  %201 = load double, ptr %200, align 8
  %202 = getelementptr double, ptr %113, i64 %197
  %203 = load double, ptr %202, align 8
  %204 = getelementptr double, ptr %126, i64 %193
  %205 = load double, ptr %204, align 8
  %206 = getelementptr double, ptr %139, i64 %197
  %207 = load double, ptr %206, align 8
  %208 = mul i64 %193, %98
  %209 = add i64 %208, %197
  %210 = getelementptr double, ptr %177, i64 %209
  %211 = load double, ptr %210, align 8
  %212 = fmul double %201, %203
  %213 = fadd double %211, %212
  %214 = fmul double %205, %207
  %215 = fadd double %213, %214
  %216 = mul i64 %193, %98
  %217 = add i64 %216, %197
  %218 = getelementptr double, ptr %177, i64 %217
  store double %215, ptr %218, align 8
  %219 = add i64 %197, 1
  br label %196

220:                                              ; preds = %196
  %221 = add i64 %193, 1
  br label %192

222:                                              ; preds = %192
  %223 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %57, 3
  %224 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %223, ptr %224, align 8
  %225 = getelementptr [2 x i64], ptr %224, i32 0, i32 0
  %226 = load i64, ptr %225, align 8
  %227 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %57, 3
  %228 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %227, ptr %228, align 8
  %229 = getelementptr [2 x i64], ptr %228, i32 0, i32 1
  %230 = load i64, ptr %229, align 8
  %231 = mul i64 %230, %226
  %232 = getelementptr double, ptr null, i64 %231
  %233 = ptrtoint ptr %232 to i64
  %234 = add i64 %233, 64
  %235 = call ptr @malloc(i64 %234)
  %236 = ptrtoint ptr %235 to i64
  %237 = add i64 %236, 63
  %238 = urem i64 %237, 64
  %239 = sub i64 %237, %238
  %240 = inttoptr i64 %239 to ptr
  %241 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %235, 0
  %242 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %241, ptr %240, 1
  %243 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %242, i64 0, 2
  %244 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %243, i64 %226, 3, 0
  %245 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %244, i64 %230, 3, 1
  %246 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %245, i64 %230, 4, 0
  %247 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %246, i64 1, 4, 1
  %248 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %57, 3, 0
  %249 = mul i64 %248, 1
  %250 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %57, 3, 1
  %251 = mul i64 %249, %250
  %252 = mul i64 %251, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %253 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %57, 1
  %254 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %57, 2
  %255 = getelementptr double, ptr %253, i64 %254
  %256 = getelementptr double, ptr %240, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %256, ptr %255, i64 %252, i1 false)
  %257 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %235, 0
  %258 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %257, ptr %240, 1
  %259 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %258, i64 0, 2
  %260 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %259, i64 %98, 3, 0
  %261 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %260, i64 %230, 4, 0
  %262 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %261, i64 %98, 3, 1
  %263 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %262, i64 1, 4, 1
  %264 = call ptr @llvm.stacksave.p0()
  %265 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %184, ptr %265, align 8
  %266 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %265, 1
  %267 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %263, ptr %267, align 8
  %268 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %267, 1
  %269 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %266, ptr %269, align 8
  %270 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %268, ptr %270, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), ptr %269, ptr %270)
  call void @llvm.stackrestore.p0(ptr %264)
  %271 = mul i64 %226, 1
  %272 = mul i64 %271, %230
  %273 = mul i64 %272, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %274 = getelementptr double, ptr %240, i64 0
  %275 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %57, 1
  %276 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %57, 2
  %277 = getelementptr double, ptr %275, i64 %276
  call void @llvm.memcpy.p0.p0.i64(ptr %277, ptr %274, i64 %273, i1 false)
  %278 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %92, 0
  %279 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %92, 1
  %280 = insertvalue { ptr, ptr, i64 } undef, ptr %278, 0
  %281 = insertvalue { ptr, ptr, i64 } %280, ptr %279, 1
  %282 = insertvalue { ptr, ptr, i64 } %281, i64 0, 2
  %283 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %92, 2
  %284 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %92, 3, 0
  %285 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %92, 4, 0
  %286 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %278, 0
  %287 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %286, ptr %279, 1
  %288 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %287, i64 0, 2
  %289 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %288, i64 %98, 3, 0
  %290 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %289, i64 1, 4, 0
  %291 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %87, 0
  %292 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %87, 1
  %293 = insertvalue { ptr, ptr, i64 } undef, ptr %291, 0
  %294 = insertvalue { ptr, ptr, i64 } %293, ptr %292, 1
  %295 = insertvalue { ptr, ptr, i64 } %294, i64 0, 2
  %296 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %87, 2
  %297 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %87, 3, 0
  %298 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %87, 4, 0
  %299 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %291, 0
  %300 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %299, ptr %292, 1
  %301 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %300, i64 0, 2
  %302 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %301, i64 %98, 3, 0
  %303 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %302, i64 1, 4, 0
  %304 = getelementptr double, ptr null, i64 %98
  %305 = ptrtoint ptr %304 to i64
  %306 = add i64 %305, 64
  %307 = call ptr @malloc(i64 %306)
  %308 = ptrtoint ptr %307 to i64
  %309 = add i64 %308, 63
  %310 = urem i64 %309, 64
  %311 = sub i64 %309, %310
  %312 = inttoptr i64 %311 to ptr
  %313 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %307, 0
  %314 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %313, ptr %312, 1
  %315 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %314, i64 0, 2
  %316 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %315, i64 %98, 3, 0
  %317 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %316, i64 1, 4, 0
  %318 = mul i64 %98, 1
  %319 = mul i64 %318, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %320 = getelementptr double, ptr %292, i64 0
  %321 = getelementptr double, ptr %312, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %321, ptr %320, i64 %319, i1 false)
  br label %322

322:                                              ; preds = %343, %222
  %323 = phi i64 [ %344, %343 ], [ 0, %222 ]
  %324 = icmp slt i64 %323, %98
  br i1 %324, label %325, label %345

325:                                              ; preds = %322
  br label %326

326:                                              ; preds = %329, %325
  %327 = phi i64 [ %342, %329 ], [ 0, %325 ]
  %328 = icmp slt i64 %327, %98
  br i1 %328, label %329, label %343

329:                                              ; preds = %326
  %330 = mul i64 %327, %98
  %331 = add i64 %330, %323
  %332 = getelementptr double, ptr %177, i64 %331
  %333 = load double, ptr %332, align 8
  %334 = getelementptr double, ptr %279, i64 %327
  %335 = load double, ptr %334, align 8
  %336 = getelementptr double, ptr %312, i64 %323
  %337 = load double, ptr %336, align 8
  %338 = fmul double %2, %333
  %339 = fmul double %338, %335
  %340 = fadd double %337, %339
  %341 = getelementptr double, ptr %312, i64 %323
  store double %340, ptr %341, align 8
  %342 = add i64 %327, 1
  br label %326

343:                                              ; preds = %326
  %344 = add i64 %323, 1
  br label %322

345:                                              ; preds = %322
  %346 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %87, 3
  %347 = alloca [1 x i64], i64 1, align 8
  store [1 x i64] %346, ptr %347, align 8
  %348 = getelementptr [1 x i64], ptr %347, i32 0, i32 0
  %349 = load i64, ptr %348, align 8
  %350 = getelementptr double, ptr null, i64 %349
  %351 = ptrtoint ptr %350 to i64
  %352 = add i64 %351, 64
  %353 = call ptr @malloc(i64 %352)
  %354 = ptrtoint ptr %353 to i64
  %355 = add i64 %354, 63
  %356 = urem i64 %355, 64
  %357 = sub i64 %355, %356
  %358 = inttoptr i64 %357 to ptr
  %359 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %353, 0
  %360 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %359, ptr %358, 1
  %361 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %360, i64 0, 2
  %362 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %361, i64 %349, 3, 0
  %363 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %362, i64 1, 4, 0
  %364 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %87, 3, 0
  %365 = mul i64 %364, 1
  %366 = mul i64 %365, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %367 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %87, 1
  %368 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %87, 2
  %369 = getelementptr double, ptr %367, i64 %368
  %370 = getelementptr double, ptr %358, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %370, ptr %369, i64 %366, i1 false)
  %371 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %353, 0
  %372 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %371, ptr %358, 1
  %373 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %372, i64 0, 2
  %374 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %373, i64 %98, 3, 0
  %375 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %374, i64 1, 4, 0
  %376 = mul i64 %98, 1
  %377 = mul i64 %376, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %378 = getelementptr double, ptr %312, i64 0
  %379 = getelementptr double, ptr %358, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %379, ptr %378, i64 %377, i1 false)
  %380 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %97, 3
  %381 = alloca [1 x i64], i64 1, align 8
  store [1 x i64] %380, ptr %381, align 8
  %382 = getelementptr [1 x i64], ptr %381, i32 0, i32 0
  %383 = load i64, ptr %382, align 8
  br label %384

384:                                              ; preds = %387, %345
  %385 = phi i64 [ %395, %387 ], [ 0, %345 ]
  %386 = icmp slt i64 %385, %383
  br i1 %386, label %387, label %396

387:                                              ; preds = %384
  %388 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %97, 1
  %389 = getelementptr double, ptr %388, i64 %385
  %390 = load double, ptr %389, align 8
  %391 = getelementptr double, ptr %358, i64 %385
  %392 = load double, ptr %391, align 8
  %393 = fadd double %392, %390
  %394 = getelementptr double, ptr %358, i64 %385
  store double %393, ptr %394, align 8
  %395 = add i64 %385, 1
  br label %384

396:                                              ; preds = %384
  %397 = getelementptr double, ptr null, i64 %349
  %398 = ptrtoint ptr %397 to i64
  %399 = add i64 %398, 64
  %400 = call ptr @malloc(i64 %399)
  %401 = ptrtoint ptr %400 to i64
  %402 = add i64 %401, 63
  %403 = urem i64 %402, 64
  %404 = sub i64 %402, %403
  %405 = inttoptr i64 %404 to ptr
  %406 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %400, 0
  %407 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %406, ptr %405, 1
  %408 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %407, i64 0, 2
  %409 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %408, i64 %349, 3, 0
  %410 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %409, i64 1, 4, 0
  %411 = mul i64 %349, 1
  %412 = mul i64 %411, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %413 = getelementptr double, ptr %358, i64 0
  %414 = getelementptr double, ptr %405, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %414, ptr %413, i64 %412, i1 false)
  %415 = mul i64 %349, 1
  %416 = mul i64 %415, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %417 = getelementptr double, ptr %405, i64 0
  %418 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %87, 1
  %419 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %87, 2
  %420 = getelementptr double, ptr %418, i64 %419
  call void @llvm.memcpy.p0.p0.i64(ptr %420, ptr %417, i64 %416, i1 false)
  %421 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %353, 0
  %422 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %421, ptr %358, 1
  %423 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %422, i64 0, 2
  %424 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %423, i64 %98, 3, 0
  %425 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %424, i64 1, 4, 0
  %426 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %82, 0
  %427 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %82, 1
  %428 = insertvalue { ptr, ptr, i64 } undef, ptr %426, 0
  %429 = insertvalue { ptr, ptr, i64 } %428, ptr %427, 1
  %430 = insertvalue { ptr, ptr, i64 } %429, i64 0, 2
  %431 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %82, 2
  %432 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %82, 3, 0
  %433 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %82, 4, 0
  %434 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %426, 0
  %435 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %434, ptr %427, 1
  %436 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %435, i64 0, 2
  %437 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %436, i64 %98, 3, 0
  %438 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %437, i64 1, 4, 0
  %439 = getelementptr double, ptr null, i64 %98
  %440 = ptrtoint ptr %439 to i64
  %441 = add i64 %440, 64
  %442 = call ptr @malloc(i64 %441)
  %443 = ptrtoint ptr %442 to i64
  %444 = add i64 %443, 63
  %445 = urem i64 %444, 64
  %446 = sub i64 %444, %445
  %447 = inttoptr i64 %446 to ptr
  %448 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %442, 0
  %449 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %448, ptr %447, 1
  %450 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %449, i64 0, 2
  %451 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %450, i64 %98, 3, 0
  %452 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %451, i64 1, 4, 0
  %453 = mul i64 %98, 1
  %454 = mul i64 %453, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %455 = getelementptr double, ptr %427, i64 0
  %456 = getelementptr double, ptr %447, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %456, ptr %455, i64 %454, i1 false)
  br label %457

457:                                              ; preds = %478, %396
  %458 = phi i64 [ %479, %478 ], [ 0, %396 ]
  %459 = icmp slt i64 %458, %98
  br i1 %459, label %460, label %480

460:                                              ; preds = %457
  br label %461

461:                                              ; preds = %464, %460
  %462 = phi i64 [ %477, %464 ], [ 0, %460 ]
  %463 = icmp slt i64 %462, %98
  br i1 %463, label %464, label %478

464:                                              ; preds = %461
  %465 = mul i64 %458, %98
  %466 = add i64 %465, %462
  %467 = getelementptr double, ptr %177, i64 %466
  %468 = load double, ptr %467, align 8
  %469 = getelementptr double, ptr %358, i64 %462
  %470 = load double, ptr %469, align 8
  %471 = getelementptr double, ptr %447, i64 %458
  %472 = load double, ptr %471, align 8
  %473 = fmul double %1, %468
  %474 = fmul double %473, %470
  %475 = fadd double %472, %474
  %476 = getelementptr double, ptr %447, i64 %458
  store double %475, ptr %476, align 8
  %477 = add i64 %462, 1
  br label %461

478:                                              ; preds = %461
  %479 = add i64 %458, 1
  br label %457

480:                                              ; preds = %457
  %481 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %82, 3
  %482 = alloca [1 x i64], i64 1, align 8
  store [1 x i64] %481, ptr %482, align 8
  %483 = getelementptr [1 x i64], ptr %482, i32 0, i32 0
  %484 = load i64, ptr %483, align 8
  %485 = getelementptr double, ptr null, i64 %484
  %486 = ptrtoint ptr %485 to i64
  %487 = add i64 %486, 64
  %488 = call ptr @malloc(i64 %487)
  %489 = ptrtoint ptr %488 to i64
  %490 = add i64 %489, 63
  %491 = urem i64 %490, 64
  %492 = sub i64 %490, %491
  %493 = inttoptr i64 %492 to ptr
  %494 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %488, 0
  %495 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %494, ptr %493, 1
  %496 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %495, i64 0, 2
  %497 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %496, i64 %484, 3, 0
  %498 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %497, i64 1, 4, 0
  %499 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %82, 3, 0
  %500 = mul i64 %499, 1
  %501 = mul i64 %500, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %502 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %82, 1
  %503 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %82, 2
  %504 = getelementptr double, ptr %502, i64 %503
  %505 = getelementptr double, ptr %493, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %505, ptr %504, i64 %501, i1 false)
  %506 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %488, 0
  %507 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %506, ptr %493, 1
  %508 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %507, i64 0, 2
  %509 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %508, i64 %98, 3, 0
  %510 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %509, i64 1, 4, 0
  %511 = mul i64 %98, 1
  %512 = mul i64 %511, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %513 = getelementptr double, ptr %447, i64 0
  %514 = getelementptr double, ptr %493, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %514, ptr %513, i64 %512, i1 false)
  %515 = mul i64 %484, 1
  %516 = mul i64 %515, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %517 = getelementptr double, ptr %493, i64 0
  %518 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %82, 1
  %519 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %82, 2
  %520 = getelementptr double, ptr %518, i64 %519
  call void @llvm.memcpy.p0.p0.i64(ptr %520, ptr %517, i64 %516, i1 false)
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
