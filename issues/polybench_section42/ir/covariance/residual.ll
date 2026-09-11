; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @malloc(i64)

declare void @free(ptr)

declare void @memrefCopy(i64, ptr, ptr)

define void @kernel_covariance_impl(i32 %0, i32 %1, double %2, ptr %3, ptr %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, ptr %10, ptr %11, i64 %12, i64 %13, i64 %14, i64 %15, i64 %16, ptr %17, ptr %18, i64 %19, i64 %20, i64 %21) {
  %23 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %3, 0
  %24 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, ptr %4, 1
  %25 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, i64 %5, 2
  %26 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, i64 %6, 3, 0
  %27 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %26, i64 %8, 4, 0
  %28 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, i64 %7, 3, 1
  %29 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, i64 %9, 4, 1
  %30 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %10, 0
  %31 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, ptr %11, 1
  %32 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, i64 %12, 2
  %33 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, i64 %13, 3, 0
  %34 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %33, i64 %15, 4, 0
  %35 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, i64 %14, 3, 1
  %36 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, i64 %16, 4, 1
  %37 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %17, 0
  %38 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %37, ptr %18, 1
  %39 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %38, i64 %19, 2
  %40 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %39, i64 %20, 3, 0
  %41 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %40, i64 %21, 4, 0
  %42 = sext i32 %1 to i64
  %43 = sext i32 %0 to i64
  %44 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %41, 3
  %45 = alloca [1 x i64], i64 1, align 8
  store [1 x i64] %44, ptr %45, align 8
  %46 = getelementptr [1 x i64], ptr %45, i32 0, i32 0
  %47 = load i64, ptr %46, align 8
  %48 = getelementptr double, ptr null, i64 %47
  %49 = ptrtoint ptr %48 to i64
  %50 = add i64 %49, 64
  %51 = call ptr @malloc(i64 %50)
  %52 = ptrtoint ptr %51 to i64
  %53 = add i64 %52, 63
  %54 = urem i64 %53, 64
  %55 = sub i64 %53, %54
  %56 = inttoptr i64 %55 to ptr
  %57 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %51, 0
  %58 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %57, ptr %56, 1
  %59 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %58, i64 0, 2
  %60 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %59, i64 %47, 3, 0
  %61 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %60, i64 1, 4, 0
  br label %62

62:                                               ; preds = %65, %22
  %63 = phi i64 [ %67, %65 ], [ 0, %22 ]
  %64 = icmp slt i64 %63, %47
  br i1 %64, label %65, label %68

65:                                               ; preds = %62
  %66 = getelementptr double, ptr %56, i64 %63
  store double 0.000000e+00, ptr %66, align 8
  %67 = add i64 %63, 1
  br label %62

68:                                               ; preds = %62
  %69 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, 3
  %70 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %69, ptr %70, align 8
  %71 = getelementptr [2 x i64], ptr %70, i32 0, i32 0
  %72 = load i64, ptr %71, align 8
  %73 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, 3
  %74 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %73, ptr %74, align 8
  %75 = getelementptr [2 x i64], ptr %74, i32 0, i32 1
  %76 = load i64, ptr %75, align 8
  br label %77

77:                                               ; preds = %96, %68
  %78 = phi i64 [ %97, %96 ], [ 0, %68 ]
  %79 = icmp slt i64 %78, %76
  br i1 %79, label %80, label %98

80:                                               ; preds = %77
  br label %81

81:                                               ; preds = %84, %80
  %82 = phi i64 [ %95, %84 ], [ 0, %80 ]
  %83 = icmp slt i64 %82, %72
  br i1 %83, label %84, label %96

84:                                               ; preds = %81
  %85 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, 1
  %86 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, 4, 0
  %87 = mul i64 %82, %86
  %88 = add i64 %87, %78
  %89 = getelementptr double, ptr %85, i64 %88
  %90 = load double, ptr %89, align 8
  %91 = getelementptr double, ptr %56, i64 %78
  %92 = load double, ptr %91, align 8
  %93 = fadd double %92, %90
  %94 = getelementptr double, ptr %56, i64 %78
  store double %93, ptr %94, align 8
  %95 = add i64 %82, 1
  br label %81

96:                                               ; preds = %81
  %97 = add i64 %78, 1
  br label %77

98:                                               ; preds = %77
  br label %99

99:                                               ; preds = %102, %98
  %100 = phi i64 [ %107, %102 ], [ 0, %98 ]
  %101 = icmp slt i64 %100, %47
  br i1 %101, label %102, label %108

102:                                              ; preds = %99
  %103 = getelementptr double, ptr %56, i64 %100
  %104 = load double, ptr %103, align 8
  %105 = fdiv double %104, %2
  %106 = getelementptr double, ptr %56, i64 %100
  store double %105, ptr %106, align 8
  %107 = add i64 %100, 1
  br label %99

108:                                              ; preds = %99
  %109 = getelementptr double, ptr null, i64 %47
  %110 = ptrtoint ptr %109 to i64
  %111 = add i64 %110, 64
  %112 = call ptr @malloc(i64 %111)
  %113 = ptrtoint ptr %112 to i64
  %114 = add i64 %113, 63
  %115 = urem i64 %114, 64
  %116 = sub i64 %114, %115
  %117 = inttoptr i64 %116 to ptr
  %118 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %112, 0
  %119 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %118, ptr %117, 1
  %120 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %119, i64 0, 2
  %121 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %120, i64 %47, 3, 0
  %122 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %121, i64 1, 4, 0
  %123 = mul i64 %47, 1
  %124 = mul i64 %123, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %125 = getelementptr double, ptr %56, i64 0
  %126 = getelementptr double, ptr %117, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %126, ptr %125, i64 %124, i1 false)
  %127 = mul i64 %47, 1
  %128 = mul i64 %127, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %129 = getelementptr double, ptr %117, i64 0
  %130 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %41, 1
  %131 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %41, 2
  %132 = getelementptr double, ptr %130, i64 %131
  call void @llvm.memcpy.p0.p0.i64(ptr %132, ptr %129, i64 %128, i1 false)
  %133 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %51, 0
  %134 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %133, ptr %56, 1
  %135 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %134, i64 0, 2
  %136 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %135, i64 %43, 3, 0
  %137 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %136, i64 1, 4, 0
  %138 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, 0
  %139 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, 1
  %140 = insertvalue { ptr, ptr, i64 } undef, ptr %138, 0
  %141 = insertvalue { ptr, ptr, i64 } %140, ptr %139, 1
  %142 = insertvalue { ptr, ptr, i64 } %141, i64 0, 2
  %143 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, 2
  %144 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, 3, 0
  %145 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, 3, 1
  %146 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, 4, 0
  %147 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, 4, 1
  %148 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %138, 0
  %149 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %148, ptr %139, 1
  %150 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %149, i64 0, 2
  %151 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %150, i64 %42, 3, 0
  %152 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %151, i64 %146, 4, 0
  %153 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %152, i64 %43, 3, 1
  %154 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %153, i64 1, 4, 1
  %155 = mul i64 %43, %42
  %156 = getelementptr double, ptr null, i64 %155
  %157 = ptrtoint ptr %156 to i64
  %158 = add i64 %157, 64
  %159 = call ptr @malloc(i64 %158)
  %160 = ptrtoint ptr %159 to i64
  %161 = add i64 %160, 63
  %162 = urem i64 %161, 64
  %163 = sub i64 %161, %162
  %164 = inttoptr i64 %163 to ptr
  %165 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %159, 0
  %166 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %165, ptr %164, 1
  %167 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %166, i64 0, 2
  %168 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %167, i64 %42, 3, 0
  %169 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %168, i64 %43, 3, 1
  %170 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %169, i64 %43, 4, 0
  %171 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %170, i64 1, 4, 1
  %172 = call ptr @llvm.stacksave.p0()
  %173 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %154, ptr %173, align 8
  %174 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %173, 1
  %175 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %171, ptr %175, align 8
  %176 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %175, 1
  %177 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %174, ptr %177, align 8
  %178 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %176, ptr %178, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), ptr %177, ptr %178)
  call void @llvm.stackrestore.p0(ptr %172)
  br label %179

179:                                              ; preds = %198, %108
  %180 = phi i64 [ %199, %198 ], [ 0, %108 ]
  %181 = icmp slt i64 %180, %42
  br i1 %181, label %182, label %200

182:                                              ; preds = %179
  br label %183

183:                                              ; preds = %186, %182
  %184 = phi i64 [ %197, %186 ], [ 0, %182 ]
  %185 = icmp slt i64 %184, %43
  br i1 %185, label %186, label %198

186:                                              ; preds = %183
  %187 = getelementptr double, ptr %56, i64 %184
  %188 = load double, ptr %187, align 8
  %189 = mul i64 %180, %43
  %190 = add i64 %189, %184
  %191 = getelementptr double, ptr %164, i64 %190
  %192 = load double, ptr %191, align 8
  %193 = fsub double %192, %188
  %194 = mul i64 %180, %43
  %195 = add i64 %194, %184
  %196 = getelementptr double, ptr %164, i64 %195
  store double %193, ptr %196, align 8
  %197 = add i64 %184, 1
  br label %183

198:                                              ; preds = %183
  %199 = add i64 %180, 1
  br label %179

200:                                              ; preds = %179
  %201 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, 3
  %202 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %201, ptr %202, align 8
  %203 = getelementptr [2 x i64], ptr %202, i32 0, i32 0
  %204 = load i64, ptr %203, align 8
  %205 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, 3
  %206 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %205, ptr %206, align 8
  %207 = getelementptr [2 x i64], ptr %206, i32 0, i32 1
  %208 = load i64, ptr %207, align 8
  %209 = mul i64 %208, %204
  %210 = getelementptr double, ptr null, i64 %209
  %211 = ptrtoint ptr %210 to i64
  %212 = add i64 %211, 64
  %213 = call ptr @malloc(i64 %212)
  %214 = ptrtoint ptr %213 to i64
  %215 = add i64 %214, 63
  %216 = urem i64 %215, 64
  %217 = sub i64 %215, %216
  %218 = inttoptr i64 %217 to ptr
  %219 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %213, 0
  %220 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %219, ptr %218, 1
  %221 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %220, i64 0, 2
  %222 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %221, i64 %204, 3, 0
  %223 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %222, i64 %208, 3, 1
  %224 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %223, i64 %208, 4, 0
  %225 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %224, i64 1, 4, 1
  %226 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, 3, 0
  %227 = mul i64 %226, 1
  %228 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, 3, 1
  %229 = mul i64 %227, %228
  %230 = mul i64 %229, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %231 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, 1
  %232 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, 2
  %233 = getelementptr double, ptr %231, i64 %232
  %234 = getelementptr double, ptr %218, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %234, ptr %233, i64 %230, i1 false)
  %235 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %213, 0
  %236 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %235, ptr %218, 1
  %237 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %236, i64 0, 2
  %238 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %237, i64 %42, 3, 0
  %239 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %238, i64 %208, 4, 0
  %240 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %239, i64 %43, 3, 1
  %241 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %240, i64 1, 4, 1
  %242 = call ptr @llvm.stacksave.p0()
  %243 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %171, ptr %243, align 8
  %244 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %243, 1
  %245 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %241, ptr %245, align 8
  %246 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %245, 1
  %247 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %244, ptr %247, align 8
  %248 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %246, ptr %248, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), ptr %247, ptr %248)
  call void @llvm.stackrestore.p0(ptr %242)
  %249 = mul i64 %208, %204
  %250 = getelementptr double, ptr null, i64 %249
  %251 = ptrtoint ptr %250 to i64
  %252 = add i64 %251, 64
  %253 = call ptr @malloc(i64 %252)
  %254 = ptrtoint ptr %253 to i64
  %255 = add i64 %254, 63
  %256 = urem i64 %255, 64
  %257 = sub i64 %255, %256
  %258 = inttoptr i64 %257 to ptr
  %259 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %253, 0
  %260 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %259, ptr %258, 1
  %261 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %260, i64 0, 2
  %262 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %261, i64 %204, 3, 0
  %263 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %262, i64 %208, 3, 1
  %264 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %263, i64 %208, 4, 0
  %265 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %264, i64 1, 4, 1
  %266 = mul i64 %204, 1
  %267 = mul i64 %266, %208
  %268 = mul i64 %267, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %269 = getelementptr double, ptr %218, i64 0
  %270 = getelementptr double, ptr %258, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %270, ptr %269, i64 %268, i1 false)
  %271 = mul i64 %204, 1
  %272 = mul i64 %271, %208
  %273 = mul i64 %272, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %274 = getelementptr double, ptr %258, i64 0
  %275 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, 1
  %276 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, 2
  %277 = getelementptr double, ptr %275, i64 %276
  call void @llvm.memcpy.p0.p0.i64(ptr %277, ptr %274, i64 %273, i1 false)
  %278 = fsub double %2, 1.000000e+00
  %279 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, 3
  %280 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %279, ptr %280, align 8
  %281 = getelementptr [2 x i64], ptr %280, i32 0, i32 0
  %282 = load i64, ptr %281, align 8
  %283 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, 3
  %284 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %283, ptr %284, align 8
  %285 = getelementptr [2 x i64], ptr %284, i32 0, i32 1
  %286 = load i64, ptr %285, align 8
  %287 = mul i64 %286, %282
  %288 = getelementptr double, ptr null, i64 %287
  %289 = ptrtoint ptr %288 to i64
  %290 = add i64 %289, 64
  %291 = call ptr @malloc(i64 %290)
  %292 = ptrtoint ptr %291 to i64
  %293 = add i64 %292, 63
  %294 = urem i64 %293, 64
  %295 = sub i64 %293, %294
  %296 = inttoptr i64 %295 to ptr
  %297 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %291, 0
  %298 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %297, ptr %296, 1
  %299 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %298, i64 0, 2
  %300 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %299, i64 %282, 3, 0
  %301 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %300, i64 %286, 3, 1
  %302 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %301, i64 %286, 4, 0
  %303 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %302, i64 1, 4, 1
  %304 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, 3, 0
  %305 = mul i64 %304, 1
  %306 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, 3, 1
  %307 = mul i64 %305, %306
  %308 = mul i64 %307, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %309 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, 1
  %310 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, 2
  %311 = getelementptr double, ptr %309, i64 %310
  %312 = getelementptr double, ptr %296, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %312, ptr %311, i64 %308, i1 false)
  br label %313

313:                                              ; preds = %407, %200
  %314 = phi i64 [ %408, %407 ], [ 0, %200 ]
  %315 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %320, %407 ], [ %303, %200 ]
  %316 = icmp slt i64 %314, %43
  br i1 %316, label %317, label %409

317:                                              ; preds = %313
  br label %318

318:                                              ; preds = %371, %317
  %319 = phi i64 [ %406, %371 ], [ %314, %317 ]
  %320 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %320, %371 ], [ %315, %317 ]
  %321 = icmp slt i64 %319, %43
  br i1 %321, label %322, label %407

322:                                              ; preds = %318
  %323 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %320, 1
  %324 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %320, 4, 0
  %325 = mul i64 %314, %324
  %326 = add i64 %325, %319
  %327 = getelementptr double, ptr %323, i64 %326
  store double 0.000000e+00, ptr %327, align 8
  %328 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %213, 0
  %329 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %328, ptr %218, 1
  %330 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %329, i64 %314, 2
  %331 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %330, i64 %42, 3, 0
  %332 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %331, i64 %208, 4, 0
  %333 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %213, 0
  %334 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %333, ptr %218, 1
  %335 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %334, i64 %319, 2
  %336 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %335, i64 %42, 3, 0
  %337 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %336, i64 %208, 4, 0
  %338 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %320, 0
  %339 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %320, 1
  %340 = insertvalue { ptr, ptr, i64 } undef, ptr %338, 0
  %341 = insertvalue { ptr, ptr, i64 } %340, ptr %339, 1
  %342 = insertvalue { ptr, ptr, i64 } %341, i64 0, 2
  %343 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %320, 2
  %344 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %320, 3, 0
  %345 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %320, 3, 1
  %346 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %320, 4, 0
  %347 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %320, 4, 1
  %348 = mul i64 %314, %346
  %349 = add i64 %348, %319
  %350 = insertvalue { ptr, ptr, i64 } undef, ptr %338, 0
  %351 = insertvalue { ptr, ptr, i64 } %350, ptr %339, 1
  %352 = insertvalue { ptr, ptr, i64 } %351, i64 %349, 2
  br label %353

353:                                              ; preds = %356, %322
  %354 = phi i64 [ %370, %356 ], [ 0, %322 ]
  %355 = icmp slt i64 %354, %42
  br i1 %355, label %356, label %371

356:                                              ; preds = %353
  %357 = getelementptr double, ptr %218, i64 %314
  %358 = mul i64 %354, %208
  %359 = getelementptr double, ptr %357, i64 %358
  %360 = load double, ptr %359, align 8
  %361 = getelementptr double, ptr %218, i64 %319
  %362 = mul i64 %354, %208
  %363 = getelementptr double, ptr %361, i64 %362
  %364 = load double, ptr %363, align 8
  %365 = getelementptr double, ptr %339, i64 %349
  %366 = load double, ptr %365, align 8
  %367 = fmul double %360, %364
  %368 = fadd double %366, %367
  %369 = getelementptr double, ptr %339, i64 %349
  store double %368, ptr %369, align 8
  %370 = add i64 %354, 1
  br label %353

371:                                              ; preds = %353
  %372 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %320, 0
  %373 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %320, 1
  %374 = insertvalue { ptr, ptr, i64 } undef, ptr %372, 0
  %375 = insertvalue { ptr, ptr, i64 } %374, ptr %373, 1
  %376 = insertvalue { ptr, ptr, i64 } %375, i64 0, 2
  %377 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %320, 2
  %378 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %320, 3, 0
  %379 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %320, 3, 1
  %380 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %320, 4, 0
  %381 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %320, 4, 1
  %382 = mul i64 %314, %380
  %383 = add i64 %382, %319
  %384 = insertvalue { ptr, ptr, i64 } undef, ptr %372, 0
  %385 = insertvalue { ptr, ptr, i64 } %384, ptr %373, 1
  %386 = insertvalue { ptr, ptr, i64 } %385, i64 %383, 2
  %387 = getelementptr double, ptr %339, i64 %349
  %388 = getelementptr double, ptr %373, i64 %383
  call void @llvm.memcpy.p0.p0.i64(ptr %388, ptr %387, i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), i1 false)
  %389 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %320, 1
  %390 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %320, 4, 0
  %391 = mul i64 %314, %390
  %392 = add i64 %391, %319
  %393 = getelementptr double, ptr %389, i64 %392
  %394 = load double, ptr %393, align 8
  %395 = fdiv double %394, %278
  %396 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %320, 1
  %397 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %320, 4, 0
  %398 = mul i64 %314, %397
  %399 = add i64 %398, %319
  %400 = getelementptr double, ptr %396, i64 %399
  store double %395, ptr %400, align 8
  %401 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %320, 1
  %402 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %320, 4, 0
  %403 = mul i64 %319, %402
  %404 = add i64 %403, %314
  %405 = getelementptr double, ptr %401, i64 %404
  store double %395, ptr %405, align 8
  %406 = add i64 %319, 1
  br label %318

407:                                              ; preds = %318
  %408 = add i64 %314, 1
  br label %313

409:                                              ; preds = %313
  %410 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %315, 3, 0
  %411 = mul i64 %410, 1
  %412 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %315, 3, 1
  %413 = mul i64 %411, %412
  %414 = mul i64 %413, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %415 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %315, 1
  %416 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %315, 2
  %417 = getelementptr double, ptr %415, i64 %416
  %418 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, 1
  %419 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, 2
  %420 = getelementptr double, ptr %418, i64 %419
  call void @llvm.memcpy.p0.p0.i64(ptr %420, ptr %417, i64 %414, i1 false)
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
