; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @malloc(i64)

declare void @free(ptr)

declare void @memrefCopy(i64, ptr, ptr)

define void @kernel_doitgen_impl(i32 %0, i32 %1, i32 %2, ptr %3, ptr %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11, ptr %12, ptr %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, ptr %19, ptr %20, i64 %21, i64 %22, i64 %23) {
  %25 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %3, 0
  %26 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %25, ptr %4, 1
  %27 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %26, i64 %5, 2
  %28 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %27, i64 %6, 3, 0
  %29 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %28, i64 %9, 4, 0
  %30 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %29, i64 %7, 3, 1
  %31 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %30, i64 %10, 4, 1
  %32 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %31, i64 %8, 3, 2
  %33 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %32, i64 %11, 4, 2
  %34 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %12, 0
  %35 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, ptr %13, 1
  %36 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, i64 %14, 2
  %37 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, i64 %15, 3, 0
  %38 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, i64 %17, 4, 0
  %39 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, i64 %16, 3, 1
  %40 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, i64 %18, 4, 1
  %41 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %19, 0
  %42 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %41, ptr %20, 1
  %43 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %42, i64 %21, 2
  %44 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %43, i64 %22, 3, 0
  %45 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %44, i64 %23, 4, 0
  %46 = sext i32 %1 to i64
  %47 = sext i32 %2 to i64
  %48 = sext i32 %0 to i64
  %49 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %45, 3
  %50 = alloca [1 x i64], i64 1, align 8
  store [1 x i64] %49, ptr %50, align 8
  %51 = getelementptr [1 x i64], ptr %50, i32 0, i32 0
  %52 = load i64, ptr %51, align 8
  %53 = getelementptr double, ptr null, i64 %52
  %54 = ptrtoint ptr %53 to i64
  %55 = add i64 %54, 64
  %56 = call ptr @malloc(i64 %55)
  %57 = ptrtoint ptr %56 to i64
  %58 = add i64 %57, 63
  %59 = urem i64 %58, 64
  %60 = sub i64 %58, %59
  %61 = inttoptr i64 %60 to ptr
  %62 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %56, 0
  %63 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %62, ptr %61, 1
  %64 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %63, i64 0, 2
  %65 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %64, i64 %52, 3, 0
  %66 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %65, i64 1, 4, 0
  %67 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %45, 3, 0
  %68 = mul i64 %67, 1
  %69 = mul i64 %68, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %70 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %45, 1
  %71 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %45, 2
  %72 = getelementptr double, ptr %70, i64 %71
  %73 = getelementptr double, ptr %61, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %73, ptr %72, i64 %69, i1 false)
  %74 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %33, 3
  %75 = alloca [3 x i64], i64 1, align 8
  store [3 x i64] %74, ptr %75, align 8
  %76 = getelementptr [3 x i64], ptr %75, i32 0, i32 0
  %77 = load i64, ptr %76, align 8
  %78 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %33, 3
  %79 = alloca [3 x i64], i64 1, align 8
  store [3 x i64] %78, ptr %79, align 8
  %80 = getelementptr [3 x i64], ptr %79, i32 0, i32 1
  %81 = load i64, ptr %80, align 8
  %82 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %33, 3
  %83 = alloca [3 x i64], i64 1, align 8
  store [3 x i64] %82, ptr %83, align 8
  %84 = getelementptr [3 x i64], ptr %83, i32 0, i32 2
  %85 = load i64, ptr %84, align 8
  %86 = mul i64 %85, %81
  %87 = mul i64 %86, %77
  %88 = getelementptr double, ptr null, i64 %87
  %89 = ptrtoint ptr %88 to i64
  %90 = add i64 %89, 64
  %91 = call ptr @malloc(i64 %90)
  %92 = ptrtoint ptr %91 to i64
  %93 = add i64 %92, 63
  %94 = urem i64 %93, 64
  %95 = sub i64 %93, %94
  %96 = inttoptr i64 %95 to ptr
  %97 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %91, 0
  %98 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %97, ptr %96, 1
  %99 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %98, i64 0, 2
  %100 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %99, i64 %77, 3, 0
  %101 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %100, i64 %81, 3, 1
  %102 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %101, i64 %85, 3, 2
  %103 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %102, i64 %86, 4, 0
  %104 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %103, i64 %85, 4, 1
  %105 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %104, i64 1, 4, 2
  %106 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %33, 3, 0
  %107 = mul i64 %106, 1
  %108 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %33, 3, 1
  %109 = mul i64 %107, %108
  %110 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %33, 3, 2
  %111 = mul i64 %109, %110
  %112 = mul i64 %111, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %113 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %33, 1
  %114 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %33, 2
  %115 = getelementptr double, ptr %113, i64 %114
  %116 = getelementptr double, ptr %96, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %116, ptr %115, i64 %112, i1 false)
  br label %117

117:                                              ; preds = %291, %24
  %118 = phi i64 [ %292, %291 ], [ 0, %24 ]
  %119 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ %125, %291 ], [ %66, %24 ]
  %120 = phi { ptr, ptr, i64, [3 x i64], [3 x i64] } [ %126, %291 ], [ %105, %24 ]
  %121 = icmp slt i64 %118, %48
  br i1 %121, label %122, label %293

122:                                              ; preds = %117
  br label %123

123:                                              ; preds = %262, %122
  %124 = phi i64 [ %290, %262 ], [ 0, %122 ]
  %125 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ %125, %262 ], [ %119, %122 ]
  %126 = phi { ptr, ptr, i64, [3 x i64], [3 x i64] } [ %126, %262 ], [ %120, %122 ]
  %127 = icmp slt i64 %124, %46
  br i1 %127, label %128, label %291

128:                                              ; preds = %123
  %129 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %125, 3
  %130 = alloca [1 x i64], i64 1, align 8
  store [1 x i64] %129, ptr %130, align 8
  %131 = getelementptr [1 x i64], ptr %130, i32 0, i32 0
  %132 = load i64, ptr %131, align 8
  br label %133

133:                                              ; preds = %136, %128
  %134 = phi i64 [ %139, %136 ], [ 0, %128 ]
  %135 = icmp slt i64 %134, %132
  br i1 %135, label %136, label %140

136:                                              ; preds = %133
  %137 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %125, 1
  %138 = getelementptr double, ptr %137, i64 %134
  store double 0.000000e+00, ptr %138, align 8
  %139 = add i64 %134, 1
  br label %133

140:                                              ; preds = %133
  %141 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %126, 0
  %142 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %126, 1
  %143 = insertvalue { ptr, ptr, i64 } undef, ptr %141, 0
  %144 = insertvalue { ptr, ptr, i64 } %143, ptr %142, 1
  %145 = insertvalue { ptr, ptr, i64 } %144, i64 0, 2
  %146 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %126, 2
  %147 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %126, 3, 0
  %148 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %126, 3, 1
  %149 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %126, 3, 2
  %150 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %126, 4, 0
  %151 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %126, 4, 1
  %152 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %126, 4, 2
  %153 = mul i64 %118, %150
  %154 = mul i64 %124, %151
  %155 = add i64 %153, %154
  %156 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %141, 0
  %157 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %156, ptr %142, 1
  %158 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %157, i64 %155, 2
  %159 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %158, i64 %47, 3, 0
  %160 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %159, i64 1, 4, 0
  %161 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, 0
  %162 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, 1
  %163 = insertvalue { ptr, ptr, i64 } undef, ptr %161, 0
  %164 = insertvalue { ptr, ptr, i64 } %163, ptr %162, 1
  %165 = insertvalue { ptr, ptr, i64 } %164, i64 0, 2
  %166 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, 2
  %167 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, 3, 0
  %168 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, 3, 1
  %169 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, 4, 0
  %170 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, 4, 1
  %171 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %161, 0
  %172 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %171, ptr %162, 1
  %173 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %172, i64 0, 2
  %174 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %173, i64 %47, 3, 0
  %175 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %174, i64 %169, 4, 0
  %176 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %175, i64 %47, 3, 1
  %177 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %176, i64 1, 4, 1
  %178 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %125, 0
  %179 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %125, 1
  %180 = insertvalue { ptr, ptr, i64 } undef, ptr %178, 0
  %181 = insertvalue { ptr, ptr, i64 } %180, ptr %179, 1
  %182 = insertvalue { ptr, ptr, i64 } %181, i64 0, 2
  %183 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %125, 2
  %184 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %125, 3, 0
  %185 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %125, 4, 0
  %186 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %178, 0
  %187 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %186, ptr %179, 1
  %188 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %187, i64 0, 2
  %189 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %188, i64 %47, 3, 0
  %190 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %189, i64 1, 4, 0
  br label %191

191:                                              ; preds = %213, %140
  %192 = phi i64 [ %214, %213 ], [ 0, %140 ]
  %193 = icmp slt i64 %192, %47
  br i1 %193, label %194, label %215

194:                                              ; preds = %191
  br label %195

195:                                              ; preds = %198, %194
  %196 = phi i64 [ %212, %198 ], [ 0, %194 ]
  %197 = icmp slt i64 %196, %47
  br i1 %197, label %198, label %213

198:                                              ; preds = %195
  %199 = getelementptr double, ptr %142, i64 %155
  %200 = getelementptr double, ptr %199, i64 %196
  %201 = load double, ptr %200, align 8
  %202 = getelementptr double, ptr %162, i64 0
  %203 = mul i64 %196, %169
  %204 = add i64 %203, %192
  %205 = getelementptr double, ptr %202, i64 %204
  %206 = load double, ptr %205, align 8
  %207 = getelementptr double, ptr %179, i64 %192
  %208 = load double, ptr %207, align 8
  %209 = fmul double %201, %206
  %210 = fadd double %208, %209
  %211 = getelementptr double, ptr %179, i64 %192
  store double %210, ptr %211, align 8
  %212 = add i64 %196, 1
  br label %195

213:                                              ; preds = %195
  %214 = add i64 %192, 1
  br label %191

215:                                              ; preds = %191
  %216 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %125, 0
  %217 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %125, 1
  %218 = insertvalue { ptr, ptr, i64 } undef, ptr %216, 0
  %219 = insertvalue { ptr, ptr, i64 } %218, ptr %217, 1
  %220 = insertvalue { ptr, ptr, i64 } %219, i64 0, 2
  %221 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %125, 2
  %222 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %125, 3, 0
  %223 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %125, 4, 0
  %224 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %216, 0
  %225 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %224, ptr %217, 1
  %226 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %225, i64 0, 2
  %227 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %226, i64 %47, 3, 0
  %228 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %227, i64 1, 4, 0
  %229 = mul i64 %47, 1
  %230 = mul i64 %229, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %231 = getelementptr double, ptr %179, i64 0
  %232 = getelementptr double, ptr %217, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %232, ptr %231, i64 %230, i1 false)
  %233 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %126, 0
  %234 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %126, 1
  %235 = insertvalue { ptr, ptr, i64 } undef, ptr %233, 0
  %236 = insertvalue { ptr, ptr, i64 } %235, ptr %234, 1
  %237 = insertvalue { ptr, ptr, i64 } %236, i64 0, 2
  %238 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %126, 2
  %239 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %126, 3, 0
  %240 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %126, 3, 1
  %241 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %126, 3, 2
  %242 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %126, 4, 0
  %243 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %126, 4, 1
  %244 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %126, 4, 2
  %245 = mul i64 %118, %242
  %246 = mul i64 %124, %243
  %247 = add i64 %245, %246
  %248 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %233, 0
  %249 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %248, ptr %234, 1
  %250 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %249, i64 %247, 2
  %251 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %250, i64 %47, 3, 0
  %252 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %251, i64 1, 4, 0
  br label %253

253:                                              ; preds = %256, %215
  %254 = phi i64 [ %261, %256 ], [ 0, %215 ]
  %255 = icmp slt i64 %254, %47
  br i1 %255, label %256, label %262

256:                                              ; preds = %253
  %257 = getelementptr double, ptr %179, i64 %254
  %258 = load double, ptr %257, align 8
  %259 = getelementptr double, ptr %234, i64 %247
  %260 = getelementptr double, ptr %259, i64 %254
  store double %258, ptr %260, align 8
  %261 = add i64 %254, 1
  br label %253

262:                                              ; preds = %253
  %263 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %126, 0
  %264 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %126, 1
  %265 = insertvalue { ptr, ptr, i64 } undef, ptr %263, 0
  %266 = insertvalue { ptr, ptr, i64 } %265, ptr %264, 1
  %267 = insertvalue { ptr, ptr, i64 } %266, i64 0, 2
  %268 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %126, 2
  %269 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %126, 3, 0
  %270 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %126, 3, 1
  %271 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %126, 3, 2
  %272 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %126, 4, 0
  %273 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %126, 4, 1
  %274 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %126, 4, 2
  %275 = mul i64 %118, %272
  %276 = mul i64 %124, %273
  %277 = add i64 %275, %276
  %278 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %263, 0
  %279 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %278, ptr %264, 1
  %280 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %279, i64 %277, 2
  %281 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %280, i64 %47, 3, 0
  %282 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %281, i64 1, 4, 0
  %283 = call ptr @llvm.stacksave.p0()
  %284 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %252, ptr %284, align 8
  %285 = insertvalue { i64, ptr } { i64 1, ptr undef }, ptr %284, 1
  %286 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %282, ptr %286, align 8
  %287 = insertvalue { i64, ptr } { i64 1, ptr undef }, ptr %286, 1
  %288 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %285, ptr %288, align 8
  %289 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %287, ptr %289, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), ptr %288, ptr %289)
  call void @llvm.stackrestore.p0(ptr %283)
  %290 = add i64 %124, 1
  br label %123

291:                                              ; preds = %123
  %292 = add i64 %118, 1
  br label %117

293:                                              ; preds = %117
  %294 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %120, 3, 0
  %295 = mul i64 %294, 1
  %296 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %120, 3, 1
  %297 = mul i64 %295, %296
  %298 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %120, 3, 2
  %299 = mul i64 %297, %298
  %300 = mul i64 %299, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %301 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %120, 1
  %302 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %120, 2
  %303 = getelementptr double, ptr %301, i64 %302
  %304 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %33, 1
  %305 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %33, 2
  %306 = getelementptr double, ptr %304, i64 %305
  call void @llvm.memcpy.p0.p0.i64(ptr %306, ptr %303, i64 %300, i1 false)
  %307 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %119, 3, 0
  %308 = mul i64 %307, 1
  %309 = mul i64 %308, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %310 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %119, 1
  %311 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %119, 2
  %312 = getelementptr double, ptr %310, i64 %311
  %313 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %45, 1
  %314 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %45, 2
  %315 = getelementptr double, ptr %313, i64 %314
  call void @llvm.memcpy.p0.p0.i64(ptr %315, ptr %312, i64 %309, i1 false)
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
