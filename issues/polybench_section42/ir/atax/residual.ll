; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @malloc(i64)

declare void @free(ptr)

define void @kernel_atax_impl(i32 %0, i32 %1, ptr %2, ptr %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr %9, ptr %10, i64 %11, i64 %12, i64 %13, ptr %14, ptr %15, i64 %16, i64 %17, i64 %18, ptr %19, ptr %20, i64 %21, i64 %22, i64 %23) {
  %25 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %2, 0
  %26 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, ptr %3, 1
  %27 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %26, i64 %4, 2
  %28 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, i64 %5, 3, 0
  %29 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, i64 %7, 4, 0
  %30 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, i64 %6, 3, 1
  %31 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, i64 %8, 4, 1
  %32 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %9, 0
  %33 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %32, ptr %10, 1
  %34 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, i64 %11, 2
  %35 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %34, i64 %12, 3, 0
  %36 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %35, i64 %13, 4, 0
  %37 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %14, 0
  %38 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %37, ptr %15, 1
  %39 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %38, i64 %16, 2
  %40 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %39, i64 %17, 3, 0
  %41 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %40, i64 %18, 4, 0
  %42 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %19, 0
  %43 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %42, ptr %20, 1
  %44 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %43, i64 %21, 2
  %45 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %44, i64 %22, 3, 0
  %46 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %45, i64 %23, 4, 0
  %47 = sext i32 %1 to i64
  %48 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %41, 3
  %49 = alloca [1 x i64], i64 1, align 8
  store [1 x i64] %48, ptr %49, align 8
  %50 = getelementptr [1 x i64], ptr %49, i32 0, i32 0
  %51 = load i64, ptr %50, align 8
  %52 = getelementptr double, ptr null, i64 %51
  %53 = ptrtoint ptr %52 to i64
  %54 = add i64 %53, 64
  %55 = call ptr @malloc(i64 %54)
  %56 = ptrtoint ptr %55 to i64
  %57 = add i64 %56, 63
  %58 = urem i64 %57, 64
  %59 = sub i64 %57, %58
  %60 = inttoptr i64 %59 to ptr
  %61 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %55, 0
  %62 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %61, ptr %60, 1
  %63 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %62, i64 0, 2
  %64 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %63, i64 %51, 3, 0
  %65 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %64, i64 1, 4, 0
  br label %66

66:                                               ; preds = %69, %24
  %67 = phi i64 [ %71, %69 ], [ 0, %24 ]
  %68 = icmp slt i64 %67, %51
  br i1 %68, label %69, label %72

69:                                               ; preds = %66
  %70 = getelementptr double, ptr %60, i64 %67
  store double 0.000000e+00, ptr %70, align 8
  %71 = add i64 %67, 1
  br label %66

72:                                               ; preds = %66
  %73 = sext i32 %0 to i64
  %74 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %46, 3
  %75 = alloca [1 x i64], i64 1, align 8
  store [1 x i64] %74, ptr %75, align 8
  %76 = getelementptr [1 x i64], ptr %75, i32 0, i32 0
  %77 = load i64, ptr %76, align 8
  %78 = getelementptr double, ptr null, i64 %77
  %79 = ptrtoint ptr %78 to i64
  %80 = add i64 %79, 64
  %81 = call ptr @malloc(i64 %80)
  %82 = ptrtoint ptr %81 to i64
  %83 = add i64 %82, 63
  %84 = urem i64 %83, 64
  %85 = sub i64 %83, %84
  %86 = inttoptr i64 %85 to ptr
  %87 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %81, 0
  %88 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %87, ptr %86, 1
  %89 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %88, i64 0, 2
  %90 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %89, i64 %77, 3, 0
  %91 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %90, i64 1, 4, 0
  br label %92

92:                                               ; preds = %95, %72
  %93 = phi i64 [ %97, %95 ], [ 0, %72 ]
  %94 = icmp slt i64 %93, %77
  br i1 %94, label %95, label %98

95:                                               ; preds = %92
  %96 = getelementptr double, ptr %86, i64 %93
  store double 0.000000e+00, ptr %96, align 8
  %97 = add i64 %93, 1
  br label %92

98:                                               ; preds = %92
  %99 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, 0
  %100 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, 1
  %101 = insertvalue { ptr, ptr, i64 } undef, ptr %99, 0
  %102 = insertvalue { ptr, ptr, i64 } %101, ptr %100, 1
  %103 = insertvalue { ptr, ptr, i64 } %102, i64 0, 2
  %104 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, 2
  %105 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, 3, 0
  %106 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, 3, 1
  %107 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, 4, 0
  %108 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, 4, 1
  %109 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %99, 0
  %110 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %109, ptr %100, 1
  %111 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %110, i64 0, 2
  %112 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, i64 %73, 3, 0
  %113 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, i64 %107, 4, 0
  %114 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %113, i64 %47, 3, 1
  %115 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %114, i64 1, 4, 1
  %116 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %36, 0
  %117 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %36, 1
  %118 = insertvalue { ptr, ptr, i64 } undef, ptr %116, 0
  %119 = insertvalue { ptr, ptr, i64 } %118, ptr %117, 1
  %120 = insertvalue { ptr, ptr, i64 } %119, i64 0, 2
  %121 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %36, 2
  %122 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %36, 3, 0
  %123 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %36, 4, 0
  %124 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %116, 0
  %125 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %124, ptr %117, 1
  %126 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %125, i64 0, 2
  %127 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %126, i64 %47, 3, 0
  %128 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %127, i64 1, 4, 0
  %129 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %81, 0
  %130 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %129, ptr %86, 1
  %131 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %130, i64 0, 2
  %132 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %131, i64 %73, 3, 0
  %133 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %132, i64 1, 4, 0
  %134 = getelementptr double, ptr null, i64 %73
  %135 = ptrtoint ptr %134 to i64
  %136 = add i64 %135, 64
  %137 = call ptr @malloc(i64 %136)
  %138 = ptrtoint ptr %137 to i64
  %139 = add i64 %138, 63
  %140 = urem i64 %139, 64
  %141 = sub i64 %139, %140
  %142 = inttoptr i64 %141 to ptr
  %143 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %137, 0
  %144 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %143, ptr %142, 1
  %145 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %144, i64 0, 2
  %146 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %145, i64 %73, 3, 0
  %147 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %146, i64 1, 4, 0
  %148 = mul i64 %73, 1
  %149 = mul i64 %148, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %150 = getelementptr double, ptr %86, i64 0
  %151 = getelementptr double, ptr %142, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %151, ptr %150, i64 %149, i1 false)
  br label %152

152:                                              ; preds = %173, %98
  %153 = phi i64 [ %174, %173 ], [ 0, %98 ]
  %154 = icmp slt i64 %153, %73
  br i1 %154, label %155, label %175

155:                                              ; preds = %152
  br label %156

156:                                              ; preds = %159, %155
  %157 = phi i64 [ %172, %159 ], [ 0, %155 ]
  %158 = icmp slt i64 %157, %47
  br i1 %158, label %159, label %173

159:                                              ; preds = %156
  %160 = getelementptr double, ptr %100, i64 0
  %161 = mul i64 %153, %107
  %162 = add i64 %161, %157
  %163 = getelementptr double, ptr %160, i64 %162
  %164 = load double, ptr %163, align 8
  %165 = getelementptr double, ptr %117, i64 %157
  %166 = load double, ptr %165, align 8
  %167 = getelementptr double, ptr %142, i64 %153
  %168 = load double, ptr %167, align 8
  %169 = fmul double %164, %166
  %170 = fadd double %168, %169
  %171 = getelementptr double, ptr %142, i64 %153
  store double %170, ptr %171, align 8
  %172 = add i64 %157, 1
  br label %156

173:                                              ; preds = %156
  %174 = add i64 %153, 1
  br label %152

175:                                              ; preds = %152
  %176 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %81, 0
  %177 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %176, ptr %86, 1
  %178 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %177, i64 0, 2
  %179 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %178, i64 %73, 3, 0
  %180 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %179, i64 1, 4, 0
  %181 = mul i64 %73, 1
  %182 = mul i64 %181, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %183 = getelementptr double, ptr %142, i64 0
  %184 = getelementptr double, ptr %86, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %184, ptr %183, i64 %182, i1 false)
  %185 = mul i64 %77, 1
  %186 = mul i64 %185, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %187 = getelementptr double, ptr %86, i64 0
  %188 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %46, 1
  %189 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %46, 2
  %190 = getelementptr double, ptr %188, i64 %189
  call void @llvm.memcpy.p0.p0.i64(ptr %190, ptr %187, i64 %186, i1 false)
  %191 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, 0
  %192 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, 1
  %193 = insertvalue { ptr, ptr, i64 } undef, ptr %191, 0
  %194 = insertvalue { ptr, ptr, i64 } %193, ptr %192, 1
  %195 = insertvalue { ptr, ptr, i64 } %194, i64 0, 2
  %196 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, 2
  %197 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, 3, 0
  %198 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, 3, 1
  %199 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, 4, 0
  %200 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, 4, 1
  %201 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %191, 0
  %202 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %201, ptr %192, 1
  %203 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %202, i64 0, 2
  %204 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %203, i64 %73, 3, 0
  %205 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %204, i64 %199, 4, 0
  %206 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %205, i64 %47, 3, 1
  %207 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %206, i64 1, 4, 1
  %208 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %55, 0
  %209 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %208, ptr %60, 1
  %210 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %209, i64 0, 2
  %211 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %210, i64 %47, 3, 0
  %212 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %211, i64 1, 4, 0
  br label %213

213:                                              ; preds = %234, %175
  %214 = phi i64 [ %235, %234 ], [ 0, %175 ]
  %215 = icmp slt i64 %214, %73
  br i1 %215, label %216, label %236

216:                                              ; preds = %213
  br label %217

217:                                              ; preds = %220, %216
  %218 = phi i64 [ %233, %220 ], [ 0, %216 ]
  %219 = icmp slt i64 %218, %47
  br i1 %219, label %220, label %234

220:                                              ; preds = %217
  %221 = getelementptr double, ptr %192, i64 0
  %222 = mul i64 %214, %199
  %223 = add i64 %222, %218
  %224 = getelementptr double, ptr %221, i64 %223
  %225 = load double, ptr %224, align 8
  %226 = getelementptr double, ptr %142, i64 %214
  %227 = load double, ptr %226, align 8
  %228 = getelementptr double, ptr %60, i64 %218
  %229 = load double, ptr %228, align 8
  %230 = fmul double %225, %227
  %231 = fadd double %229, %230
  %232 = getelementptr double, ptr %60, i64 %218
  store double %231, ptr %232, align 8
  %233 = add i64 %218, 1
  br label %217

234:                                              ; preds = %217
  %235 = add i64 %214, 1
  br label %213

236:                                              ; preds = %213
  %237 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %55, 0
  %238 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %237, ptr %60, 1
  %239 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %238, i64 0, 2
  %240 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %239, i64 %47, 3, 0
  %241 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %240, i64 1, 4, 0
  %242 = mul i64 %47, 1
  %243 = mul i64 %242, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %244 = getelementptr double, ptr %60, i64 0
  %245 = getelementptr double, ptr %60, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %245, ptr %244, i64 %243, i1 false)
  %246 = mul i64 %51, 1
  %247 = mul i64 %246, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %248 = getelementptr double, ptr %60, i64 0
  %249 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %41, 1
  %250 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %41, 2
  %251 = getelementptr double, ptr %249, i64 %250
  call void @llvm.memcpy.p0.p0.i64(ptr %251, ptr %248, i64 %247, i1 false)
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #0

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
