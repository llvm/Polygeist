; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @malloc(i64)

declare void @free(ptr)

define void @kernel_bicg_impl(i32 %0, i32 %1, ptr %2, ptr %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr %9, ptr %10, i64 %11, i64 %12, i64 %13, ptr %14, ptr %15, i64 %16, i64 %17, i64 %18, ptr %19, ptr %20, i64 %21, i64 %22, i64 %23, ptr %24, ptr %25, i64 %26, i64 %27, i64 %28) {
  %30 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %2, 0
  %31 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, ptr %3, 1
  %32 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, i64 %4, 2
  %33 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, i64 %5, 3, 0
  %34 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %33, i64 %7, 4, 0
  %35 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, i64 %6, 3, 1
  %36 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, i64 %8, 4, 1
  %37 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %9, 0
  %38 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %37, ptr %10, 1
  %39 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %38, i64 %11, 2
  %40 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %39, i64 %12, 3, 0
  %41 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %40, i64 %13, 4, 0
  %42 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %14, 0
  %43 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %42, ptr %15, 1
  %44 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %43, i64 %16, 2
  %45 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %44, i64 %17, 3, 0
  %46 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %45, i64 %18, 4, 0
  %47 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %19, 0
  %48 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %47, ptr %20, 1
  %49 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %48, i64 %21, 2
  %50 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %49, i64 %22, 3, 0
  %51 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %50, i64 %23, 4, 0
  %52 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %24, 0
  %53 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %52, ptr %25, 1
  %54 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %53, i64 %26, 2
  %55 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %54, i64 %27, 3, 0
  %56 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %55, i64 %28, 4, 0
  %57 = sext i32 %0 to i64
  %58 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %41, 3
  %59 = alloca [1 x i64], i64 1, align 8
  store [1 x i64] %58, ptr %59, align 8
  %60 = getelementptr [1 x i64], ptr %59, i32 0, i32 0
  %61 = load i64, ptr %60, align 8
  %62 = getelementptr double, ptr null, i64 %61
  %63 = ptrtoint ptr %62 to i64
  %64 = add i64 %63, 64
  %65 = call ptr @malloc(i64 %64)
  %66 = ptrtoint ptr %65 to i64
  %67 = add i64 %66, 63
  %68 = urem i64 %67, 64
  %69 = sub i64 %67, %68
  %70 = inttoptr i64 %69 to ptr
  %71 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %65, 0
  %72 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %71, ptr %70, 1
  %73 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %72, i64 0, 2
  %74 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %73, i64 %61, 3, 0
  %75 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %74, i64 1, 4, 0
  br label %76

76:                                               ; preds = %79, %29
  %77 = phi i64 [ %81, %79 ], [ 0, %29 ]
  %78 = icmp slt i64 %77, %61
  br i1 %78, label %79, label %82

79:                                               ; preds = %76
  %80 = getelementptr double, ptr %70, i64 %77
  store double 0.000000e+00, ptr %80, align 8
  %81 = add i64 %77, 1
  br label %76

82:                                               ; preds = %76
  %83 = sext i32 %1 to i64
  %84 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %46, 3
  %85 = alloca [1 x i64], i64 1, align 8
  store [1 x i64] %84, ptr %85, align 8
  %86 = getelementptr [1 x i64], ptr %85, i32 0, i32 0
  %87 = load i64, ptr %86, align 8
  %88 = getelementptr double, ptr null, i64 %87
  %89 = ptrtoint ptr %88 to i64
  %90 = add i64 %89, 64
  %91 = call ptr @malloc(i64 %90)
  %92 = ptrtoint ptr %91 to i64
  %93 = add i64 %92, 63
  %94 = urem i64 %93, 64
  %95 = sub i64 %93, %94
  %96 = inttoptr i64 %95 to ptr
  %97 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %91, 0
  %98 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %97, ptr %96, 1
  %99 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %98, i64 0, 2
  %100 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %99, i64 %87, 3, 0
  %101 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %100, i64 1, 4, 0
  br label %102

102:                                              ; preds = %105, %82
  %103 = phi i64 [ %107, %105 ], [ 0, %82 ]
  %104 = icmp slt i64 %103, %87
  br i1 %104, label %105, label %108

105:                                              ; preds = %102
  %106 = getelementptr double, ptr %96, i64 %103
  store double 0.000000e+00, ptr %106, align 8
  %107 = add i64 %103, 1
  br label %102

108:                                              ; preds = %102
  %109 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %56, 0
  %110 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %56, 1
  %111 = insertvalue { ptr, ptr, i64 } undef, ptr %109, 0
  %112 = insertvalue { ptr, ptr, i64 } %111, ptr %110, 1
  %113 = insertvalue { ptr, ptr, i64 } %112, i64 0, 2
  %114 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %56, 2
  %115 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %56, 3, 0
  %116 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %56, 4, 0
  %117 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %109, 0
  %118 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %117, ptr %110, 1
  %119 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %118, i64 0, 2
  %120 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %119, i64 %83, 3, 0
  %121 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %120, i64 1, 4, 0
  %122 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, 0
  %123 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, 1
  %124 = insertvalue { ptr, ptr, i64 } undef, ptr %122, 0
  %125 = insertvalue { ptr, ptr, i64 } %124, ptr %123, 1
  %126 = insertvalue { ptr, ptr, i64 } %125, i64 0, 2
  %127 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, 2
  %128 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, 3, 0
  %129 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, 3, 1
  %130 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, 4, 0
  %131 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, 4, 1
  %132 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %122, 0
  %133 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %132, ptr %123, 1
  %134 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %133, i64 0, 2
  %135 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %134, i64 %83, 3, 0
  %136 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %135, i64 %130, 4, 0
  %137 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %136, i64 %57, 3, 1
  %138 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %137, i64 1, 4, 1
  %139 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %65, 0
  %140 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %139, ptr %70, 1
  %141 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %140, i64 0, 2
  %142 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %141, i64 %57, 3, 0
  %143 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %142, i64 1, 4, 0
  br label %144

144:                                              ; preds = %165, %108
  %145 = phi i64 [ %166, %165 ], [ 0, %108 ]
  %146 = icmp slt i64 %145, %83
  br i1 %146, label %147, label %167

147:                                              ; preds = %144
  br label %148

148:                                              ; preds = %151, %147
  %149 = phi i64 [ %164, %151 ], [ 0, %147 ]
  %150 = icmp slt i64 %149, %57
  br i1 %150, label %151, label %165

151:                                              ; preds = %148
  %152 = getelementptr double, ptr %110, i64 %145
  %153 = load double, ptr %152, align 8
  %154 = getelementptr double, ptr %123, i64 0
  %155 = mul i64 %145, %130
  %156 = add i64 %155, %149
  %157 = getelementptr double, ptr %154, i64 %156
  %158 = load double, ptr %157, align 8
  %159 = getelementptr double, ptr %70, i64 %149
  %160 = load double, ptr %159, align 8
  %161 = fmul double %153, %158
  %162 = fadd double %160, %161
  %163 = getelementptr double, ptr %70, i64 %149
  store double %162, ptr %163, align 8
  %164 = add i64 %149, 1
  br label %148

165:                                              ; preds = %148
  %166 = add i64 %145, 1
  br label %144

167:                                              ; preds = %144
  %168 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %65, 0
  %169 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %168, ptr %70, 1
  %170 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %169, i64 0, 2
  %171 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %170, i64 %57, 3, 0
  %172 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %171, i64 1, 4, 0
  %173 = mul i64 %57, 1
  %174 = mul i64 %173, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %175 = getelementptr double, ptr %70, i64 0
  %176 = getelementptr double, ptr %70, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %176, ptr %175, i64 %174, i1 false)
  %177 = mul i64 %61, 1
  %178 = mul i64 %177, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %179 = getelementptr double, ptr %70, i64 0
  %180 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %41, 1
  %181 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %41, 2
  %182 = getelementptr double, ptr %180, i64 %181
  call void @llvm.memcpy.p0.p0.i64(ptr %182, ptr %179, i64 %178, i1 false)
  %183 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, 0
  %184 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, 1
  %185 = insertvalue { ptr, ptr, i64 } undef, ptr %183, 0
  %186 = insertvalue { ptr, ptr, i64 } %185, ptr %184, 1
  %187 = insertvalue { ptr, ptr, i64 } %186, i64 0, 2
  %188 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, 2
  %189 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, 3, 0
  %190 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, 3, 1
  %191 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, 4, 0
  %192 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, 4, 1
  %193 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %183, 0
  %194 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %193, ptr %184, 1
  %195 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %194, i64 0, 2
  %196 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %195, i64 %83, 3, 0
  %197 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %196, i64 %191, 4, 0
  %198 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %197, i64 %57, 3, 1
  %199 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %198, i64 1, 4, 1
  %200 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %51, 0
  %201 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %51, 1
  %202 = insertvalue { ptr, ptr, i64 } undef, ptr %200, 0
  %203 = insertvalue { ptr, ptr, i64 } %202, ptr %201, 1
  %204 = insertvalue { ptr, ptr, i64 } %203, i64 0, 2
  %205 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %51, 2
  %206 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %51, 3, 0
  %207 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %51, 4, 0
  %208 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %200, 0
  %209 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %208, ptr %201, 1
  %210 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %209, i64 0, 2
  %211 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %210, i64 %57, 3, 0
  %212 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %211, i64 1, 4, 0
  %213 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %91, 0
  %214 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %213, ptr %96, 1
  %215 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %214, i64 0, 2
  %216 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %215, i64 %83, 3, 0
  %217 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %216, i64 1, 4, 0
  br label %218

218:                                              ; preds = %239, %167
  %219 = phi i64 [ %240, %239 ], [ 0, %167 ]
  %220 = icmp slt i64 %219, %83
  br i1 %220, label %221, label %241

221:                                              ; preds = %218
  br label %222

222:                                              ; preds = %225, %221
  %223 = phi i64 [ %238, %225 ], [ 0, %221 ]
  %224 = icmp slt i64 %223, %57
  br i1 %224, label %225, label %239

225:                                              ; preds = %222
  %226 = getelementptr double, ptr %184, i64 0
  %227 = mul i64 %219, %191
  %228 = add i64 %227, %223
  %229 = getelementptr double, ptr %226, i64 %228
  %230 = load double, ptr %229, align 8
  %231 = getelementptr double, ptr %201, i64 %223
  %232 = load double, ptr %231, align 8
  %233 = getelementptr double, ptr %96, i64 %219
  %234 = load double, ptr %233, align 8
  %235 = fmul double %230, %232
  %236 = fadd double %234, %235
  %237 = getelementptr double, ptr %96, i64 %219
  store double %236, ptr %237, align 8
  %238 = add i64 %223, 1
  br label %222

239:                                              ; preds = %222
  %240 = add i64 %219, 1
  br label %218

241:                                              ; preds = %218
  %242 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %91, 0
  %243 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %242, ptr %96, 1
  %244 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %243, i64 0, 2
  %245 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %244, i64 %83, 3, 0
  %246 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %245, i64 1, 4, 0
  %247 = mul i64 %83, 1
  %248 = mul i64 %247, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %249 = getelementptr double, ptr %96, i64 0
  %250 = getelementptr double, ptr %96, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %250, ptr %249, i64 %248, i1 false)
  %251 = mul i64 %87, 1
  %252 = mul i64 %251, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %253 = getelementptr double, ptr %96, i64 0
  %254 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %46, 1
  %255 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %46, 2
  %256 = getelementptr double, ptr %254, i64 %255
  call void @llvm.memcpy.p0.p0.i64(ptr %256, ptr %253, i64 %252, i1 false)
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #0

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
