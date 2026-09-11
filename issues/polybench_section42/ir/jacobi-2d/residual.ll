; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @malloc(i64)

declare void @free(ptr)

declare void @memrefCopy(i64, ptr, ptr)

define void @kernel_jacobi_2d_impl(i32 %0, i32 %1, ptr %2, ptr %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr %9, ptr %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15) {
  %17 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %2, 0
  %18 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %17, ptr %3, 1
  %19 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %18, i64 %4, 2
  %20 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %19, i64 %5, 3, 0
  %21 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %20, i64 %7, 4, 0
  %22 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, i64 %6, 3, 1
  %23 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, i64 %8, 4, 1
  %24 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %9, 0
  %25 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, ptr %10, 1
  %26 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, i64 %11, 2
  %27 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %26, i64 %12, 3, 0
  %28 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, i64 %14, 4, 0
  %29 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, i64 %13, 3, 1
  %30 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, i64 %15, 4, 1
  %31 = sext i32 %1 to i64
  %32 = sext i32 %0 to i64
  %33 = add i64 %31, -1
  %34 = add i64 %31, -1
  %35 = sub i64 %34, 1
  %36 = add i64 %31, -1
  %37 = sub i64 %36, 1
  %38 = sub i64 %33, 1
  %39 = add i64 %31, -1
  %40 = sub i64 %39, 1
  %41 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, 3
  %42 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %41, ptr %42, align 8
  %43 = getelementptr [2 x i64], ptr %42, i32 0, i32 0
  %44 = load i64, ptr %43, align 8
  %45 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, 3
  %46 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %45, ptr %46, align 8
  %47 = getelementptr [2 x i64], ptr %46, i32 0, i32 1
  %48 = load i64, ptr %47, align 8
  %49 = mul i64 %48, %44
  %50 = getelementptr double, ptr null, i64 %49
  %51 = ptrtoint ptr %50 to i64
  %52 = add i64 %51, 64
  %53 = call ptr @malloc(i64 %52)
  %54 = ptrtoint ptr %53 to i64
  %55 = add i64 %54, 63
  %56 = urem i64 %55, 64
  %57 = sub i64 %55, %56
  %58 = inttoptr i64 %57 to ptr
  %59 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %53, 0
  %60 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %59, ptr %58, 1
  %61 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, i64 0, 2
  %62 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %61, i64 %44, 3, 0
  %63 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %62, i64 %48, 3, 1
  %64 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %63, i64 %48, 4, 0
  %65 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %64, i64 1, 4, 1
  %66 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, 3, 0
  %67 = mul i64 %66, 1
  %68 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, 3, 1
  %69 = mul i64 %67, %68
  %70 = mul i64 %69, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %71 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, 1
  %72 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, 2
  %73 = getelementptr double, ptr %71, i64 %72
  %74 = getelementptr double, ptr %58, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %74, ptr %73, i64 %70, i1 false)
  %75 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, 3
  %76 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %75, ptr %76, align 8
  %77 = getelementptr [2 x i64], ptr %76, i32 0, i32 0
  %78 = load i64, ptr %77, align 8
  %79 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, 3
  %80 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %79, ptr %80, align 8
  %81 = getelementptr [2 x i64], ptr %80, i32 0, i32 1
  %82 = load i64, ptr %81, align 8
  %83 = mul i64 %82, %78
  %84 = getelementptr double, ptr null, i64 %83
  %85 = ptrtoint ptr %84 to i64
  %86 = add i64 %85, 64
  %87 = call ptr @malloc(i64 %86)
  %88 = ptrtoint ptr %87 to i64
  %89 = add i64 %88, 63
  %90 = urem i64 %89, 64
  %91 = sub i64 %89, %90
  %92 = inttoptr i64 %91 to ptr
  %93 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %87, 0
  %94 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %93, ptr %92, 1
  %95 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %94, i64 0, 2
  %96 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %95, i64 %78, 3, 0
  %97 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %96, i64 %82, 3, 1
  %98 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %97, i64 %82, 4, 0
  %99 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %98, i64 1, 4, 1
  %100 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, 3, 0
  %101 = mul i64 %100, 1
  %102 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, 3, 1
  %103 = mul i64 %101, %102
  %104 = mul i64 %103, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %105 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, 1
  %106 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, 2
  %107 = getelementptr double, ptr %105, i64 %106
  %108 = getelementptr double, ptr %92, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %108, ptr %107, i64 %104, i1 false)
  br label %109

109:                                              ; preds = %445, %16
  %110 = phi i64 [ %471, %445 ], [ 0, %16 ]
  %111 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %111, %445 ], [ %65, %16 ]
  %112 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %112, %445 ], [ %99, %16 ]
  %113 = icmp slt i64 %110, %32
  br i1 %113, label %114, label %472

114:                                              ; preds = %109
  %115 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 0
  %116 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 1
  %117 = insertvalue { ptr, ptr, i64 } undef, ptr %115, 0
  %118 = insertvalue { ptr, ptr, i64 } %117, ptr %116, 1
  %119 = insertvalue { ptr, ptr, i64 } %118, i64 0, 2
  %120 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 2
  %121 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 3, 0
  %122 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 3, 1
  %123 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 4, 0
  %124 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 4, 1
  %125 = add i64 %123, 1
  %126 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %115, 0
  %127 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %126, ptr %116, 1
  %128 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %127, i64 %125, 2
  %129 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %128, i64 %37, 3, 0
  %130 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %129, i64 %123, 4, 0
  %131 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %130, i64 %35, 3, 1
  %132 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %131, i64 1, 4, 1
  %133 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 0
  %134 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 1
  %135 = insertvalue { ptr, ptr, i64 } undef, ptr %133, 0
  %136 = insertvalue { ptr, ptr, i64 } %135, ptr %134, 1
  %137 = insertvalue { ptr, ptr, i64 } %136, i64 0, 2
  %138 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 2
  %139 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 3, 0
  %140 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 3, 1
  %141 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 4, 0
  %142 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 4, 1
  %143 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %133, 0
  %144 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %143, ptr %134, 1
  %145 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %144, i64 %141, 2
  %146 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %145, i64 %37, 3, 0
  %147 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %146, i64 %141, 4, 0
  %148 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %147, i64 %35, 3, 1
  %149 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %148, i64 1, 4, 1
  %150 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 0
  %151 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 1
  %152 = insertvalue { ptr, ptr, i64 } undef, ptr %150, 0
  %153 = insertvalue { ptr, ptr, i64 } %152, ptr %151, 1
  %154 = insertvalue { ptr, ptr, i64 } %153, i64 0, 2
  %155 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 2
  %156 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 3, 0
  %157 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 3, 1
  %158 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 4, 0
  %159 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 4, 1
  %160 = add i64 %158, 2
  %161 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %150, 0
  %162 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %161, ptr %151, 1
  %163 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %162, i64 %160, 2
  %164 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %163, i64 %37, 3, 0
  %165 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %164, i64 %158, 4, 0
  %166 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %165, i64 %35, 3, 1
  %167 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %166, i64 1, 4, 1
  %168 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 0
  %169 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 1
  %170 = insertvalue { ptr, ptr, i64 } undef, ptr %168, 0
  %171 = insertvalue { ptr, ptr, i64 } %170, ptr %169, 1
  %172 = insertvalue { ptr, ptr, i64 } %171, i64 0, 2
  %173 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 2
  %174 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 3, 0
  %175 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 3, 1
  %176 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 4, 0
  %177 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 4, 1
  %178 = mul i64 %176, 2
  %179 = add i64 %178, 1
  %180 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %168, 0
  %181 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %180, ptr %169, 1
  %182 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %181, i64 %179, 2
  %183 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %182, i64 %37, 3, 0
  %184 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %183, i64 %176, 4, 0
  %185 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %184, i64 %35, 3, 1
  %186 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %185, i64 1, 4, 1
  %187 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 0
  %188 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 1
  %189 = insertvalue { ptr, ptr, i64 } undef, ptr %187, 0
  %190 = insertvalue { ptr, ptr, i64 } %189, ptr %188, 1
  %191 = insertvalue { ptr, ptr, i64 } %190, i64 0, 2
  %192 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 2
  %193 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 3, 0
  %194 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 3, 1
  %195 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 4, 0
  %196 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 4, 1
  %197 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %187, 0
  %198 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %197, ptr %188, 1
  %199 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %198, i64 1, 2
  %200 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %199, i64 %37, 3, 0
  %201 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %200, i64 %195, 4, 0
  %202 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %201, i64 %35, 3, 1
  %203 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %202, i64 1, 4, 1
  %204 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 0
  %205 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 1
  %206 = insertvalue { ptr, ptr, i64 } undef, ptr %204, 0
  %207 = insertvalue { ptr, ptr, i64 } %206, ptr %205, 1
  %208 = insertvalue { ptr, ptr, i64 } %207, i64 0, 2
  %209 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 2
  %210 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 3, 0
  %211 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 3, 1
  %212 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 4, 0
  %213 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 4, 1
  %214 = add i64 %212, 1
  %215 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %204, 0
  %216 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %215, ptr %205, 1
  %217 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %216, i64 %214, 2
  %218 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %217, i64 %37, 3, 0
  %219 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %218, i64 %212, 4, 0
  %220 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %219, i64 %35, 3, 1
  %221 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %220, i64 1, 4, 1
  br label %222

222:                                              ; preds = %265, %114
  %223 = phi i64 [ %266, %265 ], [ 0, %114 ]
  %224 = icmp slt i64 %223, %37
  br i1 %224, label %225, label %267

225:                                              ; preds = %222
  br label %226

226:                                              ; preds = %229, %225
  %227 = phi i64 [ %264, %229 ], [ 0, %225 ]
  %228 = icmp slt i64 %227, %35
  br i1 %228, label %229, label %265

229:                                              ; preds = %226
  %230 = getelementptr double, ptr %116, i64 %125
  %231 = mul i64 %223, %123
  %232 = add i64 %231, %227
  %233 = getelementptr double, ptr %230, i64 %232
  %234 = load double, ptr %233, align 8
  %235 = getelementptr double, ptr %134, i64 %141
  %236 = mul i64 %223, %141
  %237 = add i64 %236, %227
  %238 = getelementptr double, ptr %235, i64 %237
  %239 = load double, ptr %238, align 8
  %240 = getelementptr double, ptr %151, i64 %160
  %241 = mul i64 %223, %158
  %242 = add i64 %241, %227
  %243 = getelementptr double, ptr %240, i64 %242
  %244 = load double, ptr %243, align 8
  %245 = getelementptr double, ptr %169, i64 %179
  %246 = mul i64 %223, %176
  %247 = add i64 %246, %227
  %248 = getelementptr double, ptr %245, i64 %247
  %249 = load double, ptr %248, align 8
  %250 = getelementptr double, ptr %188, i64 1
  %251 = mul i64 %223, %195
  %252 = add i64 %251, %227
  %253 = getelementptr double, ptr %250, i64 %252
  %254 = load double, ptr %253, align 8
  %255 = fadd double %234, %239
  %256 = fadd double %255, %244
  %257 = fadd double %256, %249
  %258 = fadd double %257, %254
  %259 = fmul double %258, 2.000000e-01
  %260 = getelementptr double, ptr %205, i64 %214
  %261 = mul i64 %223, %212
  %262 = add i64 %261, %227
  %263 = getelementptr double, ptr %260, i64 %262
  store double %259, ptr %263, align 8
  %264 = add i64 %227, 1
  br label %226

265:                                              ; preds = %226
  %266 = add i64 %223, 1
  br label %222

267:                                              ; preds = %222
  %268 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 0
  %269 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 1
  %270 = insertvalue { ptr, ptr, i64 } undef, ptr %268, 0
  %271 = insertvalue { ptr, ptr, i64 } %270, ptr %269, 1
  %272 = insertvalue { ptr, ptr, i64 } %271, i64 0, 2
  %273 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 2
  %274 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 3, 0
  %275 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 3, 1
  %276 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 4, 0
  %277 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 4, 1
  %278 = add i64 %276, 1
  %279 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %268, 0
  %280 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %279, ptr %269, 1
  %281 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %280, i64 %278, 2
  %282 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %281, i64 %37, 3, 0
  %283 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %282, i64 %276, 4, 0
  %284 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %283, i64 %35, 3, 1
  %285 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %284, i64 1, 4, 1
  %286 = call ptr @llvm.stacksave.p0()
  %287 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %221, ptr %287, align 8
  %288 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %287, 1
  %289 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %285, ptr %289, align 8
  %290 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %289, 1
  %291 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %288, ptr %291, align 8
  %292 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %290, ptr %292, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), ptr %291, ptr %292)
  call void @llvm.stackrestore.p0(ptr %286)
  %293 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 0
  %294 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 1
  %295 = insertvalue { ptr, ptr, i64 } undef, ptr %293, 0
  %296 = insertvalue { ptr, ptr, i64 } %295, ptr %294, 1
  %297 = insertvalue { ptr, ptr, i64 } %296, i64 0, 2
  %298 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 2
  %299 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 3, 0
  %300 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 3, 1
  %301 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 4, 0
  %302 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 4, 1
  %303 = add i64 %301, 1
  %304 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %293, 0
  %305 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %304, ptr %294, 1
  %306 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %305, i64 %303, 2
  %307 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %306, i64 %40, 3, 0
  %308 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %307, i64 %301, 4, 0
  %309 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %308, i64 %38, 3, 1
  %310 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %309, i64 1, 4, 1
  %311 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 0
  %312 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 1
  %313 = insertvalue { ptr, ptr, i64 } undef, ptr %311, 0
  %314 = insertvalue { ptr, ptr, i64 } %313, ptr %312, 1
  %315 = insertvalue { ptr, ptr, i64 } %314, i64 0, 2
  %316 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 2
  %317 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 3, 0
  %318 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 3, 1
  %319 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 4, 0
  %320 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 4, 1
  %321 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %311, 0
  %322 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %321, ptr %312, 1
  %323 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %322, i64 %319, 2
  %324 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %323, i64 %40, 3, 0
  %325 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %324, i64 %319, 4, 0
  %326 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %325, i64 %38, 3, 1
  %327 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %326, i64 1, 4, 1
  %328 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 0
  %329 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 1
  %330 = insertvalue { ptr, ptr, i64 } undef, ptr %328, 0
  %331 = insertvalue { ptr, ptr, i64 } %330, ptr %329, 1
  %332 = insertvalue { ptr, ptr, i64 } %331, i64 0, 2
  %333 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 2
  %334 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 3, 0
  %335 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 3, 1
  %336 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 4, 0
  %337 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 4, 1
  %338 = add i64 %336, 2
  %339 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %328, 0
  %340 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %339, ptr %329, 1
  %341 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %340, i64 %338, 2
  %342 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %341, i64 %40, 3, 0
  %343 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %342, i64 %336, 4, 0
  %344 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %343, i64 %38, 3, 1
  %345 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %344, i64 1, 4, 1
  %346 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 0
  %347 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 1
  %348 = insertvalue { ptr, ptr, i64 } undef, ptr %346, 0
  %349 = insertvalue { ptr, ptr, i64 } %348, ptr %347, 1
  %350 = insertvalue { ptr, ptr, i64 } %349, i64 0, 2
  %351 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 2
  %352 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 3, 0
  %353 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 3, 1
  %354 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 4, 0
  %355 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 4, 1
  %356 = mul i64 %354, 2
  %357 = add i64 %356, 1
  %358 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %346, 0
  %359 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %358, ptr %347, 1
  %360 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %359, i64 %357, 2
  %361 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %360, i64 %40, 3, 0
  %362 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %361, i64 %354, 4, 0
  %363 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %362, i64 %38, 3, 1
  %364 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %363, i64 1, 4, 1
  %365 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 0
  %366 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 1
  %367 = insertvalue { ptr, ptr, i64 } undef, ptr %365, 0
  %368 = insertvalue { ptr, ptr, i64 } %367, ptr %366, 1
  %369 = insertvalue { ptr, ptr, i64 } %368, i64 0, 2
  %370 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 2
  %371 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 3, 0
  %372 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 3, 1
  %373 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 4, 0
  %374 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 4, 1
  %375 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %365, 0
  %376 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %375, ptr %366, 1
  %377 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %376, i64 1, 2
  %378 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %377, i64 %40, 3, 0
  %379 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %378, i64 %373, 4, 0
  %380 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %379, i64 %38, 3, 1
  %381 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %380, i64 1, 4, 1
  %382 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 0
  %383 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 1
  %384 = insertvalue { ptr, ptr, i64 } undef, ptr %382, 0
  %385 = insertvalue { ptr, ptr, i64 } %384, ptr %383, 1
  %386 = insertvalue { ptr, ptr, i64 } %385, i64 0, 2
  %387 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 2
  %388 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 3, 0
  %389 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 3, 1
  %390 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 4, 0
  %391 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 4, 1
  %392 = add i64 %390, 1
  %393 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %382, 0
  %394 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %393, ptr %383, 1
  %395 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %394, i64 %392, 2
  %396 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %395, i64 %40, 3, 0
  %397 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %396, i64 %390, 4, 0
  %398 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %397, i64 %38, 3, 1
  %399 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %398, i64 1, 4, 1
  br label %400

400:                                              ; preds = %443, %267
  %401 = phi i64 [ %444, %443 ], [ 0, %267 ]
  %402 = icmp slt i64 %401, %40
  br i1 %402, label %403, label %445

403:                                              ; preds = %400
  br label %404

404:                                              ; preds = %407, %403
  %405 = phi i64 [ %442, %407 ], [ 0, %403 ]
  %406 = icmp slt i64 %405, %38
  br i1 %406, label %407, label %443

407:                                              ; preds = %404
  %408 = getelementptr double, ptr %294, i64 %303
  %409 = mul i64 %401, %301
  %410 = add i64 %409, %405
  %411 = getelementptr double, ptr %408, i64 %410
  %412 = load double, ptr %411, align 8
  %413 = getelementptr double, ptr %312, i64 %319
  %414 = mul i64 %401, %319
  %415 = add i64 %414, %405
  %416 = getelementptr double, ptr %413, i64 %415
  %417 = load double, ptr %416, align 8
  %418 = getelementptr double, ptr %329, i64 %338
  %419 = mul i64 %401, %336
  %420 = add i64 %419, %405
  %421 = getelementptr double, ptr %418, i64 %420
  %422 = load double, ptr %421, align 8
  %423 = getelementptr double, ptr %347, i64 %357
  %424 = mul i64 %401, %354
  %425 = add i64 %424, %405
  %426 = getelementptr double, ptr %423, i64 %425
  %427 = load double, ptr %426, align 8
  %428 = getelementptr double, ptr %366, i64 1
  %429 = mul i64 %401, %373
  %430 = add i64 %429, %405
  %431 = getelementptr double, ptr %428, i64 %430
  %432 = load double, ptr %431, align 8
  %433 = fadd double %412, %417
  %434 = fadd double %433, %422
  %435 = fadd double %434, %427
  %436 = fadd double %435, %432
  %437 = fmul double %436, 2.000000e-01
  %438 = getelementptr double, ptr %383, i64 %392
  %439 = mul i64 %401, %390
  %440 = add i64 %439, %405
  %441 = getelementptr double, ptr %438, i64 %440
  store double %437, ptr %441, align 8
  %442 = add i64 %405, 1
  br label %404

443:                                              ; preds = %404
  %444 = add i64 %401, 1
  br label %400

445:                                              ; preds = %400
  %446 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 0
  %447 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 1
  %448 = insertvalue { ptr, ptr, i64 } undef, ptr %446, 0
  %449 = insertvalue { ptr, ptr, i64 } %448, ptr %447, 1
  %450 = insertvalue { ptr, ptr, i64 } %449, i64 0, 2
  %451 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 2
  %452 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 3, 0
  %453 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 3, 1
  %454 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 4, 0
  %455 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 4, 1
  %456 = add i64 %454, 1
  %457 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %446, 0
  %458 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %457, ptr %447, 1
  %459 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %458, i64 %456, 2
  %460 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %459, i64 %40, 3, 0
  %461 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %460, i64 %454, 4, 0
  %462 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %461, i64 %38, 3, 1
  %463 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %462, i64 1, 4, 1
  %464 = call ptr @llvm.stacksave.p0()
  %465 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %399, ptr %465, align 8
  %466 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %465, 1
  %467 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %463, ptr %467, align 8
  %468 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %467, 1
  %469 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %466, ptr %469, align 8
  %470 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %468, ptr %470, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), ptr %469, ptr %470)
  call void @llvm.stackrestore.p0(ptr %464)
  %471 = add i64 %110, 1
  br label %109

472:                                              ; preds = %109
  %473 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 3, 0
  %474 = mul i64 %473, 1
  %475 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 3, 1
  %476 = mul i64 %474, %475
  %477 = mul i64 %476, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %478 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 1
  %479 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, 2
  %480 = getelementptr double, ptr %478, i64 %479
  %481 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, 1
  %482 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, 2
  %483 = getelementptr double, ptr %481, i64 %482
  call void @llvm.memcpy.p0.p0.i64(ptr %483, ptr %480, i64 %477, i1 false)
  %484 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 3, 0
  %485 = mul i64 %484, 1
  %486 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 3, 1
  %487 = mul i64 %485, %486
  %488 = mul i64 %487, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %489 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 1
  %490 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, 2
  %491 = getelementptr double, ptr %489, i64 %490
  %492 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, 1
  %493 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, 2
  %494 = getelementptr double, ptr %492, i64 %493
  call void @llvm.memcpy.p0.p0.i64(ptr %494, ptr %491, i64 %488, i1 false)
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
