; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @malloc(i64)

declare void @free(ptr)

declare void @memrefCopy(i64, ptr, ptr)

define void @kernel_jacobi_1d_impl(i32 %0, i32 %1, ptr %2, ptr %3, i64 %4, i64 %5, i64 %6, ptr %7, ptr %8, i64 %9, i64 %10, i64 %11) {
  %13 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %2, 0
  %14 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %13, ptr %3, 1
  %15 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %14, i64 %4, 2
  %16 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %15, i64 %5, 3, 0
  %17 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %16, i64 %6, 4, 0
  %18 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %7, 0
  %19 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %18, ptr %8, 1
  %20 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, i64 %9, 2
  %21 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, i64 %10, 3, 0
  %22 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %21, i64 %11, 4, 0
  %23 = sext i32 %1 to i64
  %24 = sext i32 %0 to i64
  %25 = add i64 %23, -1
  %26 = add i64 %23, -1
  %27 = sub i64 %26, 1
  %28 = sub i64 %25, 1
  %29 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %22, 3
  %30 = alloca [1 x i64], i64 1, align 8
  store [1 x i64] %29, ptr %30, align 8
  %31 = getelementptr [1 x i64], ptr %30, i32 0, i32 0
  %32 = load i64, ptr %31, align 8
  %33 = getelementptr double, ptr null, i64 %32
  %34 = ptrtoint ptr %33 to i64
  %35 = add i64 %34, 64
  %36 = call ptr @malloc(i64 %35)
  %37 = ptrtoint ptr %36 to i64
  %38 = add i64 %37, 63
  %39 = urem i64 %38, 64
  %40 = sub i64 %38, %39
  %41 = inttoptr i64 %40 to ptr
  %42 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %36, 0
  %43 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %42, ptr %41, 1
  %44 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %43, i64 0, 2
  %45 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %44, i64 %32, 3, 0
  %46 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %45, i64 1, 4, 0
  %47 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %22, 3, 0
  %48 = mul i64 %47, 1
  %49 = mul i64 %48, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %50 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %22, 1
  %51 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %22, 2
  %52 = getelementptr double, ptr %50, i64 %51
  %53 = getelementptr double, ptr %41, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %53, ptr %52, i64 %49, i1 false)
  %54 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %17, 3
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
  %72 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %17, 3, 0
  %73 = mul i64 %72, 1
  %74 = mul i64 %73, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %75 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %17, 1
  %76 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %17, 2
  %77 = getelementptr double, ptr %75, i64 %76
  %78 = getelementptr double, ptr %66, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %78, ptr %77, i64 %74, i1 false)
  br label %79

79:                                               ; preds = %246, %12
  %80 = phi i64 [ %267, %246 ], [ 0, %12 ]
  %81 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ %81, %246 ], [ %46, %12 ]
  %82 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ %82, %246 ], [ %71, %12 ]
  %83 = icmp slt i64 %80, %24
  br i1 %83, label %84, label %268

84:                                               ; preds = %79
  %85 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %82, 0
  %86 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %82, 1
  %87 = insertvalue { ptr, ptr, i64 } undef, ptr %85, 0
  %88 = insertvalue { ptr, ptr, i64 } %87, ptr %86, 1
  %89 = insertvalue { ptr, ptr, i64 } %88, i64 0, 2
  %90 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %82, 2
  %91 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %82, 3, 0
  %92 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %82, 4, 0
  %93 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %85, 0
  %94 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %93, ptr %86, 1
  %95 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %94, i64 0, 2
  %96 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %95, i64 %27, 3, 0
  %97 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %96, i64 1, 4, 0
  %98 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %82, 0
  %99 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %82, 1
  %100 = insertvalue { ptr, ptr, i64 } undef, ptr %98, 0
  %101 = insertvalue { ptr, ptr, i64 } %100, ptr %99, 1
  %102 = insertvalue { ptr, ptr, i64 } %101, i64 0, 2
  %103 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %82, 2
  %104 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %82, 3, 0
  %105 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %82, 4, 0
  %106 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %98, 0
  %107 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %106, ptr %99, 1
  %108 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %107, i64 1, 2
  %109 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %108, i64 %27, 3, 0
  %110 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %109, i64 1, 4, 0
  %111 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %82, 0
  %112 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %82, 1
  %113 = insertvalue { ptr, ptr, i64 } undef, ptr %111, 0
  %114 = insertvalue { ptr, ptr, i64 } %113, ptr %112, 1
  %115 = insertvalue { ptr, ptr, i64 } %114, i64 0, 2
  %116 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %82, 2
  %117 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %82, 3, 0
  %118 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %82, 4, 0
  %119 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %111, 0
  %120 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %119, ptr %112, 1
  %121 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %120, i64 2, 2
  %122 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %121, i64 %27, 3, 0
  %123 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %122, i64 1, 4, 0
  %124 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %81, 0
  %125 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %81, 1
  %126 = insertvalue { ptr, ptr, i64 } undef, ptr %124, 0
  %127 = insertvalue { ptr, ptr, i64 } %126, ptr %125, 1
  %128 = insertvalue { ptr, ptr, i64 } %127, i64 0, 2
  %129 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %81, 2
  %130 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %81, 3, 0
  %131 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %81, 4, 0
  %132 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %124, 0
  %133 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %132, ptr %125, 1
  %134 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %133, i64 1, 2
  %135 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %134, i64 %27, 3, 0
  %136 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %135, i64 1, 4, 0
  br label %137

137:                                              ; preds = %140, %84
  %138 = phi i64 [ %154, %140 ], [ 0, %84 ]
  %139 = icmp slt i64 %138, %27
  br i1 %139, label %140, label %155

140:                                              ; preds = %137
  %141 = getelementptr double, ptr %86, i64 %138
  %142 = load double, ptr %141, align 8
  %143 = getelementptr double, ptr %99, i32 1
  %144 = getelementptr double, ptr %143, i64 %138
  %145 = load double, ptr %144, align 8
  %146 = getelementptr double, ptr %112, i32 2
  %147 = getelementptr double, ptr %146, i64 %138
  %148 = load double, ptr %147, align 8
  %149 = fadd double %142, %145
  %150 = fadd double %149, %148
  %151 = fmul double %150, 3.333300e-01
  %152 = getelementptr double, ptr %125, i32 1
  %153 = getelementptr double, ptr %152, i64 %138
  store double %151, ptr %153, align 8
  %154 = add i64 %138, 1
  br label %137

155:                                              ; preds = %137
  %156 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %81, 0
  %157 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %81, 1
  %158 = insertvalue { ptr, ptr, i64 } undef, ptr %156, 0
  %159 = insertvalue { ptr, ptr, i64 } %158, ptr %157, 1
  %160 = insertvalue { ptr, ptr, i64 } %159, i64 0, 2
  %161 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %81, 2
  %162 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %81, 3, 0
  %163 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %81, 4, 0
  %164 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %156, 0
  %165 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %164, ptr %157, 1
  %166 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %165, i64 1, 2
  %167 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %166, i64 %27, 3, 0
  %168 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %167, i64 1, 4, 0
  %169 = call ptr @llvm.stacksave.p0()
  %170 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %136, ptr %170, align 8
  %171 = insertvalue { i64, ptr } { i64 1, ptr undef }, ptr %170, 1
  %172 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %168, ptr %172, align 8
  %173 = insertvalue { i64, ptr } { i64 1, ptr undef }, ptr %172, 1
  %174 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %171, ptr %174, align 8
  %175 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %173, ptr %175, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), ptr %174, ptr %175)
  call void @llvm.stackrestore.p0(ptr %169)
  %176 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %81, 0
  %177 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %81, 1
  %178 = insertvalue { ptr, ptr, i64 } undef, ptr %176, 0
  %179 = insertvalue { ptr, ptr, i64 } %178, ptr %177, 1
  %180 = insertvalue { ptr, ptr, i64 } %179, i64 0, 2
  %181 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %81, 2
  %182 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %81, 3, 0
  %183 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %81, 4, 0
  %184 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %176, 0
  %185 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %184, ptr %177, 1
  %186 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %185, i64 0, 2
  %187 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %186, i64 %28, 3, 0
  %188 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %187, i64 1, 4, 0
  %189 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %81, 0
  %190 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %81, 1
  %191 = insertvalue { ptr, ptr, i64 } undef, ptr %189, 0
  %192 = insertvalue { ptr, ptr, i64 } %191, ptr %190, 1
  %193 = insertvalue { ptr, ptr, i64 } %192, i64 0, 2
  %194 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %81, 2
  %195 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %81, 3, 0
  %196 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %81, 4, 0
  %197 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %189, 0
  %198 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %197, ptr %190, 1
  %199 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %198, i64 1, 2
  %200 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %199, i64 %28, 3, 0
  %201 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %200, i64 1, 4, 0
  %202 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %81, 0
  %203 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %81, 1
  %204 = insertvalue { ptr, ptr, i64 } undef, ptr %202, 0
  %205 = insertvalue { ptr, ptr, i64 } %204, ptr %203, 1
  %206 = insertvalue { ptr, ptr, i64 } %205, i64 0, 2
  %207 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %81, 2
  %208 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %81, 3, 0
  %209 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %81, 4, 0
  %210 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %202, 0
  %211 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %210, ptr %203, 1
  %212 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %211, i64 2, 2
  %213 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %212, i64 %28, 3, 0
  %214 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %213, i64 1, 4, 0
  %215 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %82, 0
  %216 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %82, 1
  %217 = insertvalue { ptr, ptr, i64 } undef, ptr %215, 0
  %218 = insertvalue { ptr, ptr, i64 } %217, ptr %216, 1
  %219 = insertvalue { ptr, ptr, i64 } %218, i64 0, 2
  %220 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %82, 2
  %221 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %82, 3, 0
  %222 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %82, 4, 0
  %223 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %215, 0
  %224 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %223, ptr %216, 1
  %225 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %224, i64 1, 2
  %226 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %225, i64 %28, 3, 0
  %227 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %226, i64 1, 4, 0
  br label %228

228:                                              ; preds = %231, %155
  %229 = phi i64 [ %245, %231 ], [ 0, %155 ]
  %230 = icmp slt i64 %229, %28
  br i1 %230, label %231, label %246

231:                                              ; preds = %228
  %232 = getelementptr double, ptr %177, i64 %229
  %233 = load double, ptr %232, align 8
  %234 = getelementptr double, ptr %190, i32 1
  %235 = getelementptr double, ptr %234, i64 %229
  %236 = load double, ptr %235, align 8
  %237 = getelementptr double, ptr %203, i32 2
  %238 = getelementptr double, ptr %237, i64 %229
  %239 = load double, ptr %238, align 8
  %240 = fadd double %233, %236
  %241 = fadd double %240, %239
  %242 = fmul double %241, 3.333300e-01
  %243 = getelementptr double, ptr %216, i32 1
  %244 = getelementptr double, ptr %243, i64 %229
  store double %242, ptr %244, align 8
  %245 = add i64 %229, 1
  br label %228

246:                                              ; preds = %228
  %247 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %82, 0
  %248 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %82, 1
  %249 = insertvalue { ptr, ptr, i64 } undef, ptr %247, 0
  %250 = insertvalue { ptr, ptr, i64 } %249, ptr %248, 1
  %251 = insertvalue { ptr, ptr, i64 } %250, i64 0, 2
  %252 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %82, 2
  %253 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %82, 3, 0
  %254 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %82, 4, 0
  %255 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %247, 0
  %256 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %255, ptr %248, 1
  %257 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %256, i64 1, 2
  %258 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %257, i64 %28, 3, 0
  %259 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %258, i64 1, 4, 0
  %260 = call ptr @llvm.stacksave.p0()
  %261 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %227, ptr %261, align 8
  %262 = insertvalue { i64, ptr } { i64 1, ptr undef }, ptr %261, 1
  %263 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %259, ptr %263, align 8
  %264 = insertvalue { i64, ptr } { i64 1, ptr undef }, ptr %263, 1
  %265 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %262, ptr %265, align 8
  %266 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %264, ptr %266, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), ptr %265, ptr %266)
  call void @llvm.stackrestore.p0(ptr %260)
  %267 = add i64 %80, 1
  br label %79

268:                                              ; preds = %79
  %269 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %82, 3, 0
  %270 = mul i64 %269, 1
  %271 = mul i64 %270, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %272 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %82, 1
  %273 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %82, 2
  %274 = getelementptr double, ptr %272, i64 %273
  %275 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %17, 1
  %276 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %17, 2
  %277 = getelementptr double, ptr %275, i64 %276
  call void @llvm.memcpy.p0.p0.i64(ptr %277, ptr %274, i64 %271, i1 false)
  %278 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %81, 3, 0
  %279 = mul i64 %278, 1
  %280 = mul i64 %279, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %281 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %81, 1
  %282 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %81, 2
  %283 = getelementptr double, ptr %281, i64 %282
  %284 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %22, 1
  %285 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %22, 2
  %286 = getelementptr double, ptr %284, i64 %285
  call void @llvm.memcpy.p0.p0.i64(ptr %286, ptr %283, i64 %280, i1 false)
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
