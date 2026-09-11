; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @malloc(i64)

declare void @free(ptr)

declare void @memrefCopy(i64, ptr, ptr)

define void @kernel_cholesky_impl(i32 %0, ptr %1, ptr %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7) {
  %9 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %1, 0
  %10 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %9, ptr %2, 1
  %11 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %10, i64 %3, 2
  %12 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %11, i64 %4, 3, 0
  %13 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, i64 %6, 4, 0
  %14 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, i64 %5, 3, 1
  %15 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %14, i64 %7, 4, 1
  %16 = sext i32 %0 to i64
  %17 = sub i64 %16, 1
  %18 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 3
  %19 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %18, ptr %19, align 8
  %20 = getelementptr [2 x i64], ptr %19, i32 0, i32 0
  %21 = load i64, ptr %20, align 8
  %22 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 3
  %23 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %22, ptr %23, align 8
  %24 = getelementptr [2 x i64], ptr %23, i32 0, i32 1
  %25 = load i64, ptr %24, align 8
  %26 = mul i64 %25, %21
  %27 = getelementptr double, ptr null, i64 %26
  %28 = ptrtoint ptr %27 to i64
  %29 = add i64 %28, 64
  %30 = call ptr @malloc(i64 %29)
  %31 = ptrtoint ptr %30 to i64
  %32 = add i64 %31, 63
  %33 = urem i64 %32, 64
  %34 = sub i64 %32, %33
  %35 = inttoptr i64 %34 to ptr
  %36 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %30, 0
  %37 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, ptr %35, 1
  %38 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, i64 0, 2
  %39 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, i64 %21, 3, 0
  %40 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, i64 %25, 3, 1
  %41 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, i64 %25, 4, 0
  %42 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, i64 1, 4, 1
  %43 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 3, 0
  %44 = mul i64 %43, 1
  %45 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 3, 1
  %46 = mul i64 %44, %45
  %47 = mul i64 %46, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %48 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 1
  %49 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 2
  %50 = getelementptr double, ptr %48, i64 %49
  %51 = getelementptr double, ptr %35, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %51, ptr %50, i64 %47, i1 false)
  br label %52

52:                                               ; preds = %272, %8
  %53 = phi i64 [ %302, %272 ], [ 0, %8 ]
  %54 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %60, %272 ], [ %42, %8 ]
  %55 = icmp slt i64 %53, %16
  br i1 %55, label %56, label %303

56:                                               ; preds = %52
  %57 = sub i64 %53, 1
  br label %58

58:                                               ; preds = %168, %56
  %59 = phi i64 [ %204, %168 ], [ 0, %56 ]
  %60 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %60, %168 ], [ %54, %56 ]
  %61 = icmp slt i64 %59, %53
  br i1 %61, label %62, label %205

62:                                               ; preds = %58
  %63 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 0
  %64 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 1
  %65 = insertvalue { ptr, ptr, i64 } undef, ptr %63, 0
  %66 = insertvalue { ptr, ptr, i64 } %65, ptr %64, 1
  %67 = insertvalue { ptr, ptr, i64 } %66, i64 0, 2
  %68 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 2
  %69 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 3, 0
  %70 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 3, 1
  %71 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 4, 0
  %72 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 4, 1
  %73 = mul i64 %53, %71
  %74 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %63, 0
  %75 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %74, ptr %64, 1
  %76 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %75, i64 %73, 2
  %77 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %76, i64 %57, 3, 0
  %78 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %77, i64 1, 4, 0
  %79 = getelementptr double, ptr null, i64 %57
  %80 = ptrtoint ptr %79 to i64
  %81 = add i64 %80, 64
  %82 = call ptr @malloc(i64 %81)
  %83 = ptrtoint ptr %82 to i64
  %84 = add i64 %83, 63
  %85 = urem i64 %84, 64
  %86 = sub i64 %84, %85
  %87 = inttoptr i64 %86 to ptr
  %88 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %82, 0
  %89 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %88, ptr %87, 1
  %90 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %89, i64 0, 2
  %91 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %90, i64 %57, 3, 0
  %92 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %91, i64 1, 4, 0
  %93 = call ptr @llvm.stacksave.p0()
  %94 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %78, ptr %94, align 8
  %95 = insertvalue { i64, ptr } { i64 1, ptr undef }, ptr %94, 1
  %96 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %92, ptr %96, align 8
  %97 = insertvalue { i64, ptr } { i64 1, ptr undef }, ptr %96, 1
  %98 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %95, ptr %98, align 8
  %99 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %97, ptr %99, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), ptr %98, ptr %99)
  call void @llvm.stackrestore.p0(ptr %93)
  %100 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 0
  %101 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 1
  %102 = insertvalue { ptr, ptr, i64 } undef, ptr %100, 0
  %103 = insertvalue { ptr, ptr, i64 } %102, ptr %101, 1
  %104 = insertvalue { ptr, ptr, i64 } %103, i64 0, 2
  %105 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 2
  %106 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 3, 0
  %107 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 3, 1
  %108 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 4, 0
  %109 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 4, 1
  %110 = mul i64 %59, %108
  %111 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %100, 0
  %112 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %111, ptr %101, 1
  %113 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %112, i64 %110, 2
  %114 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %113, i64 %57, 3, 0
  %115 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %114, i64 1, 4, 0
  %116 = getelementptr double, ptr null, i64 %57
  %117 = ptrtoint ptr %116 to i64
  %118 = add i64 %117, 64
  %119 = call ptr @malloc(i64 %118)
  %120 = ptrtoint ptr %119 to i64
  %121 = add i64 %120, 63
  %122 = urem i64 %121, 64
  %123 = sub i64 %121, %122
  %124 = inttoptr i64 %123 to ptr
  %125 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %119, 0
  %126 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %125, ptr %124, 1
  %127 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %126, i64 0, 2
  %128 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %127, i64 %57, 3, 0
  %129 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %128, i64 1, 4, 0
  %130 = call ptr @llvm.stacksave.p0()
  %131 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %115, ptr %131, align 8
  %132 = insertvalue { i64, ptr } { i64 1, ptr undef }, ptr %131, 1
  %133 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %129, ptr %133, align 8
  %134 = insertvalue { i64, ptr } { i64 1, ptr undef }, ptr %133, 1
  %135 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %132, ptr %135, align 8
  %136 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %134, ptr %136, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), ptr %135, ptr %136)
  call void @llvm.stackrestore.p0(ptr %130)
  %137 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 0
  %138 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 1
  %139 = insertvalue { ptr, ptr, i64 } undef, ptr %137, 0
  %140 = insertvalue { ptr, ptr, i64 } %139, ptr %138, 1
  %141 = insertvalue { ptr, ptr, i64 } %140, i64 0, 2
  %142 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 2
  %143 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 3, 0
  %144 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 3, 1
  %145 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 4, 0
  %146 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 4, 1
  %147 = mul i64 %53, %145
  %148 = add i64 %147, %59
  %149 = insertvalue { ptr, ptr, i64 } undef, ptr %137, 0
  %150 = insertvalue { ptr, ptr, i64 } %149, ptr %138, 1
  %151 = insertvalue { ptr, ptr, i64 } %150, i64 %148, 2
  br label %152

152:                                              ; preds = %155, %62
  %153 = phi i64 [ %167, %155 ], [ 0, %62 ]
  %154 = icmp slt i64 %153, %57
  br i1 %154, label %155, label %168

155:                                              ; preds = %152
  %156 = getelementptr double, ptr %87, i64 %153
  %157 = load double, ptr %156, align 8
  %158 = getelementptr double, ptr %124, i64 %153
  %159 = load double, ptr %158, align 8
  %160 = getelementptr double, ptr %138, i64 %148
  %161 = load double, ptr %160, align 8
  %162 = fmul double %157, %159
  %163 = fsub double %161, %162
  %164 = icmp slt i64 %153, %59
  %165 = select i1 %164, double %163, double %161
  %166 = getelementptr double, ptr %138, i64 %148
  store double %165, ptr %166, align 8
  %167 = add i64 %153, 1
  br label %152

168:                                              ; preds = %152
  %169 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 0
  %170 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 1
  %171 = insertvalue { ptr, ptr, i64 } undef, ptr %169, 0
  %172 = insertvalue { ptr, ptr, i64 } %171, ptr %170, 1
  %173 = insertvalue { ptr, ptr, i64 } %172, i64 0, 2
  %174 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 2
  %175 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 3, 0
  %176 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 3, 1
  %177 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 4, 0
  %178 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 4, 1
  %179 = mul i64 %53, %177
  %180 = add i64 %179, %59
  %181 = insertvalue { ptr, ptr, i64 } undef, ptr %169, 0
  %182 = insertvalue { ptr, ptr, i64 } %181, ptr %170, 1
  %183 = insertvalue { ptr, ptr, i64 } %182, i64 %180, 2
  %184 = getelementptr double, ptr %138, i64 %148
  %185 = getelementptr double, ptr %170, i64 %180
  call void @llvm.memcpy.p0.p0.i64(ptr %185, ptr %184, i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), i1 false)
  %186 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 1
  %187 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 4, 0
  %188 = mul i64 %59, %187
  %189 = add i64 %188, %59
  %190 = getelementptr double, ptr %186, i64 %189
  %191 = load double, ptr %190, align 8
  %192 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 1
  %193 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 4, 0
  %194 = mul i64 %53, %193
  %195 = add i64 %194, %59
  %196 = getelementptr double, ptr %192, i64 %195
  %197 = load double, ptr %196, align 8
  %198 = fdiv double %197, %191
  %199 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 1
  %200 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 4, 0
  %201 = mul i64 %53, %200
  %202 = add i64 %201, %59
  %203 = getelementptr double, ptr %199, i64 %202
  store double %198, ptr %203, align 8
  %204 = add i64 %59, 1
  br label %58

205:                                              ; preds = %58
  %206 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 0
  %207 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 1
  %208 = insertvalue { ptr, ptr, i64 } undef, ptr %206, 0
  %209 = insertvalue { ptr, ptr, i64 } %208, ptr %207, 1
  %210 = insertvalue { ptr, ptr, i64 } %209, i64 0, 2
  %211 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 2
  %212 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 3, 0
  %213 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 3, 1
  %214 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 4, 0
  %215 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 4, 1
  %216 = mul i64 %53, %214
  %217 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %206, 0
  %218 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %217, ptr %207, 1
  %219 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %218, i64 %216, 2
  %220 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %219, i64 %17, 3, 0
  %221 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %220, i64 1, 4, 0
  %222 = getelementptr double, ptr null, i64 %17
  %223 = ptrtoint ptr %222 to i64
  %224 = add i64 %223, 64
  %225 = call ptr @malloc(i64 %224)
  %226 = ptrtoint ptr %225 to i64
  %227 = add i64 %226, 63
  %228 = urem i64 %227, 64
  %229 = sub i64 %227, %228
  %230 = inttoptr i64 %229 to ptr
  %231 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %225, 0
  %232 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %231, ptr %230, 1
  %233 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %232, i64 0, 2
  %234 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %233, i64 %17, 3, 0
  %235 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %234, i64 1, 4, 0
  %236 = call ptr @llvm.stacksave.p0()
  %237 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %221, ptr %237, align 8
  %238 = insertvalue { i64, ptr } { i64 1, ptr undef }, ptr %237, 1
  %239 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %235, ptr %239, align 8
  %240 = insertvalue { i64, ptr } { i64 1, ptr undef }, ptr %239, 1
  %241 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %238, ptr %241, align 8
  %242 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %240, ptr %242, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), ptr %241, ptr %242)
  call void @llvm.stackrestore.p0(ptr %236)
  %243 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 0
  %244 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 1
  %245 = insertvalue { ptr, ptr, i64 } undef, ptr %243, 0
  %246 = insertvalue { ptr, ptr, i64 } %245, ptr %244, 1
  %247 = insertvalue { ptr, ptr, i64 } %246, i64 0, 2
  %248 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 2
  %249 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 3, 0
  %250 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 3, 1
  %251 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 4, 0
  %252 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 4, 1
  %253 = mul i64 %53, %251
  %254 = add i64 %53, %253
  %255 = insertvalue { ptr, ptr, i64 } undef, ptr %243, 0
  %256 = insertvalue { ptr, ptr, i64 } %255, ptr %244, 1
  %257 = insertvalue { ptr, ptr, i64 } %256, i64 %254, 2
  br label %258

258:                                              ; preds = %261, %205
  %259 = phi i64 [ %271, %261 ], [ 0, %205 ]
  %260 = icmp slt i64 %259, %17
  br i1 %260, label %261, label %272

261:                                              ; preds = %258
  %262 = getelementptr double, ptr %230, i64 %259
  %263 = load double, ptr %262, align 8
  %264 = getelementptr double, ptr %244, i64 %254
  %265 = load double, ptr %264, align 8
  %266 = fmul double %263, %263
  %267 = fsub double %265, %266
  %268 = icmp slt i64 %259, %53
  %269 = select i1 %268, double %267, double %265
  %270 = getelementptr double, ptr %244, i64 %254
  store double %269, ptr %270, align 8
  %271 = add i64 %259, 1
  br label %258

272:                                              ; preds = %258
  %273 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 0
  %274 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 1
  %275 = insertvalue { ptr, ptr, i64 } undef, ptr %273, 0
  %276 = insertvalue { ptr, ptr, i64 } %275, ptr %274, 1
  %277 = insertvalue { ptr, ptr, i64 } %276, i64 0, 2
  %278 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 2
  %279 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 3, 0
  %280 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 3, 1
  %281 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 4, 0
  %282 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 4, 1
  %283 = mul i64 %53, %281
  %284 = add i64 %53, %283
  %285 = insertvalue { ptr, ptr, i64 } undef, ptr %273, 0
  %286 = insertvalue { ptr, ptr, i64 } %285, ptr %274, 1
  %287 = insertvalue { ptr, ptr, i64 } %286, i64 %284, 2
  %288 = getelementptr double, ptr %244, i64 %254
  %289 = getelementptr double, ptr %274, i64 %284
  call void @llvm.memcpy.p0.p0.i64(ptr %289, ptr %288, i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), i1 false)
  %290 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 1
  %291 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 4, 0
  %292 = mul i64 %53, %291
  %293 = add i64 %292, %53
  %294 = getelementptr double, ptr %290, i64 %293
  %295 = load double, ptr %294, align 8
  %296 = call double @llvm.sqrt.f64(double %295)
  %297 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 1
  %298 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 4, 0
  %299 = mul i64 %53, %298
  %300 = add i64 %299, %53
  %301 = getelementptr double, ptr %297, i64 %300
  store double %296, ptr %301, align 8
  %302 = add i64 %53, 1
  br label %52

303:                                              ; preds = %52
  %304 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %54, 3, 0
  %305 = mul i64 %304, 1
  %306 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %54, 3, 1
  %307 = mul i64 %305, %306
  %308 = mul i64 %307, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %309 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %54, 1
  %310 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %54, 2
  %311 = getelementptr double, ptr %309, i64 %310
  %312 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 1
  %313 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 2
  %314 = getelementptr double, ptr %312, i64 %313
  call void @llvm.memcpy.p0.p0.i64(ptr %314, ptr %311, i64 %308, i1 false)
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare ptr @llvm.stacksave.p0() #1

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.stackrestore.p0(ptr) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.sqrt.f64(double) #2

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #1 = { nocallback nofree nosync nounwind willreturn }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
