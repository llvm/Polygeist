; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @malloc(i64)

declare void @free(ptr)

declare void @memrefCopy(i64, ptr, ptr)

define void @kernel_lu_impl(i32 %0, ptr %1, ptr %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7) {
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

52:                                               ; preds = %325, %8
  %53 = phi i64 [ %349, %325 ], [ 0, %8 ]
  %54 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %60, %325 ], [ %42, %8 ]
  %55 = icmp slt i64 %53, %16
  br i1 %55, label %56, label %350

56:                                               ; preds = %52
  %57 = sub i64 %53, 1
  br label %58

58:                                               ; preds = %167, %56
  %59 = phi i64 [ %203, %167 ], [ 0, %56 ]
  %60 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %60, %167 ], [ %54, %56 ]
  %61 = icmp slt i64 %59, %53
  br i1 %61, label %62, label %204

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
  %110 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %100, 0
  %111 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %110, ptr %101, 1
  %112 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %111, i64 %59, 2
  %113 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %112, i64 %57, 3, 0
  %114 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %113, i64 %108, 4, 0
  %115 = getelementptr double, ptr null, i64 %57
  %116 = ptrtoint ptr %115 to i64
  %117 = add i64 %116, 64
  %118 = call ptr @malloc(i64 %117)
  %119 = ptrtoint ptr %118 to i64
  %120 = add i64 %119, 63
  %121 = urem i64 %120, 64
  %122 = sub i64 %120, %121
  %123 = inttoptr i64 %122 to ptr
  %124 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %118, 0
  %125 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %124, ptr %123, 1
  %126 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %125, i64 0, 2
  %127 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %126, i64 %57, 3, 0
  %128 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %127, i64 1, 4, 0
  %129 = call ptr @llvm.stacksave.p0()
  %130 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %114, ptr %130, align 8
  %131 = insertvalue { i64, ptr } { i64 1, ptr undef }, ptr %130, 1
  %132 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %128, ptr %132, align 8
  %133 = insertvalue { i64, ptr } { i64 1, ptr undef }, ptr %132, 1
  %134 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %131, ptr %134, align 8
  %135 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %133, ptr %135, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), ptr %134, ptr %135)
  call void @llvm.stackrestore.p0(ptr %129)
  %136 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 0
  %137 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 1
  %138 = insertvalue { ptr, ptr, i64 } undef, ptr %136, 0
  %139 = insertvalue { ptr, ptr, i64 } %138, ptr %137, 1
  %140 = insertvalue { ptr, ptr, i64 } %139, i64 0, 2
  %141 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 2
  %142 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 3, 0
  %143 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 3, 1
  %144 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 4, 0
  %145 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 4, 1
  %146 = mul i64 %53, %144
  %147 = add i64 %146, %59
  %148 = insertvalue { ptr, ptr, i64 } undef, ptr %136, 0
  %149 = insertvalue { ptr, ptr, i64 } %148, ptr %137, 1
  %150 = insertvalue { ptr, ptr, i64 } %149, i64 %147, 2
  br label %151

151:                                              ; preds = %154, %62
  %152 = phi i64 [ %166, %154 ], [ 0, %62 ]
  %153 = icmp slt i64 %152, %57
  br i1 %153, label %154, label %167

154:                                              ; preds = %151
  %155 = getelementptr double, ptr %87, i64 %152
  %156 = load double, ptr %155, align 8
  %157 = getelementptr double, ptr %123, i64 %152
  %158 = load double, ptr %157, align 8
  %159 = getelementptr double, ptr %137, i64 %147
  %160 = load double, ptr %159, align 8
  %161 = fmul double %156, %158
  %162 = fsub double %160, %161
  %163 = icmp slt i64 %152, %59
  %164 = select i1 %163, double %162, double %160
  %165 = getelementptr double, ptr %137, i64 %147
  store double %164, ptr %165, align 8
  %166 = add i64 %152, 1
  br label %151

167:                                              ; preds = %151
  %168 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 0
  %169 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 1
  %170 = insertvalue { ptr, ptr, i64 } undef, ptr %168, 0
  %171 = insertvalue { ptr, ptr, i64 } %170, ptr %169, 1
  %172 = insertvalue { ptr, ptr, i64 } %171, i64 0, 2
  %173 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 2
  %174 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 3, 0
  %175 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 3, 1
  %176 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 4, 0
  %177 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 4, 1
  %178 = mul i64 %53, %176
  %179 = add i64 %178, %59
  %180 = insertvalue { ptr, ptr, i64 } undef, ptr %168, 0
  %181 = insertvalue { ptr, ptr, i64 } %180, ptr %169, 1
  %182 = insertvalue { ptr, ptr, i64 } %181, i64 %179, 2
  %183 = getelementptr double, ptr %137, i64 %147
  %184 = getelementptr double, ptr %169, i64 %179
  call void @llvm.memcpy.p0.p0.i64(ptr %184, ptr %183, i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), i1 false)
  %185 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 1
  %186 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 4, 0
  %187 = mul i64 %59, %186
  %188 = add i64 %187, %59
  %189 = getelementptr double, ptr %185, i64 %188
  %190 = load double, ptr %189, align 8
  %191 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 1
  %192 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 4, 0
  %193 = mul i64 %53, %192
  %194 = add i64 %193, %59
  %195 = getelementptr double, ptr %191, i64 %194
  %196 = load double, ptr %195, align 8
  %197 = fdiv double %196, %190
  %198 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 1
  %199 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 4, 0
  %200 = mul i64 %53, %199
  %201 = add i64 %200, %59
  %202 = getelementptr double, ptr %198, i64 %201
  store double %197, ptr %202, align 8
  %203 = add i64 %59, 1
  br label %58

204:                                              ; preds = %58
  %205 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 0
  %206 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 1
  %207 = insertvalue { ptr, ptr, i64 } undef, ptr %205, 0
  %208 = insertvalue { ptr, ptr, i64 } %207, ptr %206, 1
  %209 = insertvalue { ptr, ptr, i64 } %208, i64 0, 2
  %210 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 2
  %211 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 3, 0
  %212 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 3, 1
  %213 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 4, 0
  %214 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 4, 1
  %215 = mul i64 %53, %213
  %216 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %205, 0
  %217 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %216, ptr %206, 1
  %218 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %217, i64 %215, 2
  %219 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %218, i64 %17, 3, 0
  %220 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %219, i64 1, 4, 0
  %221 = getelementptr double, ptr null, i64 %17
  %222 = ptrtoint ptr %221 to i64
  %223 = add i64 %222, 64
  %224 = call ptr @malloc(i64 %223)
  %225 = ptrtoint ptr %224 to i64
  %226 = add i64 %225, 63
  %227 = urem i64 %226, 64
  %228 = sub i64 %226, %227
  %229 = inttoptr i64 %228 to ptr
  %230 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %224, 0
  %231 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %230, ptr %229, 1
  %232 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %231, i64 0, 2
  %233 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %232, i64 %17, 3, 0
  %234 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %233, i64 1, 4, 0
  %235 = call ptr @llvm.stacksave.p0()
  %236 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %220, ptr %236, align 8
  %237 = insertvalue { i64, ptr } { i64 1, ptr undef }, ptr %236, 1
  %238 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %234, ptr %238, align 8
  %239 = insertvalue { i64, ptr } { i64 1, ptr undef }, ptr %238, 1
  %240 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %237, ptr %240, align 8
  %241 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %239, ptr %241, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), ptr %240, ptr %241)
  call void @llvm.stackrestore.p0(ptr %235)
  %242 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 0
  %243 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 1
  %244 = insertvalue { ptr, ptr, i64 } undef, ptr %242, 0
  %245 = insertvalue { ptr, ptr, i64 } %244, ptr %243, 1
  %246 = insertvalue { ptr, ptr, i64 } %245, i64 0, 2
  %247 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 2
  %248 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 3, 0
  %249 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 3, 1
  %250 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 4, 0
  %251 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 4, 1
  %252 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %242, 0
  %253 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %252, ptr %243, 1
  %254 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %253, i64 0, 2
  %255 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %254, i64 %17, 3, 0
  %256 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %255, i64 %250, 4, 0
  %257 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %256, i64 %16, 3, 1
  %258 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %257, i64 1, 4, 1
  %259 = mul i64 %16, %17
  %260 = getelementptr double, ptr null, i64 %259
  %261 = ptrtoint ptr %260 to i64
  %262 = add i64 %261, 64
  %263 = call ptr @malloc(i64 %262)
  %264 = ptrtoint ptr %263 to i64
  %265 = add i64 %264, 63
  %266 = urem i64 %265, 64
  %267 = sub i64 %265, %266
  %268 = inttoptr i64 %267 to ptr
  %269 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %263, 0
  %270 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %269, ptr %268, 1
  %271 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %270, i64 0, 2
  %272 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %271, i64 %17, 3, 0
  %273 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %272, i64 %16, 3, 1
  %274 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %273, i64 %16, 4, 0
  %275 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %274, i64 1, 4, 1
  %276 = call ptr @llvm.stacksave.p0()
  %277 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %258, ptr %277, align 8
  %278 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %277, 1
  %279 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %275, ptr %279, align 8
  %280 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %279, 1
  %281 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %278, ptr %281, align 8
  %282 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %280, ptr %282, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), ptr %281, ptr %282)
  call void @llvm.stackrestore.p0(ptr %276)
  %283 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 0
  %284 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 1
  %285 = insertvalue { ptr, ptr, i64 } undef, ptr %283, 0
  %286 = insertvalue { ptr, ptr, i64 } %285, ptr %284, 1
  %287 = insertvalue { ptr, ptr, i64 } %286, i64 0, 2
  %288 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 2
  %289 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 3, 0
  %290 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 3, 1
  %291 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 4, 0
  %292 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 4, 1
  %293 = mul i64 %53, %291
  %294 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %283, 0
  %295 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %294, ptr %284, 1
  %296 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %295, i64 %293, 2
  %297 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %296, i64 %16, 3, 0
  %298 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %297, i64 1, 4, 0
  br label %299

299:                                              ; preds = %323, %204
  %300 = phi i64 [ %324, %323 ], [ 0, %204 ]
  %301 = icmp slt i64 %300, %16
  br i1 %301, label %302, label %325

302:                                              ; preds = %299
  br label %303

303:                                              ; preds = %306, %302
  %304 = phi i64 [ %322, %306 ], [ 0, %302 ]
  %305 = icmp slt i64 %304, %17
  br i1 %305, label %306, label %323

306:                                              ; preds = %303
  %307 = getelementptr double, ptr %229, i64 %304
  %308 = load double, ptr %307, align 8
  %309 = mul i64 %304, %16
  %310 = add i64 %309, %300
  %311 = getelementptr double, ptr %268, i64 %310
  %312 = load double, ptr %311, align 8
  %313 = getelementptr double, ptr %284, i64 %293
  %314 = getelementptr double, ptr %313, i64 %300
  %315 = load double, ptr %314, align 8
  %316 = fmul double %308, %312
  %317 = fsub double %315, %316
  %318 = icmp slt i64 %304, %53
  %319 = select i1 %318, double %317, double %315
  %320 = getelementptr double, ptr %284, i64 %293
  %321 = getelementptr double, ptr %320, i64 %300
  store double %319, ptr %321, align 8
  %322 = add i64 %304, 1
  br label %303

323:                                              ; preds = %303
  %324 = add i64 %300, 1
  br label %299

325:                                              ; preds = %299
  %326 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 0
  %327 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 1
  %328 = insertvalue { ptr, ptr, i64 } undef, ptr %326, 0
  %329 = insertvalue { ptr, ptr, i64 } %328, ptr %327, 1
  %330 = insertvalue { ptr, ptr, i64 } %329, i64 0, 2
  %331 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 2
  %332 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 3, 0
  %333 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 3, 1
  %334 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 4, 0
  %335 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, 4, 1
  %336 = mul i64 %53, %334
  %337 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %326, 0
  %338 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %337, ptr %327, 1
  %339 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %338, i64 %336, 2
  %340 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %339, i64 %16, 3, 0
  %341 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %340, i64 1, 4, 0
  %342 = call ptr @llvm.stacksave.p0()
  %343 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %298, ptr %343, align 8
  %344 = insertvalue { i64, ptr } { i64 1, ptr undef }, ptr %343, 1
  %345 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %341, ptr %345, align 8
  %346 = insertvalue { i64, ptr } { i64 1, ptr undef }, ptr %345, 1
  %347 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %344, ptr %347, align 8
  %348 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %346, ptr %348, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), ptr %347, ptr %348)
  call void @llvm.stackrestore.p0(ptr %342)
  %349 = add i64 %53, 1
  br label %52

350:                                              ; preds = %52
  %351 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %54, 3, 0
  %352 = mul i64 %351, 1
  %353 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %54, 3, 1
  %354 = mul i64 %352, %353
  %355 = mul i64 %354, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %356 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %54, 1
  %357 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %54, 2
  %358 = getelementptr double, ptr %356, i64 %357
  %359 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 1
  %360 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 2
  %361 = getelementptr double, ptr %359, i64 %360
  call void @llvm.memcpy.p0.p0.i64(ptr %361, ptr %358, i64 %355, i1 false)
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
