; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @malloc(i64)

declare void @free(ptr)

define void @kernel_trisolv_impl(i32 %0, ptr %1, ptr %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, ptr %8, ptr %9, i64 %10, i64 %11, i64 %12, ptr %13, ptr %14, i64 %15, i64 %16, i64 %17) {
  %19 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %1, 0
  %20 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %19, ptr %2, 1
  %21 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %20, i64 %3, 2
  %22 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, i64 %4, 3, 0
  %23 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, i64 %6, 4, 0
  %24 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, i64 %5, 3, 1
  %25 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, i64 %7, 4, 1
  %26 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %8, 0
  %27 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %26, ptr %9, 1
  %28 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %27, i64 %10, 2
  %29 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %28, i64 %11, 3, 0
  %30 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %29, i64 %12, 4, 0
  %31 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %13, 0
  %32 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %31, ptr %14, 1
  %33 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %32, i64 %15, 2
  %34 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, i64 %16, 3, 0
  %35 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %34, i64 %17, 4, 0
  %36 = sext i32 %0 to i64
  %37 = sub i64 %36, 1
  %38 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %30, 3
  %39 = alloca [1 x i64], i64 1, align 8
  store [1 x i64] %38, ptr %39, align 8
  %40 = getelementptr [1 x i64], ptr %39, i32 0, i32 0
  %41 = load i64, ptr %40, align 8
  %42 = getelementptr double, ptr null, i64 %41
  %43 = ptrtoint ptr %42 to i64
  %44 = add i64 %43, 64
  %45 = call ptr @malloc(i64 %44)
  %46 = ptrtoint ptr %45 to i64
  %47 = add i64 %46, 63
  %48 = urem i64 %47, 64
  %49 = sub i64 %47, %48
  %50 = inttoptr i64 %49 to ptr
  %51 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %45, 0
  %52 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %51, ptr %50, 1
  %53 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %52, i64 0, 2
  %54 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %53, i64 %41, 3, 0
  %55 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %54, i64 1, 4, 0
  %56 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %30, 3, 0
  %57 = mul i64 %56, 1
  %58 = mul i64 %57, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %59 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %30, 1
  %60 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %30, 2
  %61 = getelementptr double, ptr %59, i64 %60
  %62 = getelementptr double, ptr %50, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %62, ptr %61, i64 %58, i1 false)
  br label %63

63:                                               ; preds = %148, %18
  %64 = phi i64 [ %174, %148 ], [ 0, %18 ]
  %65 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ %65, %148 ], [ %55, %18 ]
  %66 = icmp slt i64 %64, %36
  br i1 %66, label %67, label %175

67:                                               ; preds = %63
  %68 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %35, 1
  %69 = getelementptr double, ptr %68, i64 %64
  %70 = load double, ptr %69, align 8
  %71 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %65, 1
  %72 = getelementptr double, ptr %71, i64 %64
  store double %70, ptr %72, align 8
  %73 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, 0
  %74 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, 1
  %75 = insertvalue { ptr, ptr, i64 } undef, ptr %73, 0
  %76 = insertvalue { ptr, ptr, i64 } %75, ptr %74, 1
  %77 = insertvalue { ptr, ptr, i64 } %76, i64 0, 2
  %78 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, 2
  %79 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, 3, 0
  %80 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, 3, 1
  %81 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, 4, 0
  %82 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, 4, 1
  %83 = mul i64 %64, %81
  %84 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %73, 0
  %85 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %84, ptr %74, 1
  %86 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %85, i64 %83, 2
  %87 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %86, i64 %37, 3, 0
  %88 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %87, i64 1, 4, 0
  %89 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %65, 0
  %90 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %65, 1
  %91 = insertvalue { ptr, ptr, i64 } undef, ptr %89, 0
  %92 = insertvalue { ptr, ptr, i64 } %91, ptr %90, 1
  %93 = insertvalue { ptr, ptr, i64 } %92, i64 0, 2
  %94 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %65, 2
  %95 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %65, 3, 0
  %96 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %65, 4, 0
  %97 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %89, 0
  %98 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %97, ptr %90, 1
  %99 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %98, i64 0, 2
  %100 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %99, i64 %37, 3, 0
  %101 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %100, i64 1, 4, 0
  %102 = getelementptr double, ptr null, i64 %37
  %103 = ptrtoint ptr %102 to i64
  %104 = add i64 %103, 64
  %105 = call ptr @malloc(i64 %104)
  %106 = ptrtoint ptr %105 to i64
  %107 = add i64 %106, 63
  %108 = urem i64 %107, 64
  %109 = sub i64 %107, %108
  %110 = inttoptr i64 %109 to ptr
  %111 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %105, 0
  %112 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %111, ptr %110, 1
  %113 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %112, i64 0, 2
  %114 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %113, i64 %37, 3, 0
  %115 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %114, i64 1, 4, 0
  %116 = mul i64 %37, 1
  %117 = mul i64 %116, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %118 = getelementptr double, ptr %90, i64 0
  %119 = getelementptr double, ptr %110, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %119, ptr %118, i64 %117, i1 false)
  %120 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %65, 0
  %121 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %65, 1
  %122 = insertvalue { ptr, ptr, i64 } undef, ptr %120, 0
  %123 = insertvalue { ptr, ptr, i64 } %122, ptr %121, 1
  %124 = insertvalue { ptr, ptr, i64 } %123, i64 0, 2
  %125 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %65, 2
  %126 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %65, 3, 0
  %127 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %65, 4, 0
  %128 = insertvalue { ptr, ptr, i64 } undef, ptr %120, 0
  %129 = insertvalue { ptr, ptr, i64 } %128, ptr %121, 1
  %130 = insertvalue { ptr, ptr, i64 } %129, i64 %64, 2
  br label %131

131:                                              ; preds = %134, %67
  %132 = phi i64 [ %147, %134 ], [ 0, %67 ]
  %133 = icmp slt i64 %132, %37
  br i1 %133, label %134, label %148

134:                                              ; preds = %131
  %135 = getelementptr double, ptr %74, i64 %83
  %136 = getelementptr double, ptr %135, i64 %132
  %137 = load double, ptr %136, align 8
  %138 = getelementptr double, ptr %110, i64 %132
  %139 = load double, ptr %138, align 8
  %140 = getelementptr double, ptr %121, i64 %64
  %141 = load double, ptr %140, align 8
  %142 = fmul double %137, %139
  %143 = fsub double %141, %142
  %144 = icmp slt i64 %132, %64
  %145 = select i1 %144, double %143, double %141
  %146 = getelementptr double, ptr %121, i64 %64
  store double %145, ptr %146, align 8
  %147 = add i64 %132, 1
  br label %131

148:                                              ; preds = %131
  %149 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %65, 0
  %150 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %65, 1
  %151 = insertvalue { ptr, ptr, i64 } undef, ptr %149, 0
  %152 = insertvalue { ptr, ptr, i64 } %151, ptr %150, 1
  %153 = insertvalue { ptr, ptr, i64 } %152, i64 0, 2
  %154 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %65, 2
  %155 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %65, 3, 0
  %156 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %65, 4, 0
  %157 = insertvalue { ptr, ptr, i64 } undef, ptr %149, 0
  %158 = insertvalue { ptr, ptr, i64 } %157, ptr %150, 1
  %159 = insertvalue { ptr, ptr, i64 } %158, i64 %64, 2
  %160 = getelementptr double, ptr %121, i64 %64
  %161 = getelementptr double, ptr %150, i64 %64
  call void @llvm.memcpy.p0.p0.i64(ptr %161, ptr %160, i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), i1 false)
  %162 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %65, 1
  %163 = getelementptr double, ptr %162, i64 %64
  %164 = load double, ptr %163, align 8
  %165 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, 1
  %166 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, 4, 0
  %167 = mul i64 %64, %166
  %168 = add i64 %167, %64
  %169 = getelementptr double, ptr %165, i64 %168
  %170 = load double, ptr %169, align 8
  %171 = fdiv double %164, %170
  %172 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %65, 1
  %173 = getelementptr double, ptr %172, i64 %64
  store double %171, ptr %173, align 8
  %174 = add i64 %64, 1
  br label %63

175:                                              ; preds = %63
  %176 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %65, 3, 0
  %177 = mul i64 %176, 1
  %178 = mul i64 %177, ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64)
  %179 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %65, 1
  %180 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %65, 2
  %181 = getelementptr double, ptr %179, i64 %180
  %182 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %30, 1
  %183 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %30, 2
  %184 = getelementptr double, ptr %182, i64 %183
  call void @llvm.memcpy.p0.p0.i64(ptr %184, ptr %181, i64 %178, i1 false)
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #0

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
