; ModuleID = '/mnt/ccnas2/bdp/rz3515/projects/polymer/example/polybench/seidel-2d-debug/seidel-2d/seidel-2d.pluto.c'
source_filename = "/mnt/ccnas2/bdp/rz3515/projects/polymer/example/polybench/seidel-2d-debug/seidel-2d/seidel-2d.pluto.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }

@stderr = external dso_local local_unnamed_addr global %struct._IO_FILE*, align 8
@.str.1 = private unnamed_addr constant [23 x i8] c"==BEGIN DUMP_ARRAYS==\0A\00", align 1
@.str.2 = private unnamed_addr constant [15 x i8] c"begin dump: %s\00", align 1
@.str.3 = private unnamed_addr constant [2 x i8] c"A\00", align 1
@.str.5 = private unnamed_addr constant [8 x i8] c"%0.2lf \00", align 1
@.str.6 = private unnamed_addr constant [17 x i8] c"\0Aend   dump: %s\0A\00", align 1
@.str.7 = private unnamed_addr constant [23 x i8] c"==END   DUMP_ARRAYS==\0A\00", align 1

; Function Attrs: nounwind uwtable
define dso_local i32 @main(i32 %argc, i8** nocapture readonly %argv) local_unnamed_addr #0 {
entry:
  %call = tail call noalias dereferenceable_or_null(128000000) i8* @malloc(i64 128000000) #5
  %arraydecay = bitcast i8* %call to [4000 x double]*
  br label %for.cond1.preheader.i

for.cond1.preheader.i:                            ; preds = %for.inc9.i, %entry
  %indvars.iv4.i = phi i64 [ 0, %entry ], [ %indvars.iv.next5.i, %for.inc9.i ]
  %0 = trunc i64 %indvars.iv4.i to i32
  %conv.i = sitofp i32 %0 to double
  br label %for.body3.i

for.body3.i:                                      ; preds = %for.body3.i, %for.cond1.preheader.i
  %indvars.iv.i = phi i64 [ 0, %for.cond1.preheader.i ], [ %indvars.iv.next.i.1, %for.body3.i ]
  %1 = trunc i64 %indvars.iv.i to i32
  %2 = add nuw nsw i32 %1, 2
  %conv4.i = sitofp i32 %2 to double
  %mul.i = fmul double %conv.i, %conv4.i
  %add5.i = fadd double %mul.i, 2.000000e+00
  %div.i = fdiv double %add5.i, 4.000000e+03
  %arrayidx8.i = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %indvars.iv4.i, i64 %indvars.iv.i
  store double %div.i, double* %arrayidx8.i, align 8, !tbaa !2
  %indvars.iv.next.i = or i64 %indvars.iv.i, 1
  %3 = trunc i64 %indvars.iv.next.i to i32
  %4 = add nuw nsw i32 %3, 2
  %conv4.i.1 = sitofp i32 %4 to double
  %mul.i.1 = fmul double %conv.i, %conv4.i.1
  %add5.i.1 = fadd double %mul.i.1, 2.000000e+00
  %div.i.1 = fdiv double %add5.i.1, 4.000000e+03
  %arrayidx8.i.1 = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %indvars.iv4.i, i64 %indvars.iv.next.i
  store double %div.i.1, double* %arrayidx8.i.1, align 8, !tbaa !2
  %indvars.iv.next.i.1 = add nuw nsw i64 %indvars.iv.i, 2
  %exitcond.not.i.1 = icmp eq i64 %indvars.iv.next.i.1, 4000
  br i1 %exitcond.not.i.1, label %for.inc9.i, label %for.body3.i, !llvm.loop !6

for.inc9.i:                                       ; preds = %for.body3.i
  %indvars.iv.next5.i = add nuw nsw i64 %indvars.iv4.i, 1
  %exitcond6.not.i = icmp eq i64 %indvars.iv.next5.i, 4000
  br i1 %exitcond6.not.i, label %init_array.exit, label %for.cond1.preheader.i, !llvm.loop !8

init_array.exit:                                  ; preds = %for.inc9.i
  tail call void (...) @polybench_timer_start() #5
  br label %for.body90.lr.ph.i

for.body90.lr.ph.i:                               ; preds = %for.inc1588.i, %init_array.exit
  %indvars.iv192 = phi i64 [ %indvars.iv.next193, %for.inc1588.i ], [ 30, %init_array.exit ]
  %indvars.iv190 = phi i64 [ %indvars.iv.next191, %for.inc1588.i ], [ 31, %init_array.exit ]
  %indvars.iv = phi i32 [ %indvars.iv.next, %for.inc1588.i ], [ 30, %init_array.exit ]
  %indvars.iv116.i = phi i64 [ %indvars.iv.next117.i, %for.inc1588.i ], [ 126, %init_array.exit ]
  %indvars.iv92.i = phi i64 [ %indvars.iv.next93.i, %for.inc1588.i ], [ 0, %init_array.exit ]
  %indvars.iv50.i = phi i32 [ %indvars.iv.next51.i, %for.inc1588.i ], [ -4029, %init_array.exit ]
  %indvars.iv46.i = phi i32 [ %indvars.iv.next47.i, %for.inc1588.i ], [ -3998, %init_array.exit ]
  %indvars.iv40.i = phi i32 [ %indvars.iv.next41.i, %for.inc1588.i ], [ 0, %init_array.exit ]
  %indvars.iv92.i.tr = trunc i64 %indvars.iv92.i to i32
  %indvars.iv92.i.tr287 = trunc i64 %indvars.iv92.i to i32
  %indvars.iv92.i.tr288 = trunc i64 %indvars.iv92.i to i32
  %5 = shl nsw i64 %indvars.iv92.i, 5
  %6 = add nuw nsw i64 %5, 4029
  %7 = lshr i64 %6, 4
  %8 = icmp ult i64 %5, 963
  %9 = trunc i64 %7 to i32
  %cond217.i = select i1 %8, i32 %9, i32 312
  %10 = add nuw nsw i64 %5, 4060
  %11 = or i64 %5, 31
  %12 = icmp ult i64 %11, 999
  %cond1218125.i = select i1 %12, i64 %11, i64 999
  %13 = zext i32 %cond217.i to i64
  %14 = trunc i64 %indvars.iv92.i to i32
  %15 = trunc i64 %10 to i32
  br label %for.body90.i

for.body90.i:                                     ; preds = %for.inc1585.i, %for.body90.lr.ph.i
  %indvar200 = phi i32 [ %indvar.next201, %for.inc1585.i ], [ 0, %for.body90.lr.ph.i ]
  %indvars.iv194 = phi i64 [ %indvars.iv.next195, %for.inc1585.i ], [ %indvars.iv192, %for.body90.lr.ph.i ]
  %indvars.iv184 = phi i32 [ %indvars.iv.next185, %for.inc1585.i ], [ %indvars.iv, %for.body90.lr.ph.i ]
  %indvars.iv94.i = phi i64 [ %indvars.iv.next95.i, %for.inc1585.i ], [ %indvars.iv92.i, %for.body90.lr.ph.i ]
  %indvars.iv52.i = phi i32 [ %indvars.iv.next53.i, %for.inc1585.i ], [ %indvars.iv50.i, %for.body90.lr.ph.i ]
  %indvars.iv48.i = phi i32 [ %indvars.iv.next49.i, %for.inc1585.i ], [ %indvars.iv46.i, %for.body90.lr.ph.i ]
  %indvars.iv42.i = phi i32 [ %indvars.iv.next43.i, %for.inc1585.i ], [ %indvars.iv40.i, %for.body90.lr.ph.i ]
  %t2.036.i = phi i32 [ %inc1586.i, %for.inc1585.i ], [ %14, %for.body90.lr.ph.i ]
  %16 = add i32 %indvar200, %indvars.iv92.i.tr
  %17 = shl i32 %16, 5
  %18 = add i32 %indvar200, %indvars.iv92.i.tr287
  %19 = shl i32 %18, 5
  %20 = add i32 %indvar200, %indvars.iv92.i.tr288
  %21 = shl i32 %20, 5
  %22 = icmp ult i64 %indvars.iv190, %indvars.iv194
  %umin = select i1 %22, i64 %indvars.iv190, i64 %indvars.iv194
  %23 = icmp ult i64 %umin, 999
  %umin196 = select i1 %23, i64 %umin, i64 999
  %24 = trunc i64 %umin196 to i32
  %25 = shl nsw i64 %indvars.iv94.i, 6
  %cmp94.i = icmp ult i64 %25, 4028
  br i1 %cmp94.i, label %cond.end109.i, label %cond.end109.thread.i

cond.end109.i:                                    ; preds = %for.body90.i
  %26 = trunc i64 %25 to i16
  %27 = sub nuw nsw i16 4028, %26
  %div100.neg128.i = sdiv i16 %27, -32
  %div100.neg.sext.i = sext i16 %div100.neg128.i to i32
  %28 = add nuw nsw i64 %indvars.iv94.i, %indvars.iv92.i
  %29 = sext i16 %div100.neg128.i to i64
  %cmp112.i = icmp slt i64 %28, %29
  br label %cond.end136.i

cond.end109.thread.i:                             ; preds = %for.body90.i
  %30 = trunc i64 %25 to i32
  %31 = add i32 %30, -3997
  %div108.i = sdiv i32 %31, 32
  %32 = add nuw nsw i64 %indvars.iv94.i, %indvars.iv92.i
  %33 = sext i32 %div108.i to i64
  %cmp11216.i = icmp slt i64 %32, %33
  br label %cond.end136.i

cond.end136.i:                                    ; preds = %cond.end109.thread.i, %cond.end109.i
  %.sink289 = phi i64 [ %32, %cond.end109.thread.i ], [ %28, %cond.end109.i ]
  %div108.i.sink = phi i32 [ %div108.i, %cond.end109.thread.i ], [ %div100.neg.sext.i, %cond.end109.i ]
  %cmp11216.i.sink = phi i1 [ %cmp11216.i, %cond.end109.thread.i ], [ %cmp112.i, %cond.end109.i ]
  %34 = trunc i64 %.sink289 to i32
  %spec.select26.i = select i1 %cmp11216.i.sink, i32 %div108.i.sink, i32 %34
  %35 = add nuw nsw i64 %25, 4059
  %36 = lshr i64 %35, 5
  %cmp238.i = icmp ugt i64 %36, %13
  %37 = trunc i64 %36 to i32
  %cond341.i = select i1 %cmp238.i, i32 %cond217.i, i32 %37
  %38 = shl nsw i64 %indvars.iv94.i, 5
  %mul343.i = shl nsw i32 %t2.036.i, 5
  %39 = add nuw nsw i64 %38, %10
  %add346.i = add nuw nsw i32 %mul343.i, %15
  %40 = lshr i64 %39, 5
  %div365.i = lshr i32 %add346.i, 5
  %41 = zext i32 %cond341.i to i64
  %cmp368.i = icmp ugt i64 %40, %41
  %42 = add nuw nsw i64 %38, 5028
  %43 = lshr i64 %42, 5
  %cmp62520.i = icmp ult i64 %40, %43
  %44 = trunc i64 %43 to i32
  %spec.select22.i = select i1 %cmp62520.i, i32 %div365.i, i32 %44
  %cmp625.i = icmp ule i64 %43, %41
  %cmp368.not.i = xor i1 %cmp368.i, true
  %brmerge.i = or i1 %cmp625.i, %cmp368.not.i
  %div622.mux.i = select i1 %cmp625.i, i32 %44, i32 %div365.i
  %spec.select27.i = select i1 %brmerge.i, i32 %div622.mux.i, i32 %cond341.i
  %45 = add nsw i64 %38, -3998
  %cmp1122.i = icmp sgt i64 %5, %45
  %cond1130.v.i = select i1 %cmp1122.i, i64 %5, i64 %45
  %cond1130.i = trunc i64 %cond1130.v.i to i32
  %46 = or i64 %38, 30
  %cmp1221.i = icmp ult i64 %cond1218125.i, %46
  %cond1238.v.i = select i1 %cmp1221.i, i64 %cond1218125.i, i64 %46
  %cond1238.i = trunc i64 %cond1238.v.i to i32
  %47 = or i64 %38, 31
  %48 = shl i32 %spec.select26.i, 5
  %49 = add i32 %48, -3998
  %50 = add i32 %48, %indvars.iv52.i
  %51 = shl i32 %spec.select26.i, 4
  %52 = add i32 %51, -3998
  %53 = zext i32 %spec.select26.i to i64
  %54 = icmp sgt i32 %indvars.iv40.i, %indvars.iv48.i
  %smax81.i = select i1 %54, i32 %indvars.iv40.i, i32 %indvars.iv48.i
  %cond1115.i = select i1 %cmp368.i, i32 %spec.select27.i, i32 %spec.select22.i
  %55 = sext i32 %cond1115.i to i64
  %56 = trunc i64 %indvars.iv94.i to i32
  %57 = mul i32 %56, -32
  %58 = icmp sgt i32 %smax81.i, %50
  %smax82.i175 = select i1 %58, i32 %smax81.i, i32 %50
  %59 = icmp sgt i32 %smax82.i175, %52
  %smax83.i176 = select i1 %59, i32 %smax82.i175, i32 %52
  %cmp1116.not.i177 = icmp sgt i64 %53, %55
  br i1 %cmp1116.not.i177, label %for.inc1585.i, label %for.body1117.i.preheader

for.body1117.i.preheader:                         ; preds = %cond.end136.i
  %60 = sub i32 %49, %smax83.i176
  %61 = add i32 %indvars.iv184, %48
  %62 = or i32 %51, 14
  %63 = icmp sgt i64 %55, %53
  %smax198 = select i1 %63, i64 %55, i64 %53
  br label %for.body1117.i

for.body1117.i:                                   ; preds = %for.body1117.i.preheader, %for.inc1582.i
  %indvar = phi i32 [ 0, %for.body1117.i.preheader ], [ %indvar.next, %for.inc1582.i ]
  %indvars.iv188 = phi i32 [ %62, %for.body1117.i.preheader ], [ %indvars.iv.next189, %for.inc1582.i ]
  %indvars.iv186 = phi i32 [ %61, %for.body1117.i.preheader ], [ %indvars.iv.next187, %for.inc1582.i ]
  %64 = phi i32 [ %60, %for.body1117.i.preheader ], [ %280, %for.inc1582.i ]
  %smax83.i183 = phi i32 [ %smax83.i176, %for.body1117.i.preheader ], [ %smax83.i, %for.inc1582.i ]
  %indvars.iv.i169182 = phi i32 [ %48, %for.body1117.i.preheader ], [ %indvars.iv.next.i170, %for.inc1582.i ]
  %indvars.iv44.i181 = phi i32 [ %49, %for.body1117.i.preheader ], [ %indvars.iv.next45.i, %for.inc1582.i ]
  %indvars.iv54.i180 = phi i32 [ %50, %for.body1117.i.preheader ], [ %indvars.iv.next55.i, %for.inc1582.i ]
  %indvars.iv57.i179 = phi i32 [ %52, %for.body1117.i.preheader ], [ %indvars.iv.next58.i, %for.inc1582.i ]
  %indvars.iv90.i178 = phi i64 [ %53, %for.body1117.i.preheader ], [ %indvars.iv.next91.i, %for.inc1582.i ]
  %65 = add i32 %spec.select26.i, %indvar
  %66 = shl i32 %65, 5
  %67 = add i32 %spec.select26.i, %indvar
  %68 = shl i32 %67, 5
  %69 = add i32 %spec.select26.i, %indvar
  %70 = shl i32 %69, 5
  %71 = icmp slt i32 %indvars.iv186, %indvars.iv188
  %smin = select i1 %71, i32 %indvars.iv186, i32 %indvars.iv188
  %72 = icmp slt i32 %smin, %24
  %smin197 = select i1 %72, i32 %smin, i32 %24
  %73 = sext i32 %smin197 to i64
  %74 = trunc i64 %indvars.iv90.i178 to i32
  %mul1131.i = shl nsw i32 %74, 4
  %add1133.i = add nsw i32 %mul1131.i, -3998
  %cmp1134.i = icmp slt i32 %add1133.i, %cond1130.i
  %cond1154.i = select i1 %cmp1134.i, i32 %cond1130.i, i32 %add1133.i
  %mul1156.i = shl nsw i32 %74, 5
  %add1157.i = add nsw i32 %mul1156.i, %57
  %sub1159.i = add nsw i32 %add1157.i, -4029
  %cmp1160.i = icmp sgt i32 %cond1154.i, %sub1159.i
  %cond1206.i = select i1 %cmp1160.i, i32 %cond1154.i, i32 %sub1159.i
  %add1240.i = or i32 %mul1131.i, 14
  %cmp1241.i = icmp sgt i32 %add1240.i, %cond1238.i
  %cond1278.i = select i1 %cmp1241.i, i32 %cond1238.i, i32 %add1240.i
  %add1282.i = or i32 %add1157.i, 30
  %cmp1283.i = icmp slt i32 %cond1278.i, %add1282.i
  %cond1362.i = select i1 %cmp1283.i, i32 %cond1278.i, i32 %add1282.i
  %cmp1363.not33.i = icmp sgt i32 %cond1206.i, %cond1362.i
  br i1 %cmp1363.not33.i, label %for.inc1582.i, label %for.body1364.lr.ph.i

for.body1364.lr.ph.i:                             ; preds = %for.body1117.i
  %75 = zext i32 %smax83.i183 to i64
  %add1451.i = or i32 %mul1156.i, 31
  %76 = sext i32 %mul1156.i to i64
  %77 = sext i32 %add1451.i to i64
  %78 = icmp sgt i64 %73, %75
  %smax = select i1 %78, i64 %73, i64 %75
  %79 = add nuw i32 %smax83.i183, 1
  %80 = add nuw i32 %smax83.i183, 1
  %81 = xor i32 %smax83.i183, -1
  %82 = sub i32 1, %smax83.i183
  %83 = add nuw nsw i64 %75, 1
  %84 = sub nsw i32 0, %smax83.i183
  %85 = xor i64 %75, -1
  %86 = zext i32 %84 to i64
  %87 = add nuw nsw i64 %86, 1
  %88 = sub nsw i64 1, %75
  br label %for.body1364.i

for.cond1207.loopexit.i:                          ; preds = %for.inc1576.i, %for.body1364.i
  %indvars.iv.next61.i = add i32 %indvars.iv60.i, -1
  %exitcond.not = icmp eq i64 %indvars.iv84.i, %smax
  %indvar.next203 = add i32 %indvar202, 1
  %indvar.next221 = add i64 %indvar220, 1
  br i1 %exitcond.not, label %for.inc1582.i, label %for.body1364.i, !llvm.loop !9

for.body1364.i:                                   ; preds = %for.cond1207.loopexit.i, %for.body1364.lr.ph.i
  %indvar220 = phi i64 [ %indvar.next221, %for.cond1207.loopexit.i ], [ 0, %for.body1364.lr.ph.i ]
  %indvar202 = phi i32 [ %indvar.next203, %for.cond1207.loopexit.i ], [ 0, %for.body1364.lr.ph.i ]
  %indvars.iv84.i = phi i64 [ %indvars.iv.next85.i, %for.cond1207.loopexit.i ], [ %75, %for.body1364.lr.ph.i ]
  %indvars.iv63.in.i = phi i32 [ %indvars.iv63.i, %for.cond1207.loopexit.i ], [ %smax83.i183, %for.body1364.lr.ph.i ]
  %indvars.iv60.i = phi i32 [ %indvars.iv.next61.i, %for.cond1207.loopexit.i ], [ %64, %for.body1364.lr.ph.i ]
  %89 = add i64 %indvar220, %75
  %90 = trunc i64 %indvar220 to i32
  %91 = sub i32 %64, %90
  %92 = icmp sgt i32 %17, %91
  %smax222 = select i1 %92, i32 %17, i32 %91
  %93 = add i64 %83, %indvar220
  %94 = trunc i64 %93 to i32
  %95 = icmp sgt i32 %smax222, %94
  %smax223 = select i1 %95, i32 %smax222, i32 %94
  %96 = zext i32 %smax223 to i64
  %97 = sub i64 %96, %89
  %98 = mul i64 %97, 32000
  %99 = add i32 %smax223, %94
  %100 = trunc i64 %indvar220 to i32
  %101 = sub i32 %84, %100
  %102 = sub i32 %101, %smax223
  %103 = or i64 %98, 8
  %104 = sub i64 %85, %indvar220
  %105 = add i64 %104, %96
  %106 = mul i64 %105, 32000
  %107 = trunc i64 %indvar220 to i32
  %108 = xor i32 %107, -1
  %109 = sub i32 %108, %smax83.i183
  %110 = sub i32 %109, %smax223
  %111 = or i64 %106, 8
  %112 = sub i64 %87, %indvar220
  %113 = trunc i64 %112 to i32
  %114 = sub i32 %113, %smax223
  %115 = sub i64 %88, %indvar220
  %116 = add i64 %115, %96
  %117 = mul i64 %116, 32000
  %118 = or i64 %117, 8
  %119 = sub i32 %64, %indvar202
  %120 = icmp sgt i32 %19, %119
  %smax210 = select i1 %120, i32 %19, i32 %119
  %121 = add i32 %80, %indvar202
  %122 = icmp sgt i32 %smax210, %121
  %smax211 = select i1 %122, i32 %smax210, i32 %121
  %123 = add i32 %smax211, %121
  %124 = add i32 %smax83.i183, %indvar202
  %125 = sub i32 0, %124
  %126 = sub i32 %125, %smax211
  %127 = sub i32 %81, %indvar202
  %128 = sub i32 %127, %smax211
  %129 = sub i32 %82, %indvar202
  %130 = sub i32 %129, %smax211
  %131 = sub i32 %64, %indvar202
  %132 = icmp sgt i32 %21, %131
  %smax204 = select i1 %132, i32 %21, i32 %131
  %133 = add i32 %79, %indvar202
  %134 = icmp sgt i32 %smax204, %133
  %smax205 = select i1 %134, i32 %smax204, i32 %133
  %135 = add i32 %smax205, %133
  %indvars.iv63.i = add nuw i32 %indvars.iv63.in.i, 1
  %136 = icmp sgt i32 %indvars.iv42.i, %indvars.iv60.i
  %smax71.i = select i1 %136, i32 %indvars.iv42.i, i32 %indvars.iv60.i
  %137 = icmp sgt i32 %smax71.i, %indvars.iv63.i
  %smax72.i = select i1 %137, i32 %smax71.i, i32 %indvars.iv63.i
  %indvars.iv.next85.i = add nuw nsw i64 %indvars.iv84.i, 1
  %cmp1367.i = icmp ugt i64 %38, %indvars.iv.next85.i
  %cond1373.v.i = select i1 %cmp1367.i, i64 %38, i64 %indvars.iv.next85.i
  %138 = sub nsw i64 %76, %indvars.iv84.i
  %139 = add nsw i64 %138, -3998
  %sext.i = shl i64 %cond1373.v.i, 32
  %140 = ashr exact i64 %sext.i, 32
  %cmp1378.i = icmp sgt i64 %140, %139
  %cond1395.v.i = select i1 %cmp1378.i, i64 %cond1373.v.i, i64 %139
  %cond1395.i = trunc i64 %cond1395.v.i to i32
  %141 = add nsw i64 %138, 30
  %cmp1402.i = icmp slt i64 %47, %141
  %cond1411.v.i = select i1 %cmp1402.i, i64 %47, i64 %141
  %142 = add nuw nsw i64 %indvars.iv84.i, 3998
  %sext126.i = shl i64 %cond1411.v.i, 32
  %143 = ashr exact i64 %sext126.i, 32
  %cmp1414.i = icmp slt i64 %143, %142
  %cond1435.v.i = select i1 %cmp1414.i, i64 %cond1411.v.i, i64 %142
  %cond1435.i = trunc i64 %cond1435.v.i to i32
  %cmp1436.not30.i = icmp sgt i32 %cond1395.i, %cond1435.i
  br i1 %cmp1436.not30.i, label %for.cond1207.loopexit.i, label %for.body1437.lr.ph.i

for.body1437.lr.ph.i:                             ; preds = %for.body1364.i
  %144 = add i32 %smax72.i, %indvars.iv63.i
  %145 = zext i32 %smax72.i to i64
  %sext127.i = shl i64 %cond1435.v.i, 32
  %146 = ashr exact i64 %sext127.i, 32
  %147 = trunc i64 %indvars.iv84.i to i32
  br label %for.body1437.i

for.body1437.i:                                   ; preds = %for.inc1576.i, %for.body1437.lr.ph.i
  %indvar224 = phi i64 [ %indvar.next225, %for.inc1576.i ], [ 0, %for.body1437.lr.ph.i ]
  %indvar206 = phi i32 [ %indvar.next207, %for.inc1576.i ], [ 0, %for.body1437.lr.ph.i ]
  %indvars.iv73.i = phi i64 [ %indvars.iv.next74.i, %for.inc1576.i ], [ %145, %for.body1437.lr.ph.i ]
  %indvars.iv66.i = phi i32 [ %indvars.iv.next67.i, %for.inc1576.i ], [ %144, %for.body1437.lr.ph.i ]
  %148 = mul nuw nsw i64 %indvar224, 32000
  %149 = add i64 %98, %148
  %scevgep = getelementptr i8, i8* %call, i64 %149
  %150 = trunc i64 %indvar224 to i32
  %151 = add i32 %99, %150
  %152 = icmp sgt i32 %66, %151
  %smax226 = select i1 %152, i32 %66, i32 %151
  %153 = trunc i64 %indvar224 to i32
  %154 = sub i32 %102, %153
  %155 = add i32 %smax226, %154
  %156 = sext i32 %155 to i64
  %157 = shl nsw i64 %156, 3
  %scevgep227 = getelementptr i8, i8* %scevgep, i64 %157
  %158 = add i64 %103, %148
  %scevgep228 = getelementptr i8, i8* %call, i64 %158
  %159 = sext i32 %smax226 to i64
  %160 = shl nsw i64 %159, 3
  %161 = add i64 %106, %148
  %scevgep231 = getelementptr i8, i8* %call, i64 %161
  %162 = trunc i64 %indvar224 to i32
  %163 = sub i32 %110, %162
  %164 = add i32 %smax226, %163
  %165 = sext i32 %164 to i64
  %166 = shl nsw i64 %165, 3
  %scevgep232 = getelementptr i8, i8* %scevgep231, i64 %166
  %167 = add i64 %111, %148
  %scevgep233 = getelementptr i8, i8* %call, i64 %167
  %scevgep235 = getelementptr i8, i8* %scevgep231, i64 %157
  %168 = trunc i64 %indvar224 to i32
  %169 = sub i32 %114, %168
  %170 = add i32 %smax226, %169
  %171 = sext i32 %170 to i64
  %172 = shl nsw i64 %171, 3
  %scevgep237 = getelementptr i8, i8* %scevgep231, i64 %172
  %scevgep239 = getelementptr i8, i8* %scevgep, i64 %166
  %scevgep241 = getelementptr i8, i8* %scevgep, i64 %172
  %173 = add i64 %117, %148
  %scevgep243 = getelementptr i8, i8* %call, i64 %173
  %scevgep244 = getelementptr i8, i8* %scevgep243, i64 %166
  %174 = add i64 %118, %148
  %scevgep245 = getelementptr i8, i8* %call, i64 %174
  %scevgep247 = getelementptr i8, i8* %scevgep243, i64 %157
  %scevgep249 = getelementptr i8, i8* %scevgep243, i64 %172
  %175 = add i32 %123, %indvar206
  %176 = icmp sgt i32 %68, %175
  %smax212 = select i1 %176, i32 %68, i32 %175
  %177 = sext i32 %smax212 to i64
  %178 = sub i32 %126, %indvar206
  %179 = add i32 %smax212, %178
  %180 = sub i32 %128, %indvar206
  %181 = add i32 %smax212, %180
  %182 = sub i32 %130, %indvar206
  %183 = add i32 %smax212, %182
  %184 = add i32 %135, %indvar206
  %185 = icmp sgt i32 %70, %184
  %smax208 = select i1 %185, i32 %70, i32 %184
  %186 = sext i32 %smax208 to i64
  %187 = add nuw nsw i64 %indvars.iv73.i, %indvars.iv84.i
  %188 = add nuw nsw i64 %187, 1
  %cmp1441.i = icmp slt i64 %188, %76
  %189 = trunc i64 %188 to i32
  %cond1448.i = select i1 %cmp1441.i, i32 %mul1156.i, i32 %189
  %190 = add nuw nsw i64 %187, 3998
  %cmp1455.i = icmp sgt i64 %190, %77
  %191 = trunc i64 %190 to i32
  %cond1464.i = select i1 %cmp1455.i, i32 %add1451.i, i32 %191
  %cmp1465.not28.i = icmp sgt i32 %cond1448.i, %cond1464.i
  br i1 %cmp1465.not28.i, label %for.inc1576.i, label %for.body1466.lr.ph.i

for.body1466.lr.ph.i:                             ; preds = %for.body1437.i
  %192 = icmp sgt i32 %indvars.iv.i169182, %indvars.iv66.i
  %smax68.i = select i1 %192, i32 %indvars.iv.i169182, i32 %indvars.iv66.i
  %193 = sext i32 %smax68.i to i64
  %194 = sub nsw i64 %indvars.iv73.i, %indvars.iv84.i
  %195 = add nsw i64 %194, -1
  %196 = trunc i64 %indvars.iv73.i to i32
  %197 = add nsw i64 %194, 1
  %198 = sext i32 %cond1464.i to i64
  %199 = add i32 %196, %147
  %200 = icmp sgt i64 %198, %186
  %smax209 = select i1 %200, i64 %198, i64 %186
  %201 = add nsw i64 %smax209, 1
  %202 = sub nsw i64 %201, %186
  %min.iters.check = icmp ult i64 %202, 2
  br i1 %min.iters.check, label %for.body1466.i.preheader, label %vector.scevcheck

vector.scevcheck:                                 ; preds = %for.body1466.lr.ph.i
  %203 = icmp sgt i64 %198, %177
  %204 = sub nsw i64 %198, %177
  %205 = select i1 %203, i64 %204, i64 0
  %206 = trunc i64 %205 to i32
  %207 = add i32 %179, %206
  %208 = icmp slt i32 %207, %179
  %209 = icmp ugt i64 %205, 4294967295
  %210 = or i1 %208, %209
  %211 = trunc i64 %205 to i32
  %212 = add i32 %181, %211
  %213 = icmp slt i32 %212, %181
  %214 = icmp ugt i64 %205, 4294967295
  %215 = or i1 %213, %214
  %216 = or i1 %210, %215
  %217 = trunc i64 %205 to i32
  %218 = add i32 %183, %217
  %219 = icmp slt i32 %218, %183
  %220 = icmp ugt i64 %205, 4294967295
  %221 = or i1 %219, %220
  %222 = or i1 %216, %221
  br i1 %222, label %for.body1466.i.preheader, label %vector.memcheck

vector.memcheck:                                  ; preds = %vector.scevcheck
  %223 = icmp sgt i64 %198, %159
  %smax229 = select i1 %223, i64 %198, i64 %159
  %224 = shl nsw i64 %smax229, 3
  %225 = add nsw i64 %224, %157
  %226 = sub nsw i64 %225, %160
  %scevgep230 = getelementptr i8, i8* %scevgep228, i64 %226
  %227 = add nsw i64 %224, %166
  %228 = sub nsw i64 %227, %160
  %scevgep234 = getelementptr i8, i8* %scevgep233, i64 %228
  %scevgep236 = getelementptr i8, i8* %scevgep233, i64 %226
  %229 = add nsw i64 %224, %172
  %230 = sub nsw i64 %229, %160
  %scevgep238 = getelementptr i8, i8* %scevgep233, i64 %230
  %scevgep240 = getelementptr i8, i8* %scevgep228, i64 %228
  %scevgep242 = getelementptr i8, i8* %scevgep228, i64 %230
  %scevgep246 = getelementptr i8, i8* %scevgep245, i64 %228
  %scevgep248 = getelementptr i8, i8* %scevgep245, i64 %226
  %scevgep250 = getelementptr i8, i8* %scevgep245, i64 %230
  %bound0 = icmp ult i8* %scevgep227, %scevgep234
  %bound1 = icmp ult i8* %scevgep232, %scevgep230
  %found.conflict = and i1 %bound0, %bound1
  %bound0251 = icmp ult i8* %scevgep227, %scevgep236
  %bound1252 = icmp ult i8* %scevgep235, %scevgep230
  %found.conflict253 = and i1 %bound0251, %bound1252
  %conflict.rdx = or i1 %found.conflict, %found.conflict253
  %bound0254 = icmp ult i8* %scevgep227, %scevgep238
  %bound1255 = icmp ult i8* %scevgep237, %scevgep230
  %found.conflict256 = and i1 %bound0254, %bound1255
  %conflict.rdx257 = or i1 %conflict.rdx, %found.conflict256
  %bound0258 = icmp ult i8* %scevgep227, %scevgep240
  %bound1259 = icmp ult i8* %scevgep239, %scevgep230
  %found.conflict260 = and i1 %bound0258, %bound1259
  %conflict.rdx261 = or i1 %conflict.rdx257, %found.conflict260
  %bound0262 = icmp ult i8* %scevgep227, %scevgep242
  %bound1263 = icmp ult i8* %scevgep241, %scevgep230
  %found.conflict264 = and i1 %bound0262, %bound1263
  %conflict.rdx265 = or i1 %conflict.rdx261, %found.conflict264
  %bound0266 = icmp ult i8* %scevgep227, %scevgep246
  %bound1267 = icmp ult i8* %scevgep244, %scevgep230
  %found.conflict268 = and i1 %bound0266, %bound1267
  %conflict.rdx269 = or i1 %conflict.rdx265, %found.conflict268
  %bound0270 = icmp ult i8* %scevgep227, %scevgep248
  %bound1271 = icmp ult i8* %scevgep247, %scevgep230
  %found.conflict272 = and i1 %bound0270, %bound1271
  %conflict.rdx273 = or i1 %conflict.rdx269, %found.conflict272
  %bound0274 = icmp ult i8* %scevgep227, %scevgep250
  %bound1275 = icmp ult i8* %scevgep249, %scevgep230
  %found.conflict276 = and i1 %bound0274, %bound1275
  %conflict.rdx277 = or i1 %conflict.rdx273, %found.conflict276
  br i1 %conflict.rdx277, label %for.body1466.i.preheader, label %vector.ph

vector.ph:                                        ; preds = %vector.memcheck
  %n.vec = and i64 %202, -2
  %ind.end = add nsw i64 %n.vec, %193
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %231 = trunc i64 %index to i32
  %232 = add i32 %smax68.i, %231
  %233 = sub i32 %232, %199
  %234 = add nsw i32 %233, -1
  %235 = sext i32 %234 to i64
  %236 = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %195, i64 %235
  %237 = bitcast double* %236 to <2 x double>*
  %wide.load = load <2 x double>, <2 x double>* %237, align 8, !tbaa !2, !alias.scope !10
  %238 = sext i32 %233 to i64
  %239 = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %195, i64 %238
  %240 = bitcast double* %239 to <2 x double>*
  %wide.load279 = load <2 x double>, <2 x double>* %240, align 8, !tbaa !2, !alias.scope !13
  %241 = fadd <2 x double> %wide.load, %wide.load279
  %242 = add nsw i32 %233, 1
  %243 = sext i32 %242 to i64
  %244 = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %195, i64 %243
  %245 = bitcast double* %244 to <2 x double>*
  %wide.load280 = load <2 x double>, <2 x double>* %245, align 8, !tbaa !2, !alias.scope !15
  %246 = fadd <2 x double> %241, %wide.load280
  %247 = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %194, i64 %235
  %248 = bitcast double* %247 to <2 x double>*
  %wide.load281 = load <2 x double>, <2 x double>* %248, align 8, !tbaa !2, !alias.scope !17
  %249 = fadd <2 x double> %246, %wide.load281
  %250 = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %194, i64 %238
  %251 = bitcast double* %250 to <2 x double>*
  %wide.load282 = load <2 x double>, <2 x double>* %251, align 8, !tbaa !2, !alias.scope !19, !noalias !21
  %252 = fadd <2 x double> %249, %wide.load282
  %253 = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %194, i64 %243
  %254 = bitcast double* %253 to <2 x double>*
  %wide.load283 = load <2 x double>, <2 x double>* %254, align 8, !tbaa !2, !alias.scope !26
  %255 = fadd <2 x double> %252, %wide.load283
  %256 = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %197, i64 %235
  %257 = bitcast double* %256 to <2 x double>*
  %wide.load284 = load <2 x double>, <2 x double>* %257, align 8, !tbaa !2, !alias.scope !27
  %258 = fadd <2 x double> %255, %wide.load284
  %259 = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %197, i64 %238
  %260 = bitcast double* %259 to <2 x double>*
  %wide.load285 = load <2 x double>, <2 x double>* %260, align 8, !tbaa !2, !alias.scope !28
  %261 = fadd <2 x double> %258, %wide.load285
  %262 = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %197, i64 %243
  %263 = bitcast double* %262 to <2 x double>*
  %wide.load286 = load <2 x double>, <2 x double>* %263, align 8, !tbaa !2, !alias.scope !29
  %264 = fadd <2 x double> %261, %wide.load286
  %265 = fdiv <2 x double> %264, <double 9.000000e+00, double 9.000000e+00>
  %266 = bitcast double* %250 to <2 x double>*
  store <2 x double> %265, <2 x double>* %266, align 8, !tbaa !2, !alias.scope !19, !noalias !21
  %index.next = add i64 %index, 2
  %267 = icmp eq i64 %index.next, %n.vec
  br i1 %267, label %middle.block, label %vector.body, !llvm.loop !30

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i64 %202, %n.vec
  br i1 %cmp.n, label %for.inc1576.i, label %for.body1466.i.preheader

for.body1466.i.preheader:                         ; preds = %middle.block, %vector.memcheck, %vector.scevcheck, %for.body1466.lr.ph.i
  %indvars.iv69.i.ph = phi i64 [ %193, %vector.memcheck ], [ %193, %vector.scevcheck ], [ %193, %for.body1466.lr.ph.i ], [ %ind.end, %middle.block ]
  br label %for.body1466.i

for.body1466.i:                                   ; preds = %for.body1466.i.preheader, %for.body1466.i
  %indvars.iv69.i = phi i64 [ %indvars.iv.next70.i, %for.body1466.i ], [ %indvars.iv69.i.ph, %for.body1466.i.preheader ]
  %268 = trunc i64 %indvars.iv69.i to i32
  %add1472.i = sub i32 %268, %199
  %sub1473.i = add nsw i32 %add1472.i, -1
  %idxprom1474.i = sext i32 %sub1473.i to i64
  %arrayidx1475.i = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %195, i64 %idxprom1474.i
  %269 = load double, double* %arrayidx1475.i, align 8, !tbaa !2
  %idxprom1484.i = sext i32 %add1472.i to i64
  %arrayidx1485.i = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %195, i64 %idxprom1484.i
  %270 = load double, double* %arrayidx1485.i, align 8, !tbaa !2
  %add1486.i = fadd double %269, %270
  %add1495.i = add nsw i32 %add1472.i, 1
  %idxprom1496.i = sext i32 %add1495.i to i64
  %arrayidx1497.i = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %195, i64 %idxprom1496.i
  %271 = load double, double* %arrayidx1497.i, align 8, !tbaa !2
  %add1498.i = fadd double %add1486.i, %271
  %arrayidx1508.i = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %194, i64 %idxprom1474.i
  %272 = load double, double* %arrayidx1508.i, align 8, !tbaa !2
  %add1509.i = fadd double %add1498.i, %272
  %arrayidx1518.i = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %194, i64 %idxprom1484.i
  %273 = load double, double* %arrayidx1518.i, align 8, !tbaa !2
  %add1519.i = fadd double %add1509.i, %273
  %arrayidx1529.i = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %194, i64 %idxprom1496.i
  %274 = load double, double* %arrayidx1529.i, align 8, !tbaa !2
  %add1530.i = fadd double %add1519.i, %274
  %arrayidx1541.i = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %197, i64 %idxprom1474.i
  %275 = load double, double* %arrayidx1541.i, align 8, !tbaa !2
  %add1542.i = fadd double %add1530.i, %275
  %arrayidx1552.i = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %197, i64 %idxprom1484.i
  %276 = load double, double* %arrayidx1552.i, align 8, !tbaa !2
  %add1553.i = fadd double %add1542.i, %276
  %arrayidx1564.i = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %197, i64 %idxprom1496.i
  %277 = load double, double* %arrayidx1564.i, align 8, !tbaa !2
  %add1565.i = fadd double %add1553.i, %277
  %div1566.i = fdiv double %add1565.i, 9.000000e+00
  store double %div1566.i, double* %arrayidx1518.i, align 8, !tbaa !2
  %indvars.iv.next70.i = add nsw i64 %indvars.iv69.i, 1
  %cmp1465.not.not.i = icmp slt i64 %indvars.iv69.i, %198
  br i1 %cmp1465.not.not.i, label %for.body1466.i, label %for.inc1576.i, !llvm.loop !32

for.inc1576.i:                                    ; preds = %for.body1466.i, %middle.block, %for.body1437.i
  %indvars.iv.next74.i = add nuw nsw i64 %indvars.iv73.i, 1
  %cmp1436.not.not.i = icmp slt i64 %indvars.iv73.i, %146
  %indvars.iv.next67.i = add i32 %indvars.iv66.i, 1
  %indvar.next207 = add i32 %indvar206, 1
  %indvar.next225 = add i64 %indvar224, 1
  br i1 %cmp1436.not.not.i, label %for.body1437.i, label %for.cond1207.loopexit.i, !llvm.loop !33

for.inc1582.i:                                    ; preds = %for.cond1207.loopexit.i, %for.body1117.i
  %indvars.iv.next91.i = add nuw nsw i64 %indvars.iv90.i178, 1
  %indvars.iv.next.i170 = add i32 %indvars.iv.i169182, 32
  %indvars.iv.next45.i = add i32 %indvars.iv44.i181, 32
  %indvars.iv.next55.i = add i32 %indvars.iv54.i180, 32
  %indvars.iv.next58.i = add i32 %indvars.iv57.i179, 16
  %278 = icmp sgt i32 %smax81.i, %indvars.iv.next55.i
  %smax82.i = select i1 %278, i32 %smax81.i, i32 %indvars.iv.next55.i
  %279 = icmp sgt i32 %smax82.i, %indvars.iv.next58.i
  %smax83.i = select i1 %279, i32 %smax82.i, i32 %indvars.iv.next58.i
  %280 = sub i32 %indvars.iv.next45.i, %smax83.i
  %indvars.iv.next187 = add i32 %indvars.iv186, 32
  %indvars.iv.next189 = add i32 %indvars.iv188, 16
  %exitcond199.not = icmp eq i64 %indvars.iv90.i178, %smax198
  %indvar.next = add i32 %indvar, 1
  br i1 %exitcond199.not, label %for.inc1585.i, label %for.body1117.i, !llvm.loop !34

for.inc1585.i:                                    ; preds = %for.inc1582.i, %cond.end136.i
  %indvars.iv.next95.i = add nuw nsw i64 %indvars.iv94.i, 1
  %inc1586.i = add nuw nsw i32 %t2.036.i, 1
  %indvars.iv.next43.i = add nuw nsw i32 %indvars.iv42.i, 32
  %indvars.iv.next49.i = add nsw i32 %indvars.iv48.i, 32
  %indvars.iv.next53.i = add nsw i32 %indvars.iv52.i, -32
  %exitcond.not.i171 = icmp eq i64 %indvars.iv.next95.i, %indvars.iv116.i
  %indvars.iv.next185 = add nsw i32 %indvars.iv184, -32
  %indvars.iv.next195 = add nuw nsw i64 %indvars.iv194, 32
  %indvar.next201 = add i32 %indvar200, 1
  br i1 %exitcond.not.i171, label %for.inc1588.i, label %for.body90.i, !llvm.loop !35

for.inc1588.i:                                    ; preds = %for.inc1585.i
  %indvars.iv.next93.i = add nuw nsw i64 %indvars.iv92.i, 1
  %indvars.iv.next41.i = add nuw nsw i32 %indvars.iv40.i, 32
  %indvars.iv.next47.i = add nuw nsw i32 %indvars.iv46.i, 32
  %indvars.iv.next51.i = add nsw i32 %indvars.iv50.i, -32
  %indvars.iv.next117.i = add nuw nsw i64 %indvars.iv116.i, 1
  %exitcond124.not.i = icmp eq i64 %indvars.iv.next93.i, 32
  %indvars.iv.next = add nsw i32 %indvars.iv, -32
  %indvars.iv.next191 = add nuw nsw i64 %indvars.iv190, 32
  %indvars.iv.next193 = add nuw nsw i64 %indvars.iv192, 32
  br i1 %exitcond124.not.i, label %kernel_seidel_2d.exit, label %for.body90.lr.ph.i, !llvm.loop !36

kernel_seidel_2d.exit:                            ; preds = %for.inc1588.i
  tail call void (...) @polybench_timer_stop() #5
  tail call void (...) @polybench_timer_print() #5
  %cmp = icmp sgt i32 %argc, 42
  br i1 %cmp, label %if.end123, label %if.end138

if.end123:                                        ; preds = %kernel_seidel_2d.exit
  %281 = load i8*, i8** %argv, align 8, !tbaa !37
  %282 = load i8, i8* %281, align 1, !tbaa !39
  %phi.cmp = icmp eq i8 %282, 0
  br i1 %phi.cmp, label %if.then136, label %if.end138

if.then136:                                       ; preds = %if.end123
  %283 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !37
  %284 = tail call i64 @fwrite(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.1, i64 0, i64 0), i64 22, i64 1, %struct._IO_FILE* %283) #6
  %285 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !37
  %call1.i = tail call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %285, i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str.2, i64 0, i64 0), i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.3, i64 0, i64 0)) #6
  br label %for.cond2.preheader.i

for.cond2.preheader.i:                            ; preds = %for.inc10.i, %if.then136
  %indvars.iv4.i163 = phi i64 [ 0, %if.then136 ], [ %indvars.iv.next5.i168, %for.inc10.i ]
  %286 = mul nuw nsw i64 %indvars.iv4.i163, 4000
  br label %for.body4.i

for.body4.i:                                      ; preds = %if.end.i, %for.cond2.preheader.i
  %indvars.iv.i164 = phi i64 [ 0, %for.cond2.preheader.i ], [ %indvars.iv.next.i166, %if.end.i ]
  %287 = add nuw nsw i64 %indvars.iv.i164, %286
  %288 = trunc i64 %287 to i32
  %rem.i = urem i32 %288, 20
  %cmp5.i = icmp eq i32 %rem.i, 0
  br i1 %cmp5.i, label %if.then.i, label %if.end.i

if.then.i:                                        ; preds = %for.body4.i
  %289 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !37
  %fputc.i = tail call i32 @fputc(i32 10, %struct._IO_FILE* %289) #5
  br label %if.end.i

if.end.i:                                         ; preds = %if.then.i, %for.body4.i
  %290 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !37
  %arrayidx8.i165 = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %indvars.iv4.i163, i64 %indvars.iv.i164
  %291 = load double, double* %arrayidx8.i165, align 8, !tbaa !2
  %call9.i = tail call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %290, i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.5, i64 0, i64 0), double %291) #6
  %indvars.iv.next.i166 = add nuw nsw i64 %indvars.iv.i164, 1
  %exitcond.not.i167 = icmp eq i64 %indvars.iv.next.i166, 4000
  br i1 %exitcond.not.i167, label %for.inc10.i, label %for.body4.i, !llvm.loop !40

for.inc10.i:                                      ; preds = %if.end.i
  %indvars.iv.next5.i168 = add nuw nsw i64 %indvars.iv4.i163, 1
  %exitcond7.not.i = icmp eq i64 %indvars.iv.next5.i168, 4000
  br i1 %exitcond7.not.i, label %print_array.exit, label %for.cond2.preheader.i, !llvm.loop !41

print_array.exit:                                 ; preds = %for.inc10.i
  %292 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !37
  %call13.i = tail call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %292, i8* getelementptr inbounds ([17 x i8], [17 x i8]* @.str.6, i64 0, i64 0), i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.3, i64 0, i64 0)) #6
  %293 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !37
  %294 = tail call i64 @fwrite(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.7, i64 0, i64 0), i64 22, i64 1, %struct._IO_FILE* %293) #6
  br label %if.end138

if.end138:                                        ; preds = %print_array.exit, %if.end123, %kernel_seidel_2d.exit
  tail call void @free(i8* %call) #5
  ret i32 0
}

; Function Attrs: nofree nounwind
declare dso_local noalias noundef i8* @malloc(i64) local_unnamed_addr #1

declare dso_local void @polybench_timer_start(...) local_unnamed_addr #2

declare dso_local void @polybench_timer_stop(...) local_unnamed_addr #2

declare dso_local void @polybench_timer_print(...) local_unnamed_addr #2

; Function Attrs: nounwind
declare dso_local void @free(i8* nocapture noundef) local_unnamed_addr #3

; Function Attrs: nofree nounwind
declare dso_local noundef i32 @fprintf(%struct._IO_FILE* nocapture noundef, i8* nocapture noundef readonly, ...) local_unnamed_addr #1

; Function Attrs: nofree nounwind
declare noundef i64 @fwrite(i8* nocapture noundef, i64 noundef, i64 noundef, %struct._IO_FILE* nocapture noundef) local_unnamed_addr #4

; Function Attrs: nofree nounwind
declare noundef i32 @fputc(i32 noundef, %struct._IO_FILE* nocapture noundef) local_unnamed_addr #4

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nofree nounwind "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nofree nounwind }
attributes #5 = { nounwind }
attributes #6 = { cold nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 12.0.0 (git@github.com:wsmoses/MLIR-GPU 1112d5451cea635029a160c950f14a85f31b2258)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
!8 = distinct !{!8, !7}
!9 = distinct !{!9, !7}
!10 = !{!11}
!11 = distinct !{!11, !12}
!12 = distinct !{!12, !"LVerDomain"}
!13 = !{!14}
!14 = distinct !{!14, !12}
!15 = !{!16}
!16 = distinct !{!16, !12}
!17 = !{!18}
!18 = distinct !{!18, !12}
!19 = !{!20}
!20 = distinct !{!20, !12}
!21 = !{!11, !14, !16, !18, !22, !23, !24, !25}
!22 = distinct !{!22, !12}
!23 = distinct !{!23, !12}
!24 = distinct !{!24, !12}
!25 = distinct !{!25, !12}
!26 = !{!22}
!27 = !{!23}
!28 = !{!24}
!29 = !{!25}
!30 = distinct !{!30, !7, !31}
!31 = !{!"llvm.loop.isvectorized", i32 1}
!32 = distinct !{!32, !7, !31}
!33 = distinct !{!33, !7}
!34 = distinct !{!34, !7}
!35 = distinct !{!35, !7}
!36 = distinct !{!36, !7}
!37 = !{!38, !38, i64 0}
!38 = !{!"any pointer", !4, i64 0}
!39 = !{!4, !4, i64 0}
!40 = distinct !{!40, !7}
!41 = distinct !{!41, !7}
