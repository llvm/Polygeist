; ModuleID = '/mnt/ccnas2/bdp/rz3515/projects/polymer/example/polybench/seidel-2d-debug/seidel-2d/seidel-2d.pluto.ll'
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
  %indvars.iv194 = phi i64 [ %indvars.iv.next195, %for.inc1585.i ], [ %indvars.iv192, %for.body90.lr.ph.i ]
  %indvars.iv184 = phi i32 [ %indvars.iv.next185, %for.inc1585.i ], [ %indvars.iv, %for.body90.lr.ph.i ]
  %indvars.iv94.i = phi i64 [ %indvars.iv.next95.i, %for.inc1585.i ], [ %indvars.iv92.i, %for.body90.lr.ph.i ]
  %indvars.iv52.i = phi i32 [ %indvars.iv.next53.i, %for.inc1585.i ], [ %indvars.iv50.i, %for.body90.lr.ph.i ]
  %indvars.iv48.i = phi i32 [ %indvars.iv.next49.i, %for.inc1585.i ], [ %indvars.iv46.i, %for.body90.lr.ph.i ]
  %indvars.iv42.i = phi i32 [ %indvars.iv.next43.i, %for.inc1585.i ], [ %indvars.iv40.i, %for.body90.lr.ph.i ]
  %t2.036.i = phi i32 [ %inc1586.i, %for.inc1585.i ], [ %14, %for.body90.lr.ph.i ]
  %16 = icmp ult i64 %indvars.iv190, %indvars.iv194
  %umin = select i1 %16, i64 %indvars.iv190, i64 %indvars.iv194
  %17 = icmp ult i64 %umin, 999
  %umin196 = select i1 %17, i64 %umin, i64 999
  %18 = trunc i64 %umin196 to i32
  %19 = shl nsw i64 %indvars.iv94.i, 6
  %cmp94.i = icmp ult i64 %19, 4028
  br i1 %cmp94.i, label %cond.end109.i, label %cond.end109.thread.i

cond.end109.i:                                    ; preds = %for.body90.i
  %20 = trunc i64 %19 to i16
  %21 = sub nuw nsw i16 4028, %20
  %div100.neg128.i = sdiv i16 %21, -32
  %div100.neg.sext.i = sext i16 %div100.neg128.i to i32
  %22 = add nuw nsw i64 %indvars.iv94.i, %indvars.iv92.i
  %23 = sext i16 %div100.neg128.i to i64
  %cmp112.i = icmp slt i64 %22, %23
  br label %cond.end136.i

cond.end109.thread.i:                             ; preds = %for.body90.i
  %24 = trunc i64 %19 to i32
  %25 = add i32 %24, -3997
  %div108.i = sdiv i32 %25, 32
  %26 = add nuw nsw i64 %indvars.iv94.i, %indvars.iv92.i
  %27 = sext i32 %div108.i to i64
  %cmp11216.i = icmp slt i64 %26, %27
  br label %cond.end136.i

cond.end136.i:                                    ; preds = %cond.end109.thread.i, %cond.end109.i
  %.sink200 = phi i64 [ %26, %cond.end109.thread.i ], [ %22, %cond.end109.i ]
  %div108.i.sink = phi i32 [ %div108.i, %cond.end109.thread.i ], [ %div100.neg.sext.i, %cond.end109.i ]
  %cmp11216.i.sink = phi i1 [ %cmp11216.i, %cond.end109.thread.i ], [ %cmp112.i, %cond.end109.i ]
  %28 = trunc i64 %.sink200 to i32
  %spec.select26.i = select i1 %cmp11216.i.sink, i32 %div108.i.sink, i32 %28
  %29 = add nuw nsw i64 %19, 4059
  %30 = lshr i64 %29, 5
  %cmp238.i = icmp ugt i64 %30, %13
  %31 = trunc i64 %30 to i32
  %cond341.i = select i1 %cmp238.i, i32 %cond217.i, i32 %31
  %32 = shl nsw i64 %indvars.iv94.i, 5
  %mul343.i = shl nsw i32 %t2.036.i, 5
  %33 = add nuw nsw i64 %32, %10
  %add346.i = add nuw nsw i32 %mul343.i, %15
  %34 = lshr i64 %33, 5
  %div365.i = lshr i32 %add346.i, 5
  %35 = zext i32 %cond341.i to i64
  %cmp368.i = icmp ugt i64 %34, %35
  %36 = add nuw nsw i64 %32, 5028
  %37 = lshr i64 %36, 5
  %cmp62520.i = icmp ult i64 %34, %37
  %38 = trunc i64 %37 to i32
  %spec.select22.i = select i1 %cmp62520.i, i32 %div365.i, i32 %38
  %cmp625.i = icmp ule i64 %37, %35
  %cmp368.not.i = xor i1 %cmp368.i, true
  %brmerge.i = or i1 %cmp625.i, %cmp368.not.i
  %div622.mux.i = select i1 %cmp625.i, i32 %38, i32 %div365.i
  %spec.select27.i = select i1 %brmerge.i, i32 %div622.mux.i, i32 %cond341.i
  %39 = add nsw i64 %32, -3998
  %cmp1122.i = icmp sgt i64 %5, %39
  %cond1130.v.i = select i1 %cmp1122.i, i64 %5, i64 %39
  %cond1130.i = trunc i64 %cond1130.v.i to i32
  %40 = or i64 %32, 30
  %cmp1221.i = icmp ult i64 %cond1218125.i, %40
  %cond1238.v.i = select i1 %cmp1221.i, i64 %cond1218125.i, i64 %40
  %cond1238.i = trunc i64 %cond1238.v.i to i32
  %41 = or i64 %32, 31
  %42 = shl i32 %spec.select26.i, 5
  %43 = add i32 %42, -3998
  %44 = add i32 %42, %indvars.iv52.i
  %45 = shl i32 %spec.select26.i, 4
  %46 = add i32 %45, -3998
  %47 = zext i32 %spec.select26.i to i64
  %48 = icmp sgt i32 %indvars.iv40.i, %indvars.iv48.i
  %smax81.i = select i1 %48, i32 %indvars.iv40.i, i32 %indvars.iv48.i
  %cond1115.i = select i1 %cmp368.i, i32 %spec.select27.i, i32 %spec.select22.i
  %49 = sext i32 %cond1115.i to i64
  %50 = trunc i64 %indvars.iv94.i to i32
  %51 = mul i32 %50, -32
  %52 = icmp sgt i32 %smax81.i, %44
  %smax82.i175 = select i1 %52, i32 %smax81.i, i32 %44
  %53 = icmp sgt i32 %smax82.i175, %46
  %smax83.i176 = select i1 %53, i32 %smax82.i175, i32 %46
  %cmp1116.not.i177 = icmp sgt i64 %47, %49
  br i1 %cmp1116.not.i177, label %for.inc1585.i, label %for.body1117.i.preheader

for.body1117.i.preheader:                         ; preds = %cond.end136.i
  %54 = sub i32 %43, %smax83.i176
  %55 = add i32 %42, %indvars.iv184
  %56 = or i32 %45, 14
  %57 = icmp sgt i64 %49, %47
  %smax198 = select i1 %57, i64 %49, i64 %47
  br label %for.body1117.i

for.body1117.i:                                   ; preds = %for.inc1582.i, %for.body1117.i.preheader
  %indvars.iv188 = phi i32 [ %56, %for.body1117.i.preheader ], [ %indvars.iv.next189, %for.inc1582.i ]
  %indvars.iv186 = phi i32 [ %55, %for.body1117.i.preheader ], [ %indvars.iv.next187, %for.inc1582.i ]
  %58 = phi i32 [ %54, %for.body1117.i.preheader ], [ %104, %for.inc1582.i ]
  %smax83.i183 = phi i32 [ %smax83.i176, %for.body1117.i.preheader ], [ %smax83.i, %for.inc1582.i ]
  %indvars.iv.i169182 = phi i32 [ %42, %for.body1117.i.preheader ], [ %indvars.iv.next.i170, %for.inc1582.i ]
  %indvars.iv44.i181 = phi i32 [ %43, %for.body1117.i.preheader ], [ %indvars.iv.next45.i, %for.inc1582.i ]
  %indvars.iv54.i180 = phi i32 [ %44, %for.body1117.i.preheader ], [ %indvars.iv.next55.i, %for.inc1582.i ]
  %indvars.iv57.i179 = phi i32 [ %46, %for.body1117.i.preheader ], [ %indvars.iv.next58.i, %for.inc1582.i ]
  %indvars.iv90.i178 = phi i64 [ %47, %for.body1117.i.preheader ], [ %indvars.iv.next91.i, %for.inc1582.i ]
  %59 = icmp slt i32 %indvars.iv186, %indvars.iv188
  %smin = select i1 %59, i32 %indvars.iv186, i32 %indvars.iv188
  %60 = icmp slt i32 %smin, %18
  %smin197 = select i1 %60, i32 %smin, i32 %18
  %61 = sext i32 %smin197 to i64
  %62 = trunc i64 %indvars.iv90.i178 to i32
  %mul1131.i = shl nsw i32 %62, 4
  %add1133.i = add nsw i32 %mul1131.i, -3998
  %cmp1134.i = icmp slt i32 %add1133.i, %cond1130.i
  %cond1154.i = select i1 %cmp1134.i, i32 %cond1130.i, i32 %add1133.i
  %mul1156.i = shl nsw i32 %62, 5
  %add1157.i = add nsw i32 %mul1156.i, %51
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
  %63 = zext i32 %smax83.i183 to i64
  %add1451.i = or i32 %mul1156.i, 31
  %64 = sext i32 %mul1156.i to i64
  %65 = sext i32 %add1451.i to i64
  %66 = icmp sgt i64 %61, %63
  %smax = select i1 %66, i64 %61, i64 %63
  br label %for.body1364.i

for.cond1207.loopexit.i:                          ; preds = %for.inc1576.i, %for.body1364.i
  %indvars.iv.next61.i = add i32 %indvars.iv60.i, -1
  %exitcond.not = icmp eq i64 %indvars.iv84.i, %smax
  br i1 %exitcond.not, label %for.inc1582.i, label %for.body1364.i, !llvm.loop !9

for.body1364.i:                                   ; preds = %for.cond1207.loopexit.i, %for.body1364.lr.ph.i
  %indvars.iv84.i = phi i64 [ %63, %for.body1364.lr.ph.i ], [ %indvars.iv.next85.i, %for.cond1207.loopexit.i ]
  %indvars.iv63.in.i = phi i32 [ %smax83.i183, %for.body1364.lr.ph.i ], [ %indvars.iv63.i, %for.cond1207.loopexit.i ]
  %indvars.iv60.i = phi i32 [ %58, %for.body1364.lr.ph.i ], [ %indvars.iv.next61.i, %for.cond1207.loopexit.i ]
  %indvars.iv63.i = add nuw i32 %indvars.iv63.in.i, 1
  %67 = icmp sgt i32 %indvars.iv42.i, %indvars.iv60.i
  %smax71.i = select i1 %67, i32 %indvars.iv42.i, i32 %indvars.iv60.i
  %68 = icmp sgt i32 %smax71.i, %indvars.iv63.i
  %smax72.i = select i1 %68, i32 %smax71.i, i32 %indvars.iv63.i
  %indvars.iv.next85.i = add nuw nsw i64 %indvars.iv84.i, 1
  %cmp1367.i = icmp ugt i64 %32, %indvars.iv.next85.i
  %cond1373.v.i = select i1 %cmp1367.i, i64 %32, i64 %indvars.iv.next85.i
  %69 = sub nsw i64 %64, %indvars.iv84.i
  %70 = add nsw i64 %69, -3998
  %sext.i = shl i64 %cond1373.v.i, 32
  %71 = ashr exact i64 %sext.i, 32
  %cmp1378.i = icmp sgt i64 %71, %70
  %cond1395.v.i = select i1 %cmp1378.i, i64 %cond1373.v.i, i64 %70
  %cond1395.i = trunc i64 %cond1395.v.i to i32
  %72 = add nsw i64 %69, 30
  %cmp1402.i = icmp slt i64 %41, %72
  %cond1411.v.i = select i1 %cmp1402.i, i64 %41, i64 %72
  %73 = add nuw nsw i64 %indvars.iv84.i, 3998
  %sext126.i = shl i64 %cond1411.v.i, 32
  %74 = ashr exact i64 %sext126.i, 32
  %cmp1414.i = icmp slt i64 %74, %73
  %cond1435.v.i = select i1 %cmp1414.i, i64 %cond1411.v.i, i64 %73
  %cond1435.i = trunc i64 %cond1435.v.i to i32
  %cmp1436.not30.i = icmp sgt i32 %cond1395.i, %cond1435.i
  br i1 %cmp1436.not30.i, label %for.cond1207.loopexit.i, label %for.body1437.lr.ph.i

for.body1437.lr.ph.i:                             ; preds = %for.body1364.i
  %75 = add i32 %smax72.i, %indvars.iv63.i
  %76 = zext i32 %smax72.i to i64
  %sext127.i = shl i64 %cond1435.v.i, 32
  %77 = ashr exact i64 %sext127.i, 32
  %78 = trunc i64 %indvars.iv84.i to i32
  br label %for.body1437.i

for.body1437.i:                                   ; preds = %for.inc1576.i, %for.body1437.lr.ph.i
  %indvars.iv73.i = phi i64 [ %76, %for.body1437.lr.ph.i ], [ %indvars.iv.next74.i, %for.inc1576.i ]
  %indvars.iv66.i = phi i32 [ %75, %for.body1437.lr.ph.i ], [ %indvars.iv.next67.i, %for.inc1576.i ]
  %79 = add nuw nsw i64 %indvars.iv73.i, %indvars.iv84.i
  %80 = add nuw nsw i64 %79, 1
  %cmp1441.i = icmp slt i64 %80, %64
  %81 = trunc i64 %80 to i32
  %cond1448.i = select i1 %cmp1441.i, i32 %mul1156.i, i32 %81
  %82 = add nuw nsw i64 %79, 3998
  %cmp1455.i = icmp sgt i64 %82, %65
  %83 = trunc i64 %82 to i32
  %cond1464.i = select i1 %cmp1455.i, i32 %add1451.i, i32 %83
  %cmp1465.not28.i = icmp sgt i32 %cond1448.i, %cond1464.i
  br i1 %cmp1465.not28.i, label %for.inc1576.i, label %for.body1466.lr.ph.i

for.body1466.lr.ph.i:                             ; preds = %for.body1437.i
  %84 = icmp sgt i32 %indvars.iv.i169182, %indvars.iv66.i
  %smax68.i = select i1 %84, i32 %indvars.iv.i169182, i32 %indvars.iv66.i
  %85 = sext i32 %smax68.i to i64
  %86 = sub nsw i64 %indvars.iv73.i, %indvars.iv84.i
  %87 = add nsw i64 %86, -1
  %88 = trunc i64 %indvars.iv73.i to i32
  %89 = add nsw i64 %86, 1
  %90 = sext i32 %cond1464.i to i64
  %91 = add i32 %88, %78
  br label %for.body1466.i

for.body1466.i:                                   ; preds = %for.body1466.i, %for.body1466.lr.ph.i
  %indvars.iv69.i = phi i64 [ %85, %for.body1466.lr.ph.i ], [ %indvars.iv.next70.i, %for.body1466.i ]
  %92 = trunc i64 %indvars.iv69.i to i32
  %add1472.i = sub i32 %92, %91
  %sub1473.i = add nsw i32 %add1472.i, -1
  %idxprom1474.i = sext i32 %sub1473.i to i64
  %arrayidx1475.i = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %87, i64 %idxprom1474.i
  %93 = load double, double* %arrayidx1475.i, align 8, !tbaa !2
  %idxprom1484.i = sext i32 %add1472.i to i64
  %arrayidx1485.i = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %87, i64 %idxprom1484.i
  %94 = load double, double* %arrayidx1485.i, align 8, !tbaa !2
  %add1486.i = fadd double %93, %94
  %add1495.i = add nsw i32 %add1472.i, 1
  %idxprom1496.i = sext i32 %add1495.i to i64
  %arrayidx1497.i = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %87, i64 %idxprom1496.i
  %95 = load double, double* %arrayidx1497.i, align 8, !tbaa !2
  %add1498.i = fadd double %add1486.i, %95
  %arrayidx1508.i = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %86, i64 %idxprom1474.i
  %96 = load double, double* %arrayidx1508.i, align 8, !tbaa !2
  %add1509.i = fadd double %add1498.i, %96
  %arrayidx1518.i = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %86, i64 %idxprom1484.i
  %97 = load double, double* %arrayidx1518.i, align 8, !tbaa !2
  %add1519.i = fadd double %add1509.i, %97
  %arrayidx1529.i = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %86, i64 %idxprom1496.i
  %98 = load double, double* %arrayidx1529.i, align 8, !tbaa !2
  %add1530.i = fadd double %add1519.i, %98
  %arrayidx1541.i = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %89, i64 %idxprom1474.i
  %99 = load double, double* %arrayidx1541.i, align 8, !tbaa !2
  %add1542.i = fadd double %add1530.i, %99
  %arrayidx1552.i = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %89, i64 %idxprom1484.i
  %100 = load double, double* %arrayidx1552.i, align 8, !tbaa !2
  %add1553.i = fadd double %add1542.i, %100
  %arrayidx1564.i = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %89, i64 %idxprom1496.i
  %101 = load double, double* %arrayidx1564.i, align 8, !tbaa !2
  %add1565.i = fadd double %add1553.i, %101
  %div1566.i = fdiv double %add1565.i, 9.000000e+00
  store double %div1566.i, double* %arrayidx1518.i, align 8, !tbaa !2
  %indvars.iv.next70.i = add nsw i64 %indvars.iv69.i, 1
  %cmp1465.not.not.i = icmp slt i64 %indvars.iv69.i, %90
  br i1 %cmp1465.not.not.i, label %for.body1466.i, label %for.inc1576.i, !llvm.loop !10

for.inc1576.i:                                    ; preds = %for.body1466.i, %for.body1437.i
  %indvars.iv.next74.i = add nuw nsw i64 %indvars.iv73.i, 1
  %cmp1436.not.not.i = icmp slt i64 %indvars.iv73.i, %77
  %indvars.iv.next67.i = add i32 %indvars.iv66.i, 1
  br i1 %cmp1436.not.not.i, label %for.body1437.i, label %for.cond1207.loopexit.i, !llvm.loop !11

for.inc1582.i:                                    ; preds = %for.cond1207.loopexit.i, %for.body1117.i
  %indvars.iv.next91.i = add nuw nsw i64 %indvars.iv90.i178, 1
  %indvars.iv.next.i170 = add i32 %indvars.iv.i169182, 32
  %indvars.iv.next45.i = add i32 %indvars.iv44.i181, 32
  %indvars.iv.next55.i = add i32 %indvars.iv54.i180, 32
  %indvars.iv.next58.i = add i32 %indvars.iv57.i179, 16
  %102 = icmp sgt i32 %smax81.i, %indvars.iv.next55.i
  %smax82.i = select i1 %102, i32 %smax81.i, i32 %indvars.iv.next55.i
  %103 = icmp sgt i32 %smax82.i, %indvars.iv.next58.i
  %smax83.i = select i1 %103, i32 %smax82.i, i32 %indvars.iv.next58.i
  %104 = sub i32 %indvars.iv.next45.i, %smax83.i
  %indvars.iv.next187 = add i32 %indvars.iv186, 32
  %indvars.iv.next189 = add i32 %indvars.iv188, 16
  %exitcond199.not = icmp eq i64 %indvars.iv90.i178, %smax198
  br i1 %exitcond199.not, label %for.inc1585.i, label %for.body1117.i, !llvm.loop !12

for.inc1585.i:                                    ; preds = %for.inc1582.i, %cond.end136.i
  %indvars.iv.next95.i = add nuw nsw i64 %indvars.iv94.i, 1
  %inc1586.i = add nuw nsw i32 %t2.036.i, 1
  %indvars.iv.next43.i = add nuw nsw i32 %indvars.iv42.i, 32
  %indvars.iv.next49.i = add nsw i32 %indvars.iv48.i, 32
  %indvars.iv.next53.i = add nsw i32 %indvars.iv52.i, -32
  %exitcond.not.i171 = icmp eq i64 %indvars.iv.next95.i, %indvars.iv116.i
  %indvars.iv.next185 = add nsw i32 %indvars.iv184, -32
  %indvars.iv.next195 = add nuw nsw i64 %indvars.iv194, 32
  br i1 %exitcond.not.i171, label %for.inc1588.i, label %for.body90.i, !llvm.loop !13

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
  br i1 %exitcond124.not.i, label %kernel_seidel_2d.exit, label %for.body90.lr.ph.i, !llvm.loop !14

kernel_seidel_2d.exit:                            ; preds = %for.inc1588.i
  tail call void (...) @polybench_timer_stop() #5
  tail call void (...) @polybench_timer_print() #5
  %cmp = icmp sgt i32 %argc, 42
  br i1 %cmp, label %if.end123, label %if.end138

if.end123:                                        ; preds = %kernel_seidel_2d.exit
  %105 = load i8*, i8** %argv, align 8, !tbaa !15
  %106 = load i8, i8* %105, align 1, !tbaa !17
  %phi.cmp = icmp eq i8 %106, 0
  br i1 %phi.cmp, label %if.then136, label %if.end138

if.then136:                                       ; preds = %if.end123
  %107 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !15
  %108 = tail call i64 @fwrite(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.1, i64 0, i64 0), i64 22, i64 1, %struct._IO_FILE* %107) #6
  %109 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !15
  %call1.i = tail call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %109, i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str.2, i64 0, i64 0), i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.3, i64 0, i64 0)) #6
  br label %for.cond2.preheader.i

for.cond2.preheader.i:                            ; preds = %for.inc10.i, %if.then136
  %indvars.iv4.i163 = phi i64 [ 0, %if.then136 ], [ %indvars.iv.next5.i168, %for.inc10.i ]
  %110 = mul nuw nsw i64 %indvars.iv4.i163, 4000
  br label %for.body4.i

for.body4.i:                                      ; preds = %if.end.i, %for.cond2.preheader.i
  %indvars.iv.i164 = phi i64 [ 0, %for.cond2.preheader.i ], [ %indvars.iv.next.i166, %if.end.i ]
  %111 = add nuw nsw i64 %indvars.iv.i164, %110
  %112 = trunc i64 %111 to i32
  %rem.i = urem i32 %112, 20
  %cmp5.i = icmp eq i32 %rem.i, 0
  br i1 %cmp5.i, label %if.then.i, label %if.end.i

if.then.i:                                        ; preds = %for.body4.i
  %113 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !15
  %fputc.i = tail call i32 @fputc(i32 10, %struct._IO_FILE* %113) #5
  br label %if.end.i

if.end.i:                                         ; preds = %if.then.i, %for.body4.i
  %114 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !15
  %arrayidx8.i165 = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %indvars.iv4.i163, i64 %indvars.iv.i164
  %115 = load double, double* %arrayidx8.i165, align 8, !tbaa !2
  %call9.i = tail call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %114, i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.5, i64 0, i64 0), double %115) #6
  %indvars.iv.next.i166 = add nuw nsw i64 %indvars.iv.i164, 1
  %exitcond.not.i167 = icmp eq i64 %indvars.iv.next.i166, 4000
  br i1 %exitcond.not.i167, label %for.inc10.i, label %for.body4.i, !llvm.loop !18

for.inc10.i:                                      ; preds = %if.end.i
  %indvars.iv.next5.i168 = add nuw nsw i64 %indvars.iv4.i163, 1
  %exitcond7.not.i = icmp eq i64 %indvars.iv.next5.i168, 4000
  br i1 %exitcond7.not.i, label %print_array.exit, label %for.cond2.preheader.i, !llvm.loop !19

print_array.exit:                                 ; preds = %for.inc10.i
  %116 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !15
  %call13.i = tail call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %116, i8* getelementptr inbounds ([17 x i8], [17 x i8]* @.str.6, i64 0, i64 0), i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.3, i64 0, i64 0)) #6
  %117 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !15
  %118 = tail call i64 @fwrite(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.7, i64 0, i64 0), i64 22, i64 1, %struct._IO_FILE* %117) #6
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
!10 = distinct !{!10, !7}
!11 = distinct !{!11, !7}
!12 = distinct !{!12, !7}
!13 = distinct !{!13, !7}
!14 = distinct !{!14, !7}
!15 = !{!16, !16, i64 0}
!16 = !{!"any pointer", !4, i64 0}
!17 = !{!4, !4, i64 0}
!18 = distinct !{!18, !7}
!19 = distinct !{!19, !7}
