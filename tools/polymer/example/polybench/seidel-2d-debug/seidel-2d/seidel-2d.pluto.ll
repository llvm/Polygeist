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

for.body90.lr.ph.i:                               ; preds = %for.inc1569.i, %init_array.exit
  %indvars.iv.i177 = phi i64 [ 126, %init_array.exit ], [ %indvars.iv.next.i180, %for.inc1569.i ]
  %t1.038.i = phi i64 [ 0, %init_array.exit ], [ %inc1570.i, %for.inc1569.i ]
  %5 = shl nuw nsw i64 %t1.038.i, 5
  %6 = add nsw i64 %5, -3998
  %7 = mul nsw i64 %t1.038.i, -32
  %8 = add i64 %7, -4029
  %mul.i178 = shl nsw i64 %t1.038.i, 5
  %add29.i = add nuw nsw i64 %mul.i178, 4029
  %div173.i = lshr i64 %add29.i, 4
  %9 = icmp ult i64 %div173.i, 312
  %cond217.i = select i1 %9, i64 %div173.i, i64 312
  %add344.i = add nuw nsw i64 %mul.i178, 4060
  %add1210.i = or i64 %mul.i178, 31
  %10 = icmp ult i64 %add1210.i, 999
  %cond1218.i = select i1 %10, i64 %add1210.i, i64 999
  br label %for.body90.i

for.body90.i:                                     ; preds = %for.inc1566.i, %for.body90.lr.ph.i
  %indvar = phi i64 [ %indvar.next, %for.inc1566.i ], [ 0, %for.body90.lr.ph.i ]
  %t2.036.i = phi i64 [ %inc1567.i, %for.inc1566.i ], [ %t1.038.i, %for.body90.lr.ph.i ]
  %11 = shl nuw nsw i64 %indvar, 5
  %12 = add i64 %5, %11
  %13 = add i64 %6, %11
  %14 = mul nsw i64 %indvar, -32
  %15 = add i64 %8, %14
  %mul91.i = shl nsw i64 %t2.036.i, 6
  %cmp94.i = icmp ult i64 %mul91.i, 4028
  br i1 %cmp94.i, label %cond.end109.i, label %cond.end109.thread.i

cond.end109.i:                                    ; preds = %for.body90.i
  %16 = trunc i64 %mul91.i to i16
  %div100.neg.lhs.trunc.i = sub nuw nsw i16 4028, %16
  %div100.neg41.i = sdiv i16 %div100.neg.lhs.trunc.i, -32
  %div100.neg.sext.i = sext i16 %div100.neg41.i to i64
  %add111.i = add nuw nsw i64 %t2.036.i, %t1.038.i
  %cmp112.i = icmp slt i64 %add111.i, %div100.neg.sext.i
  %spec.select25.i = select i1 %cmp112.i, i64 %div100.neg.sext.i, i64 %add111.i
  br label %cond.end136.i

cond.end109.thread.i:                             ; preds = %for.body90.i
  %sub107.i = add nsw i64 %mul91.i, -3997
  %div108.i = sdiv i64 %sub107.i, 32
  %add11115.i = add nuw nsw i64 %t2.036.i, %t1.038.i
  %cmp11216.i = icmp sgt i64 %div108.i, %add11115.i
  %spec.select26.i = select i1 %cmp11216.i, i64 %div108.i, i64 %add11115.i
  br label %cond.end136.i

cond.end136.i:                                    ; preds = %cond.end109.thread.i, %cond.end109.i
  %cond137.i = phi i64 [ %spec.select25.i, %cond.end109.i ], [ %spec.select26.i, %cond.end109.thread.i ]
  %add220.i = add nuw nsw i64 %mul91.i, 4059
  %div235.i = lshr i64 %add220.i, 5
  %cmp238.i = icmp ult i64 %cond217.i, %div235.i
  %cond341.i = select i1 %cmp238.i, i64 %cond217.i, i64 %div235.i
  %mul343.i = shl nsw i64 %t2.036.i, 5
  %add346.i = add nuw nsw i64 %add344.i, %mul343.i
  %div365.i = lshr i64 %add346.i, 5
  %cmp368.i = icmp ult i64 %cond341.i, %div365.i
  %add60518.i = add nuw nsw i64 %mul343.i, 5028
  %div62219.i = lshr i64 %add60518.i, 5
  %cmp62520.i = icmp ult i64 %div365.i, %div62219.i
  %spec.select22.i = select i1 %cmp62520.i, i64 %div365.i, i64 %div62219.i
  %cmp625.i = icmp uge i64 %cond341.i, %div62219.i
  %cmp368.not.i = xor i1 %cmp368.i, true
  %brmerge.i = or i1 %cmp625.i, %cmp368.not.i
  %div622.mux.i = select i1 %cmp625.i, i64 %div62219.i, i64 %div365.i
  %spec.select27.i = select i1 %brmerge.i, i64 %div622.mux.i, i64 %cond341.i
  %add1121.i = add nsw i64 %mul343.i, -3998
  %cmp1122.i = icmp sgt i64 %mul.i178, %add1121.i
  %cond1130.i = select i1 %cmp1122.i, i64 %mul.i178, i64 %add1121.i
  %mul1155.i = mul nsw i64 %t2.036.i, -32
  %add1220.i = or i64 %mul343.i, 30
  %cmp1221.i = icmp ult i64 %cond1218.i, %add1220.i
  %cond1238.i = select i1 %cmp1221.i, i64 %cond1218.i, i64 %add1220.i
  %add1398.i = or i64 %mul343.i, 31
  %cond1115.i = select i1 %cmp368.i, i64 %spec.select27.i, i64 %spec.select22.i
  %cmp1116.not.i182 = icmp sgt i64 %cond137.i, %cond1115.i
  br i1 %cmp1116.not.i182, label %for.inc1566.i, label %for.body1117.i.preheader

for.body1117.i.preheader:                         ; preds = %cond.end136.i
  %17 = icmp sgt i64 %cond1115.i, %cond137.i
  %smax = select i1 %17, i64 %cond1115.i, i64 %cond137.i
  %18 = shl nsw i64 %cond137.i, 5
  %19 = add nsw i64 %18, -3998
  %20 = add i64 %15, %18
  %21 = shl nsw i64 %cond137.i, 4
  %22 = add nsw i64 %21, -3998
  %23 = icmp sgt i64 %5, %13
  %smax186 = select i1 %23, i64 %5, i64 %13
  br label %for.body1117.i

for.body1117.i:                                   ; preds = %for.body1117.i.preheader, %for.inc1563.i
  %indvar184 = phi i64 [ 0, %for.body1117.i.preheader ], [ %indvar.next185, %for.inc1563.i ]
  %t3.0.i183 = phi i64 [ %cond137.i, %for.body1117.i.preheader ], [ %inc1564.i, %for.inc1563.i ]
  %24 = shl i64 %indvar184, 5
  %25 = add i64 %19, %24
  %26 = add i64 %20, %24
  %27 = icmp sgt i64 %smax186, %26
  %smax187 = select i1 %27, i64 %smax186, i64 %26
  %28 = shl i64 %indvar184, 4
  %29 = add i64 %22, %28
  %30 = icmp sgt i64 %smax187, %29
  %smax188 = select i1 %30, i64 %smax187, i64 %29
  %31 = sub i64 %25, %smax188
  %32 = add nuw i64 %smax188, 1
  %33 = add i64 %18, %24
  %34 = xor i64 %smax188, -1
  %mul1131.i = shl nsw i64 %t3.0.i183, 4
  %add1133.i = add nsw i64 %mul1131.i, -3998
  %cmp1134.i = icmp sgt i64 %cond1130.i, %add1133.i
  %cond1154.i = select i1 %cmp1134.i, i64 %cond1130.i, i64 %add1133.i
  %mul1156.i = shl nsw i64 %t3.0.i183, 5
  %add1157.i = add nsw i64 %mul1156.i, %mul1155.i
  %sub1159.i = add nsw i64 %add1157.i, -4029
  %cmp1160.i = icmp sgt i64 %cond1154.i, %sub1159.i
  %cond1206.i = select i1 %cmp1160.i, i64 %cond1154.i, i64 %sub1159.i
  %add1240.i = or i64 %mul1131.i, 14
  %cmp1241.i = icmp slt i64 %cond1238.i, %add1240.i
  %cond1278.i = select i1 %cmp1241.i, i64 %cond1238.i, i64 %add1240.i
  %add1282.i = or i64 %add1157.i, 30
  %cmp1283.i = icmp slt i64 %cond1278.i, %add1282.i
  %cond1362.i = select i1 %cmp1283.i, i64 %cond1278.i, i64 %add1282.i
  %cmp1363.not33.i = icmp sgt i64 %cond1206.i, %cond1362.i
  br i1 %cmp1363.not33.i, label %for.inc1563.i, label %for.body1364.lr.ph.i

for.body1364.lr.ph.i:                             ; preds = %for.body1117.i
  %add1451.i = or i64 %mul1156.i, 31
  br label %for.body1364.i

for.cond1207.loopexit.i:                          ; preds = %for.inc1557.i, %for.body1364.i
  %cmp1363.not.not.i = icmp slt i64 %t4.034.i, %cond1362.i
  %indvar.next190 = add i64 %indvar189, 1
  br i1 %cmp1363.not.not.i, label %for.body1364.i, label %for.inc1563.i, !llvm.loop !9

for.body1364.i:                                   ; preds = %for.cond1207.loopexit.i, %for.body1364.lr.ph.i
  %indvar189 = phi i64 [ %indvar.next190, %for.cond1207.loopexit.i ], [ 0, %for.body1364.lr.ph.i ]
  %t4.034.i = phi i64 [ %add1366.i, %for.cond1207.loopexit.i ], [ %cond1206.i, %for.body1364.lr.ph.i ]
  %35 = sub i64 %31, %indvar189
  %36 = icmp sgt i64 %12, %35
  %smax191 = select i1 %36, i64 %12, i64 %35
  %37 = add i64 %32, %indvar189
  %38 = icmp sgt i64 %smax191, %37
  %smax192 = select i1 %38, i64 %smax191, i64 %37
  %39 = add i64 %smax188, %indvar189
  %40 = sub i64 %smax192, %39
  %41 = add i64 %smax192, %37
  %42 = sub i64 %34, %indvar189
  %43 = sub i64 %42, %smax192
  %add1366.i = add nuw nsw i64 %t4.034.i, 1
  %cmp1367.i = icmp sgt i64 %mul343.i, %add1366.i
  %cond1373.i = select i1 %cmp1367.i, i64 %mul343.i, i64 %add1366.i
  %sub1375.i = sub nsw i64 %mul1156.i, %t4.034.i
  %add1377.i = add nsw i64 %sub1375.i, -3998
  %cmp1378.i = icmp sgt i64 %cond1373.i, %add1377.i
  %cond1395.i = select i1 %cmp1378.i, i64 %cond1373.i, i64 %add1377.i
  %add1401.i = add nsw i64 %sub1375.i, 30
  %cmp1402.i = icmp slt i64 %add1398.i, %add1401.i
  %cond1411.i = select i1 %cmp1402.i, i64 %add1398.i, i64 %add1401.i
  %sub1413.i = add nuw nsw i64 %t4.034.i, 3998
  %cmp1414.i = icmp slt i64 %cond1411.i, %sub1413.i
  %cond1435.i = select i1 %cmp1414.i, i64 %cond1411.i, i64 %sub1413.i
  %cmp1436.not30.i = icmp sgt i64 %cond1395.i, %cond1435.i
  br i1 %cmp1436.not30.i, label %for.cond1207.loopexit.i, label %for.body1437.i

for.body1437.i:                                   ; preds = %for.body1364.i, %for.inc1557.i
  %indvar193 = phi i64 [ %indvar.next194, %for.inc1557.i ], [ 0, %for.body1364.i ]
  %t5.031.i = phi i64 [ %inc1558.i, %for.inc1557.i ], [ %cond1395.i, %for.body1364.i ]
  %44 = add i64 %41, %indvar193
  %add1439.i = add nuw i64 %t5.031.i, %t4.034.i
  %add1440.i = add nsw i64 %add1439.i, 1
  %cmp1441.i = icmp sgt i64 %mul1156.i, %add1440.i
  %cond1448.i = select i1 %cmp1441.i, i64 %mul1156.i, i64 %add1440.i
  %sub1454.i = add nsw i64 %add1439.i, 3998
  %cmp1455.i = icmp slt i64 %add1451.i, %sub1454.i
  %cond1464.i = select i1 %cmp1455.i, i64 %add1451.i, i64 %sub1454.i
  %cmp1465.not28.i = icmp sgt i64 %cond1448.i, %cond1464.i
  br i1 %cmp1465.not28.i, label %for.inc1557.i, label %for.body1466.lr.ph.i

for.body1466.lr.ph.i:                             ; preds = %for.body1437.i
  %45 = add i64 %40, %indvar193
  %46 = mul i64 %45, 32000
  %scevgep = getelementptr i8, i8* %call, i64 %46
  %47 = icmp sgt i64 %33, %44
  %smax195 = select i1 %47, i64 %33, i64 %44
  %48 = sub i64 %43, %indvar193
  %49 = add i64 %smax195, %48
  %50 = shl i64 %49, 3
  %scevgep196 = getelementptr i8, i8* %scevgep, i64 %50
  %scevgep196197 = bitcast i8* %scevgep196 to double*
  %add1468.i = sub nsw i64 %t5.031.i, %t4.034.i
  %sub1469.i = add nsw i64 %add1468.i, -1
  %add1522.i = add nsw i64 %add1468.i, 1
  %load_initial = load double, double* %scevgep196197, align 8
  br label %for.body1466.i

for.body1466.i:                                   ; preds = %for.body1466.i, %for.body1466.lr.ph.i
  %store_forwarded = phi double [ %load_initial, %for.body1466.lr.ph.i ], [ %div1549.i, %for.body1466.i ]
  %t6.029.i = phi i64 [ %cond1448.i, %for.body1466.lr.ph.i ], [ %inc.i, %for.body1466.i ]
  %add1472.i = sub i64 %t6.029.i, %add1439.i
  %sub1473.i = add nsw i64 %add1472.i, -1
  %arrayidx1474.i = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %sub1469.i, i64 %sub1473.i
  %51 = load double, double* %arrayidx1474.i, align 8, !tbaa !2
  %arrayidx1482.i = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %sub1469.i, i64 %add1472.i
  %52 = load double, double* %arrayidx1482.i, align 8, !tbaa !2
  %add1483.i = fadd double %51, %52
  %add1491.i = add nsw i64 %add1472.i, 1
  %arrayidx1492.i = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %sub1469.i, i64 %add1491.i
  %53 = load double, double* %arrayidx1492.i, align 8, !tbaa !2
  %add1493.i = fadd double %add1483.i, %53
  %add1502.i = fadd double %add1493.i, %store_forwarded
  %arrayidx1509.i = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %add1468.i, i64 %add1472.i
  %54 = load double, double* %arrayidx1509.i, align 8, !tbaa !2
  %add1510.i = fadd double %add1502.i, %54
  %arrayidx1518.i = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %add1468.i, i64 %add1491.i
  %55 = load double, double* %arrayidx1518.i, align 8, !tbaa !2
  %add1519.i = fadd double %add1510.i, %55
  %arrayidx1528.i = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %add1522.i, i64 %sub1473.i
  %56 = load double, double* %arrayidx1528.i, align 8, !tbaa !2
  %add1529.i = fadd double %add1519.i, %56
  %arrayidx1537.i = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %add1522.i, i64 %add1472.i
  %57 = load double, double* %arrayidx1537.i, align 8, !tbaa !2
  %add1538.i = fadd double %add1529.i, %57
  %arrayidx1547.i = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %add1522.i, i64 %add1491.i
  %58 = load double, double* %arrayidx1547.i, align 8, !tbaa !2
  %add1548.i = fadd double %add1538.i, %58
  %div1549.i = fdiv double %add1548.i, 9.000000e+00
  store double %div1549.i, double* %arrayidx1509.i, align 8, !tbaa !2
  %inc.i = add nsw i64 %t6.029.i, 1
  %cmp1465.not.not.i = icmp slt i64 %t6.029.i, %cond1464.i
  br i1 %cmp1465.not.not.i, label %for.body1466.i, label %for.inc1557.i, !llvm.loop !10

for.inc1557.i:                                    ; preds = %for.body1466.i, %for.body1437.i
  %inc1558.i = add nuw nsw i64 %t5.031.i, 1
  %cmp1436.not.not.i = icmp slt i64 %t5.031.i, %cond1435.i
  %indvar.next194 = add i64 %indvar193, 1
  br i1 %cmp1436.not.not.i, label %for.body1437.i, label %for.cond1207.loopexit.i, !llvm.loop !11

for.inc1563.i:                                    ; preds = %for.cond1207.loopexit.i, %for.body1117.i
  %inc1564.i = add nuw nsw i64 %t3.0.i183, 1
  %exitcond.not = icmp eq i64 %t3.0.i183, %smax
  %indvar.next185 = add i64 %indvar184, 1
  br i1 %exitcond.not, label %for.inc1566.i, label %for.body1117.i, !llvm.loop !12

for.inc1566.i:                                    ; preds = %for.inc1563.i, %cond.end136.i
  %inc1567.i = add nuw nsw i64 %t2.036.i, 1
  %exitcond.not.i179 = icmp eq i64 %inc1567.i, %indvars.iv.i177
  %indvar.next = add i64 %indvar, 1
  br i1 %exitcond.not.i179, label %for.inc1569.i, label %for.body90.i, !llvm.loop !13

for.inc1569.i:                                    ; preds = %for.inc1566.i
  %inc1570.i = add nuw nsw i64 %t1.038.i, 1
  %indvars.iv.next.i180 = add nuw nsw i64 %indvars.iv.i177, 1
  %exitcond40.not.i = icmp eq i64 %inc1570.i, 32
  br i1 %exitcond40.not.i, label %kernel_seidel_2d.exit, label %for.body90.lr.ph.i, !llvm.loop !14

kernel_seidel_2d.exit:                            ; preds = %for.inc1569.i
  tail call void (...) @polybench_timer_stop() #5
  tail call void (...) @polybench_timer_print() #5
  %cmp = icmp sgt i32 %argc, 42
  br i1 %cmp, label %if.end131, label %if.end146

if.end131:                                        ; preds = %kernel_seidel_2d.exit
  %59 = load i8*, i8** %argv, align 8, !tbaa !15
  %60 = load i8, i8* %59, align 1, !tbaa !17
  %phi.cmp = icmp eq i8 %60, 0
  br i1 %phi.cmp, label %if.then144, label %if.end146

if.then144:                                       ; preds = %if.end131
  %61 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !15
  %62 = tail call i64 @fwrite(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.1, i64 0, i64 0), i64 22, i64 1, %struct._IO_FILE* %61) #6
  %63 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !15
  %call1.i = tail call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %63, i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str.2, i64 0, i64 0), i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.3, i64 0, i64 0)) #6
  br label %for.cond2.preheader.i

for.cond2.preheader.i:                            ; preds = %for.inc10.i, %if.then144
  %indvars.iv4.i171 = phi i64 [ 0, %if.then144 ], [ %indvars.iv.next5.i176, %for.inc10.i ]
  %64 = mul nuw nsw i64 %indvars.iv4.i171, 4000
  br label %for.body4.i

for.body4.i:                                      ; preds = %if.end.i, %for.cond2.preheader.i
  %indvars.iv.i172 = phi i64 [ 0, %for.cond2.preheader.i ], [ %indvars.iv.next.i174, %if.end.i ]
  %65 = add nuw nsw i64 %indvars.iv.i172, %64
  %66 = trunc i64 %65 to i32
  %rem.i = urem i32 %66, 20
  %cmp5.i = icmp eq i32 %rem.i, 0
  br i1 %cmp5.i, label %if.then.i, label %if.end.i

if.then.i:                                        ; preds = %for.body4.i
  %67 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !15
  %fputc.i = tail call i32 @fputc(i32 10, %struct._IO_FILE* %67) #5
  br label %if.end.i

if.end.i:                                         ; preds = %if.then.i, %for.body4.i
  %68 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !15
  %arrayidx8.i173 = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %indvars.iv4.i171, i64 %indvars.iv.i172
  %69 = load double, double* %arrayidx8.i173, align 8, !tbaa !2
  %call9.i = tail call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %68, i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.5, i64 0, i64 0), double %69) #6
  %indvars.iv.next.i174 = add nuw nsw i64 %indvars.iv.i172, 1
  %exitcond.not.i175 = icmp eq i64 %indvars.iv.next.i174, 4000
  br i1 %exitcond.not.i175, label %for.inc10.i, label %for.body4.i, !llvm.loop !18

for.inc10.i:                                      ; preds = %if.end.i
  %indvars.iv.next5.i176 = add nuw nsw i64 %indvars.iv4.i171, 1
  %exitcond7.not.i = icmp eq i64 %indvars.iv.next5.i176, 4000
  br i1 %exitcond7.not.i, label %print_array.exit, label %for.cond2.preheader.i, !llvm.loop !19

print_array.exit:                                 ; preds = %for.inc10.i
  %70 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !15
  %call13.i = tail call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %70, i8* getelementptr inbounds ([17 x i8], [17 x i8]* @.str.6, i64 0, i64 0), i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.3, i64 0, i64 0)) #6
  %71 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !15
  %72 = tail call i64 @fwrite(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.7, i64 0, i64 0), i64 22, i64 1, %struct._IO_FILE* %71) #6
  br label %if.end146

if.end146:                                        ; preds = %print_array.exit, %if.end131, %kernel_seidel_2d.exit
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
