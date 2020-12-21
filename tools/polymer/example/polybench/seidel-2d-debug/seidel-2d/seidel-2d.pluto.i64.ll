; ModuleID = '/mnt/ccnas2/bdp/rz3515/projects/polymer/example/polybench/seidel-2d-debug/seidel-2d/seidel-2d.pluto.i64.c'
source_filename = "/mnt/ccnas2/bdp/rz3515/projects/polymer/example/polybench/seidel-2d-debug/seidel-2d/seidel-2d.pluto.i64.c"
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
  br label %for.body105.lr.ph.i

for.body105.lr.ph.i:                              ; preds = %for.inc1836.i, %init_array.exit
  %indvars.iv.i169 = phi i64 [ 126, %init_array.exit ], [ %indvars.iv.next.i172, %for.inc1836.i ]
  %t1.038.i = phi i64 [ 0, %init_array.exit ], [ %inc1837.i, %for.inc1836.i ]
  %5 = shl nuw nsw i64 %t1.038.i, 5
  %6 = add nsw i64 %5, -3998
  %7 = mul nsw i64 %t1.038.i, -32
  %8 = add i64 %7, -4029
  %mul.i170 = shl nsw i64 %t1.038.i, 5
  %add33.i = add nuw nsw i64 %mul.i170, 4029
  %div203.i = lshr i64 %add33.i, 4
  %9 = icmp ult i64 %div203.i, 312
  %cond254.i = select i1 %9, i64 %div203.i, i64 312
  %add403.i = add nuw nsw i64 %mul.i170, 4060
  %add1430.i = or i64 %mul.i170, 31
  %10 = icmp ult i64 %add1430.i, 999
  %cond1440.i = select i1 %10, i64 %add1430.i, i64 999
  br label %for.body105.i

for.body105.i:                                    ; preds = %for.inc1833.i, %for.body105.lr.ph.i
  %indvar = phi i64 [ %indvar.next, %for.inc1833.i ], [ 0, %for.body105.lr.ph.i ]
  %t2.036.i = phi i64 [ %inc1834.i, %for.inc1833.i ], [ %t1.038.i, %for.body105.lr.ph.i ]
  %11 = shl nuw nsw i64 %indvar, 5
  %12 = add i64 %5, %11
  %13 = add i64 %6, %11
  %14 = mul nsw i64 %indvar, -32
  %15 = add i64 %8, %14
  %mul106.i = shl nsw i64 %t2.036.i, 6
  %cmp110.i = icmp ult i64 %mul106.i, 4028
  br i1 %cmp110.i, label %cond.end128.i, label %cond.end128.thread.i

cond.end128.i:                                    ; preds = %for.body105.i
  %16 = trunc i64 %mul106.i to i16
  %div118.neg.lhs.trunc.i = sub nuw nsw i16 4028, %16
  %div118.neg41.i = sdiv i16 %div118.neg.lhs.trunc.i, -32
  %div118.neg.sext.i = sext i16 %div118.neg41.i to i64
  %add130.i = add nuw nsw i64 %t2.036.i, %t1.038.i
  %cmp131.i = icmp slt i64 %add130.i, %div118.neg.sext.i
  %spec.select25.i = select i1 %cmp131.i, i64 %div118.neg.sext.i, i64 %add130.i
  br label %cond.end160.i

cond.end128.thread.i:                             ; preds = %for.body105.i
  %sub126.i = add nsw i64 %mul106.i, -3997
  %div127.i = sdiv i64 %sub126.i, 32
  %add13015.i = add nuw nsw i64 %t2.036.i, %t1.038.i
  %cmp13116.i = icmp sgt i64 %div127.i, %add13015.i
  %spec.select26.i = select i1 %cmp13116.i, i64 %div127.i, i64 %add13015.i
  br label %cond.end160.i

cond.end160.i:                                    ; preds = %cond.end128.thread.i, %cond.end128.i
  %cond161.i = phi i64 [ %spec.select25.i, %cond.end128.i ], [ %spec.select26.i, %cond.end128.thread.i ]
  %add258.i = add nuw nsw i64 %mul106.i, 4059
  %div276.i = lshr i64 %add258.i, 5
  %cmp279.i = icmp ult i64 %cond254.i, %div276.i
  %cond400.i = select i1 %cmp279.i, i64 %cond254.i, i64 %div276.i
  %mul402.i = shl nsw i64 %t2.036.i, 5
  %add406.i = add nuw nsw i64 %add403.i, %mul402.i
  %div428.i = lshr i64 %add406.i, 5
  %cmp431.i = icmp ult i64 %cond400.i, %div428.i
  %add71018.i = add nuw nsw i64 %mul402.i, 5028
  %div73219.i = lshr i64 %add71018.i, 5
  %cmp73520.i = icmp ult i64 %div428.i, %div73219.i
  %spec.select22.i = select i1 %cmp73520.i, i64 %div428.i, i64 %div73219.i
  %cmp735.i = icmp uge i64 %cond400.i, %div73219.i
  %cmp431.not.i = xor i1 %cmp431.i, true
  %brmerge.i = or i1 %cmp735.i, %cmp431.not.i
  %div732.mux.i = select i1 %cmp735.i, i64 %div73219.i, i64 %div428.i
  %spec.select27.i = select i1 %brmerge.i, i64 %div732.mux.i, i64 %cond400.i
  %add1320.i = add nsw i64 %mul402.i, -3998
  %cmp1321.i = icmp sgt i64 %mul.i170, %add1320.i
  %cond1331.i = select i1 %cmp1321.i, i64 %mul.i170, i64 %add1320.i
  %mul1362.i = mul nsw i64 %t2.036.i, -32
  %add1442.i = or i64 %mul402.i, 30
  %cmp1443.i = icmp ult i64 %cond1440.i, %add1442.i
  %cond1464.i = select i1 %cmp1443.i, i64 %cond1440.i, i64 %add1442.i
  %add1654.i = or i64 %mul402.i, 31
  %cond1312.i = select i1 %cmp431.i, i64 %spec.select27.i, i64 %spec.select22.i
  %cmp1313.not.i174 = icmp sgt i64 %cond161.i, %cond1312.i
  br i1 %cmp1313.not.i174, label %for.inc1833.i, label %for.body1315.i.preheader

for.body1315.i.preheader:                         ; preds = %cond.end160.i
  %17 = icmp sgt i64 %cond1312.i, %cond161.i
  %smax = select i1 %17, i64 %cond1312.i, i64 %cond161.i
  %18 = shl nsw i64 %cond161.i, 5
  %19 = add nsw i64 %18, -3998
  %20 = add i64 %15, %18
  %21 = shl nsw i64 %cond161.i, 4
  %22 = add nsw i64 %21, -3998
  %23 = icmp sgt i64 %5, %13
  %smax178 = select i1 %23, i64 %5, i64 %13
  br label %for.body1315.i

for.body1315.i:                                   ; preds = %for.body1315.i.preheader, %for.inc1830.i
  %indvar176 = phi i64 [ 0, %for.body1315.i.preheader ], [ %indvar.next177, %for.inc1830.i ]
  %t3.0.i175 = phi i64 [ %cond161.i, %for.body1315.i.preheader ], [ %inc1831.i, %for.inc1830.i ]
  %24 = shl i64 %indvar176, 5
  %25 = add i64 %19, %24
  %26 = add i64 %20, %24
  %27 = icmp sgt i64 %smax178, %26
  %smax179 = select i1 %27, i64 %smax178, i64 %26
  %28 = shl i64 %indvar176, 4
  %29 = add i64 %22, %28
  %30 = icmp sgt i64 %smax179, %29
  %smax180 = select i1 %30, i64 %smax179, i64 %29
  %31 = sub i64 %25, %smax180
  %32 = add nuw i64 %smax180, 1
  %33 = add i64 %18, %24
  %34 = xor i64 %smax180, -1
  %mul1332.i = shl nsw i64 %t3.0.i175, 4
  %add1335.i = add nsw i64 %mul1332.i, -3998
  %cmp1336.i = icmp sgt i64 %cond1331.i, %add1335.i
  %cond1361.i = select i1 %cmp1336.i, i64 %cond1331.i, i64 %add1335.i
  %mul1363.i = shl nsw i64 %t3.0.i175, 5
  %add1364.i = add nsw i64 %mul1363.i, %mul1362.i
  %sub1367.i = add nsw i64 %add1364.i, -4029
  %cmp1368.i = icmp sgt i64 %cond1361.i, %sub1367.i
  %cond1425.i = select i1 %cmp1368.i, i64 %cond1361.i, i64 %sub1367.i
  %add1466.i = or i64 %mul1332.i, 14
  %cmp1467.i = icmp slt i64 %cond1464.i, %add1466.i
  %cond1512.i = select i1 %cmp1467.i, i64 %cond1464.i, i64 %add1466.i
  %add1516.i = or i64 %add1364.i, 30
  %cmp1517.i = icmp slt i64 %cond1512.i, %add1516.i
  %cond1612.i = select i1 %cmp1517.i, i64 %cond1512.i, i64 %add1516.i
  %cmp1613.not33.i = icmp sgt i64 %cond1425.i, %cond1612.i
  br i1 %cmp1613.not33.i, label %for.inc1830.i, label %for.body1615.lr.ph.i

for.body1615.lr.ph.i:                             ; preds = %for.body1315.i
  %add1714.i = or i64 %mul1363.i, 31
  br label %for.body1615.i

for.cond1426.loopexit.i:                          ; preds = %for.inc1824.i, %for.body1615.i
  %cmp1613.not.not.i = icmp slt i64 %t4.034.i, %cond1612.i
  %indvar.next182 = add i64 %indvar181, 1
  br i1 %cmp1613.not.not.i, label %for.body1615.i, label %for.inc1830.i, !llvm.loop !9

for.body1615.i:                                   ; preds = %for.cond1426.loopexit.i, %for.body1615.lr.ph.i
  %indvar181 = phi i64 [ %indvar.next182, %for.cond1426.loopexit.i ], [ 0, %for.body1615.lr.ph.i ]
  %t4.034.i = phi i64 [ %add1617.i, %for.cond1426.loopexit.i ], [ %cond1425.i, %for.body1615.lr.ph.i ]
  %35 = sub i64 %31, %indvar181
  %36 = icmp sgt i64 %12, %35
  %smax183 = select i1 %36, i64 %12, i64 %35
  %37 = add i64 %32, %indvar181
  %38 = icmp sgt i64 %smax183, %37
  %smax184 = select i1 %38, i64 %smax183, i64 %37
  %39 = add i64 %smax180, %indvar181
  %40 = sub i64 %smax184, %39
  %41 = add i64 %smax184, %37
  %42 = sub i64 %34, %indvar181
  %43 = sub i64 %42, %smax184
  %add1617.i = add nuw nsw i64 %t4.034.i, 1
  %cmp1618.i = icmp sgt i64 %mul402.i, %add1617.i
  %cond1625.i = select i1 %cmp1618.i, i64 %mul402.i, i64 %add1617.i
  %sub1627.i = sub nsw i64 %mul1363.i, %t4.034.i
  %add1630.i = add nsw i64 %sub1627.i, -3998
  %cmp1631.i = icmp sgt i64 %cond1625.i, %add1630.i
  %cond1651.i = select i1 %cmp1631.i, i64 %cond1625.i, i64 %add1630.i
  %add1657.i = add nsw i64 %sub1627.i, 30
  %cmp1658.i = icmp slt i64 %add1654.i, %add1657.i
  %cond1668.i = select i1 %cmp1658.i, i64 %add1654.i, i64 %add1657.i
  %sub1671.i = add nuw nsw i64 %t4.034.i, 3998
  %cmp1672.i = icmp slt i64 %cond1668.i, %sub1671.i
  %cond1696.i = select i1 %cmp1672.i, i64 %cond1668.i, i64 %sub1671.i
  %cmp1697.not30.i = icmp sgt i64 %cond1651.i, %cond1696.i
  br i1 %cmp1697.not30.i, label %for.cond1426.loopexit.i, label %for.body1699.i

for.body1699.i:                                   ; preds = %for.body1615.i, %for.inc1824.i
  %indvar185 = phi i64 [ %indvar.next186, %for.inc1824.i ], [ 0, %for.body1615.i ]
  %t5.031.i = phi i64 [ %inc1825.i, %for.inc1824.i ], [ %cond1651.i, %for.body1615.i ]
  %44 = add i64 %41, %indvar185
  %add1701.i = add nuw i64 %t5.031.i, %t4.034.i
  %add1702.i = add nsw i64 %add1701.i, 1
  %cmp1703.i = icmp sgt i64 %mul1363.i, %add1702.i
  %cond1711.i = select i1 %cmp1703.i, i64 %mul1363.i, i64 %add1702.i
  %sub1718.i = add nsw i64 %add1701.i, 3998
  %cmp1719.i = icmp slt i64 %add1714.i, %sub1718.i
  %cond1730.i = select i1 %cmp1719.i, i64 %add1714.i, i64 %sub1718.i
  %cmp1731.not28.i = icmp sgt i64 %cond1711.i, %cond1730.i
  br i1 %cmp1731.not28.i, label %for.inc1824.i, label %for.body1733.lr.ph.i

for.body1733.lr.ph.i:                             ; preds = %for.body1699.i
  %45 = add i64 %40, %indvar185
  %46 = mul i64 %45, 32000
  %scevgep = getelementptr i8, i8* %call, i64 %46
  %47 = icmp sgt i64 %33, %44
  %smax187 = select i1 %47, i64 %33, i64 %44
  %48 = sub i64 %43, %indvar185
  %49 = add i64 %smax187, %48
  %50 = shl i64 %49, 3
  %scevgep188 = getelementptr i8, i8* %scevgep, i64 %50
  %scevgep188189 = bitcast i8* %scevgep188 to double*
  %add1735.i = sub nsw i64 %t5.031.i, %t4.034.i
  %sub1736.i = add nsw i64 %add1735.i, -1
  %add1789.i = add nsw i64 %add1735.i, 1
  %load_initial = load double, double* %scevgep188189, align 8
  br label %for.body1733.i

for.body1733.i:                                   ; preds = %for.body1733.i, %for.body1733.lr.ph.i
  %store_forwarded = phi double [ %load_initial, %for.body1733.lr.ph.i ], [ %div1816.i, %for.body1733.i ]
  %t6.029.i = phi i64 [ %cond1711.i, %for.body1733.lr.ph.i ], [ %inc.i, %for.body1733.i ]
  %add1739.i = sub i64 %t6.029.i, %add1701.i
  %sub1740.i = add nsw i64 %add1739.i, -1
  %arrayidx1741.i = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %sub1736.i, i64 %sub1740.i
  %51 = load double, double* %arrayidx1741.i, align 8, !tbaa !2
  %arrayidx1749.i = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %sub1736.i, i64 %add1739.i
  %52 = load double, double* %arrayidx1749.i, align 8, !tbaa !2
  %add1750.i = fadd double %51, %52
  %add1758.i = add nsw i64 %add1739.i, 1
  %arrayidx1759.i = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %sub1736.i, i64 %add1758.i
  %53 = load double, double* %arrayidx1759.i, align 8, !tbaa !2
  %add1760.i = fadd double %add1750.i, %53
  %add1769.i = fadd double %add1760.i, %store_forwarded
  %arrayidx1776.i = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %add1735.i, i64 %add1739.i
  %54 = load double, double* %arrayidx1776.i, align 8, !tbaa !2
  %add1777.i = fadd double %add1769.i, %54
  %arrayidx1785.i = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %add1735.i, i64 %add1758.i
  %55 = load double, double* %arrayidx1785.i, align 8, !tbaa !2
  %add1786.i = fadd double %add1777.i, %55
  %arrayidx1795.i = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %add1789.i, i64 %sub1740.i
  %56 = load double, double* %arrayidx1795.i, align 8, !tbaa !2
  %add1796.i = fadd double %add1786.i, %56
  %arrayidx1804.i = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %add1789.i, i64 %add1739.i
  %57 = load double, double* %arrayidx1804.i, align 8, !tbaa !2
  %add1805.i = fadd double %add1796.i, %57
  %arrayidx1814.i = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %add1789.i, i64 %add1758.i
  %58 = load double, double* %arrayidx1814.i, align 8, !tbaa !2
  %add1815.i = fadd double %add1805.i, %58
  %div1816.i = fdiv double %add1815.i, 9.000000e+00
  store double %div1816.i, double* %arrayidx1776.i, align 8, !tbaa !2
  %inc.i = add nsw i64 %t6.029.i, 1
  %cmp1731.not.not.i = icmp slt i64 %t6.029.i, %cond1730.i
  br i1 %cmp1731.not.not.i, label %for.body1733.i, label %for.inc1824.i, !llvm.loop !10

for.inc1824.i:                                    ; preds = %for.body1733.i, %for.body1699.i
  %inc1825.i = add nuw nsw i64 %t5.031.i, 1
  %cmp1697.not.not.i = icmp slt i64 %t5.031.i, %cond1696.i
  %indvar.next186 = add i64 %indvar185, 1
  br i1 %cmp1697.not.not.i, label %for.body1699.i, label %for.cond1426.loopexit.i, !llvm.loop !11

for.inc1830.i:                                    ; preds = %for.cond1426.loopexit.i, %for.body1315.i
  %inc1831.i = add nuw nsw i64 %t3.0.i175, 1
  %exitcond.not = icmp eq i64 %t3.0.i175, %smax
  %indvar.next177 = add i64 %indvar176, 1
  br i1 %exitcond.not, label %for.inc1833.i, label %for.body1315.i, !llvm.loop !12

for.inc1833.i:                                    ; preds = %for.inc1830.i, %cond.end160.i
  %inc1834.i = add nuw nsw i64 %t2.036.i, 1
  %exitcond.not.i171 = icmp eq i64 %inc1834.i, %indvars.iv.i169
  %indvar.next = add i64 %indvar, 1
  br i1 %exitcond.not.i171, label %for.inc1836.i, label %for.body105.i, !llvm.loop !13

for.inc1836.i:                                    ; preds = %for.inc1833.i
  %inc1837.i = add nuw nsw i64 %t1.038.i, 1
  %indvars.iv.next.i172 = add nuw nsw i64 %indvars.iv.i169, 1
  %exitcond40.not.i = icmp eq i64 %inc1837.i, 32
  br i1 %exitcond40.not.i, label %kernel_seidel_2d.exit, label %for.body105.lr.ph.i, !llvm.loop !14

kernel_seidel_2d.exit:                            ; preds = %for.inc1836.i
  tail call void (...) @polybench_timer_stop() #5
  tail call void (...) @polybench_timer_print() #5
  %cmp = icmp sgt i32 %argc, 42
  br i1 %cmp, label %if.end123, label %if.end138

if.end123:                                        ; preds = %kernel_seidel_2d.exit
  %59 = load i8*, i8** %argv, align 8, !tbaa !15
  %60 = load i8, i8* %59, align 1, !tbaa !17
  %phi.cmp = icmp eq i8 %60, 0
  br i1 %phi.cmp, label %if.then136, label %if.end138

if.then136:                                       ; preds = %if.end123
  %61 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !15
  %62 = tail call i64 @fwrite(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.1, i64 0, i64 0), i64 22, i64 1, %struct._IO_FILE* %61) #6
  %63 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !15
  %call1.i = tail call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %63, i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str.2, i64 0, i64 0), i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.3, i64 0, i64 0)) #6
  br label %for.cond2.preheader.i

for.cond2.preheader.i:                            ; preds = %for.inc10.i, %if.then136
  %indvars.iv4.i163 = phi i64 [ 0, %if.then136 ], [ %indvars.iv.next5.i168, %for.inc10.i ]
  %64 = mul nuw nsw i64 %indvars.iv4.i163, 4000
  br label %for.body4.i

for.body4.i:                                      ; preds = %if.end.i, %for.cond2.preheader.i
  %indvars.iv.i164 = phi i64 [ 0, %for.cond2.preheader.i ], [ %indvars.iv.next.i166, %if.end.i ]
  %65 = add nuw nsw i64 %indvars.iv.i164, %64
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
  %arrayidx8.i165 = getelementptr inbounds [4000 x double], [4000 x double]* %arraydecay, i64 %indvars.iv4.i163, i64 %indvars.iv.i164
  %69 = load double, double* %arrayidx8.i165, align 8, !tbaa !2
  %call9.i = tail call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %68, i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.5, i64 0, i64 0), double %69) #6
  %indvars.iv.next.i166 = add nuw nsw i64 %indvars.iv.i164, 1
  %exitcond.not.i167 = icmp eq i64 %indvars.iv.next.i166, 4000
  br i1 %exitcond.not.i167, label %for.inc10.i, label %for.body4.i, !llvm.loop !18

for.inc10.i:                                      ; preds = %if.end.i
  %indvars.iv.next5.i168 = add nuw nsw i64 %indvars.iv4.i163, 1
  %exitcond7.not.i = icmp eq i64 %indvars.iv.next5.i168, 4000
  br i1 %exitcond7.not.i, label %print_array.exit, label %for.cond2.preheader.i, !llvm.loop !19

print_array.exit:                                 ; preds = %for.inc10.i
  %70 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !15
  %call13.i = tail call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %70, i8* getelementptr inbounds ([17 x i8], [17 x i8]* @.str.6, i64 0, i64 0), i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.3, i64 0, i64 0)) #6
  %71 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !15
  %72 = tail call i64 @fwrite(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.7, i64 0, i64 0), i64 22, i64 1, %struct._IO_FILE* %71) #6
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
