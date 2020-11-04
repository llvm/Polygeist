; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=riscv64 -verify-machineinstrs < %s \
; RUN:   | FileCheck %s -check-prefix=RV64I
; RUN: llc -mtriple=riscv64 -mattr=+experimental-b -verify-machineinstrs < %s \
; RUN:   | FileCheck %s -check-prefix=RV64IB
; RUN: llc -mtriple=riscv64 -mattr=+experimental-zbt -verify-machineinstrs < %s \
; RUN:   | FileCheck %s -check-prefix=RV64IBT

define signext i32 @cmix_i32(i32 signext %a, i32 signext %b, i32 signext %c) nounwind {
; RV64I-LABEL: cmix_i32:
; RV64I:       # %bb.0:
; RV64I-NEXT:    and a0, a1, a0
; RV64I-NEXT:    not a1, a1
; RV64I-NEXT:    and a1, a1, a2
; RV64I-NEXT:    or a0, a1, a0
; RV64I-NEXT:    ret
;
; RV64IB-LABEL: cmix_i32:
; RV64IB:       # %bb.0:
; RV64IB-NEXT:    cmix a0, a1, a0, a2
; RV64IB-NEXT:    ret
;
; RV64IBT-LABEL: cmix_i32:
; RV64IBT:       # %bb.0:
; RV64IBT-NEXT:    cmix a0, a1, a0, a2
; RV64IBT-NEXT:    ret
  %and = and i32 %b, %a
  %neg = xor i32 %b, -1
  %and1 = and i32 %neg, %c
  %or = or i32 %and1, %and
  ret i32 %or
}

define i64 @cmix_i64(i64 %a, i64 %b, i64 %c) nounwind {
; RV64I-LABEL: cmix_i64:
; RV64I:       # %bb.0:
; RV64I-NEXT:    and a0, a1, a0
; RV64I-NEXT:    not a1, a1
; RV64I-NEXT:    and a1, a1, a2
; RV64I-NEXT:    or a0, a1, a0
; RV64I-NEXT:    ret
;
; RV64IB-LABEL: cmix_i64:
; RV64IB:       # %bb.0:
; RV64IB-NEXT:    cmix a0, a1, a0, a2
; RV64IB-NEXT:    ret
;
; RV64IBT-LABEL: cmix_i64:
; RV64IBT:       # %bb.0:
; RV64IBT-NEXT:    cmix a0, a1, a0, a2
; RV64IBT-NEXT:    ret
  %and = and i64 %b, %a
  %neg = xor i64 %b, -1
  %and1 = and i64 %neg, %c
  %or = or i64 %and1, %and
  ret i64 %or
}

define signext i32 @cmov_i32(i32 signext %a, i32 signext %b, i32 signext %c) nounwind {
; RV64I-LABEL: cmov_i32:
; RV64I:       # %bb.0:
; RV64I-NEXT:    beqz a1, .LBB2_2
; RV64I-NEXT:  # %bb.1:
; RV64I-NEXT:    mv a2, a0
; RV64I-NEXT:  .LBB2_2:
; RV64I-NEXT:    mv a0, a2
; RV64I-NEXT:    ret
;
; RV64IB-LABEL: cmov_i32:
; RV64IB:       # %bb.0:
; RV64IB-NEXT:    cmov a0, a1, a0, a2
; RV64IB-NEXT:    ret
;
; RV64IBT-LABEL: cmov_i32:
; RV64IBT:       # %bb.0:
; RV64IBT-NEXT:    cmov a0, a1, a0, a2
; RV64IBT-NEXT:    ret
  %tobool.not = icmp eq i32 %b, 0
  %cond = select i1 %tobool.not, i32 %c, i32 %a
  ret i32 %cond
}

define i64 @cmov_i64(i64 %a, i64 %b, i64 %c) nounwind {
; RV64I-LABEL: cmov_i64:
; RV64I:       # %bb.0:
; RV64I-NEXT:    beqz a1, .LBB3_2
; RV64I-NEXT:  # %bb.1:
; RV64I-NEXT:    mv a2, a0
; RV64I-NEXT:  .LBB3_2:
; RV64I-NEXT:    mv a0, a2
; RV64I-NEXT:    ret
;
; RV64IB-LABEL: cmov_i64:
; RV64IB:       # %bb.0:
; RV64IB-NEXT:    cmov a0, a1, a0, a2
; RV64IB-NEXT:    ret
;
; RV64IBT-LABEL: cmov_i64:
; RV64IBT:       # %bb.0:
; RV64IBT-NEXT:    cmov a0, a1, a0, a2
; RV64IBT-NEXT:    ret
  %tobool.not = icmp eq i64 %b, 0
  %cond = select i1 %tobool.not, i64 %c, i64 %a
  ret i64 %cond
}

declare i32 @llvm.fshl.i32(i32, i32, i32)

define signext i32 @fshl_i32(i32 signext %a, i32 signext %b, i32 signext %c) nounwind {
; RV64I-LABEL: fshl_i32:
; RV64I:       # %bb.0:
; RV64I-NEXT:    slli a0, a0, 32
; RV64I-NEXT:    slli a1, a1, 32
; RV64I-NEXT:    srli a1, a1, 32
; RV64I-NEXT:    or a0, a0, a1
; RV64I-NEXT:    andi a1, a2, 31
; RV64I-NEXT:    sll a0, a0, a1
; RV64I-NEXT:    srai a0, a0, 32
; RV64I-NEXT:    ret
;
; RV64IB-LABEL: fshl_i32:
; RV64IB:       # %bb.0:
; RV64IB-NEXT:    fslw a0, a0, a1, a2
; RV64IB-NEXT:    ret
;
; RV64IBT-LABEL: fshl_i32:
; RV64IBT:       # %bb.0:
; RV64IBT-NEXT:    fslw a0, a0, a1, a2
; RV64IBT-NEXT:    ret
  %1 = tail call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %c)
  ret i32 %1
}

declare i64 @llvm.fshl.i64(i64, i64, i64)

define i64 @fshl_i64(i64 %a, i64 %b, i64 %c) nounwind {
; RV64I-LABEL: fshl_i64:
; RV64I:       # %bb.0:
; RV64I-NEXT:    sll a0, a0, a2
; RV64I-NEXT:    not a2, a2
; RV64I-NEXT:    srli a1, a1, 1
; RV64I-NEXT:    srl a1, a1, a2
; RV64I-NEXT:    or a0, a0, a1
; RV64I-NEXT:    ret
;
; RV64IB-LABEL: fshl_i64:
; RV64IB:       # %bb.0:
; RV64IB-NEXT:    fsl a0, a0, a1, a2
; RV64IB-NEXT:    ret
;
; RV64IBT-LABEL: fshl_i64:
; RV64IBT:       # %bb.0:
; RV64IBT-NEXT:    fsl a0, a0, a1, a2
; RV64IBT-NEXT:    ret
  %1 = tail call i64 @llvm.fshl.i64(i64 %a, i64 %b, i64 %c)
  ret i64 %1
}

declare i32 @llvm.fshr.i32(i32, i32, i32)

define signext i32 @fshr_i32(i32 signext %a, i32 signext %b, i32 signext %c) nounwind {
; RV64I-LABEL: fshr_i32:
; RV64I:       # %bb.0:
; RV64I-NEXT:    slli a0, a0, 32
; RV64I-NEXT:    slli a1, a1, 32
; RV64I-NEXT:    srli a1, a1, 32
; RV64I-NEXT:    or a0, a0, a1
; RV64I-NEXT:    andi a1, a2, 31
; RV64I-NEXT:    srl a0, a0, a1
; RV64I-NEXT:    sext.w a0, a0
; RV64I-NEXT:    ret
;
; RV64IB-LABEL: fshr_i32:
; RV64IB:       # %bb.0:
; RV64IB-NEXT:    fsrw a0, a1, a0, a2
; RV64IB-NEXT:    ret
;
; RV64IBT-LABEL: fshr_i32:
; RV64IBT:       # %bb.0:
; RV64IBT-NEXT:    fsrw a0, a1, a0, a2
; RV64IBT-NEXT:    ret
  %1 = tail call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 %c)
  ret i32 %1
}

declare i64 @llvm.fshr.i64(i64, i64, i64)

define i64 @fshr_i64(i64 %a, i64 %b, i64 %c) nounwind {
; RV64I-LABEL: fshr_i64:
; RV64I:       # %bb.0:
; RV64I-NEXT:    srl a1, a1, a2
; RV64I-NEXT:    not a2, a2
; RV64I-NEXT:    slli a0, a0, 1
; RV64I-NEXT:    sll a0, a0, a2
; RV64I-NEXT:    or a0, a0, a1
; RV64I-NEXT:    ret
;
; RV64IB-LABEL: fshr_i64:
; RV64IB:       # %bb.0:
; RV64IB-NEXT:    fsr a0, a1, a0, a2
; RV64IB-NEXT:    ret
;
; RV64IBT-LABEL: fshr_i64:
; RV64IBT:       # %bb.0:
; RV64IBT-NEXT:    fsr a0, a1, a0, a2
; RV64IBT-NEXT:    ret
  %1 = tail call i64 @llvm.fshr.i64(i64 %a, i64 %b, i64 %c)
  ret i64 %1
}

define signext i32 @fshri_i32(i32 signext %a, i32 signext %b) nounwind {
; RV64I-LABEL: fshri_i32:
; RV64I:       # %bb.0:
; RV64I-NEXT:    srliw a1, a1, 5
; RV64I-NEXT:    slli a0, a0, 27
; RV64I-NEXT:    or a0, a0, a1
; RV64I-NEXT:    sext.w a0, a0
; RV64I-NEXT:    ret
;
; RV64IB-LABEL: fshri_i32:
; RV64IB:       # %bb.0:
; RV64IB-NEXT:    fsriw a0, a1, a0, 5
; RV64IB-NEXT:    ret
;
; RV64IBT-LABEL: fshri_i32:
; RV64IBT:       # %bb.0:
; RV64IBT-NEXT:    fsriw a0, a1, a0, 5
; RV64IBT-NEXT:    ret
  %1 = tail call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 5)
  ret i32 %1
}

define i64 @fshri_i64(i64 %a, i64 %b) nounwind {
; RV64I-LABEL: fshri_i64:
; RV64I:       # %bb.0:
; RV64I-NEXT:    srli a1, a1, 5
; RV64I-NEXT:    slli a0, a0, 59
; RV64I-NEXT:    or a0, a0, a1
; RV64I-NEXT:    ret
;
; RV64IB-LABEL: fshri_i64:
; RV64IB:       # %bb.0:
; RV64IB-NEXT:    fsri a0, a1, a0, 5
; RV64IB-NEXT:    ret
;
; RV64IBT-LABEL: fshri_i64:
; RV64IBT:       # %bb.0:
; RV64IBT-NEXT:    fsri a0, a1, a0, 5
; RV64IBT-NEXT:    ret
  %1 = tail call i64 @llvm.fshr.i64(i64 %a, i64 %b, i64 5)
  ret i64 %1
}

define signext i32 @fshli_i32(i32 signext %a, i32 signext %b) nounwind {
; RV64I-LABEL: fshli_i32:
; RV64I:       # %bb.0:
; RV64I-NEXT:    srliw a1, a1, 27
; RV64I-NEXT:    slli a0, a0, 5
; RV64I-NEXT:    or a0, a0, a1
; RV64I-NEXT:    sext.w a0, a0
; RV64I-NEXT:    ret
;
; RV64IB-LABEL: fshli_i32:
; RV64IB:       # %bb.0:
; RV64IB-NEXT:    slli a1, a1, 32
; RV64IB-NEXT:    addi a2, zero, 5
; RV64IB-NEXT:    fsl a0, a0, a1, a2
; RV64IB-NEXT:    sext.w a0, a0
; RV64IB-NEXT:    ret
;
; RV64IBT-LABEL: fshli_i32:
; RV64IBT:       # %bb.0:
; RV64IBT-NEXT:    slli a1, a1, 32
; RV64IBT-NEXT:    addi a2, zero, 5
; RV64IBT-NEXT:    fsl a0, a0, a1, a2
; RV64IBT-NEXT:    sext.w a0, a0
; RV64IBT-NEXT:    ret
  %1 = tail call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 5)
  ret i32 %1
}

define i64 @fshli_i64(i64 %a, i64 %b) nounwind {
; RV64I-LABEL: fshli_i64:
; RV64I:       # %bb.0:
; RV64I-NEXT:    srli a1, a1, 59
; RV64I-NEXT:    slli a0, a0, 5
; RV64I-NEXT:    or a0, a0, a1
; RV64I-NEXT:    ret
;
; RV64IB-LABEL: fshli_i64:
; RV64IB:       # %bb.0:
; RV64IB-NEXT:    addi a2, zero, 5
; RV64IB-NEXT:    fsl a0, a0, a1, a2
; RV64IB-NEXT:    ret
;
; RV64IBT-LABEL: fshli_i64:
; RV64IBT:       # %bb.0:
; RV64IBT-NEXT:    addi a2, zero, 5
; RV64IBT-NEXT:    fsl a0, a0, a1, a2
; RV64IBT-NEXT:    ret
  %1 = tail call i64 @llvm.fshl.i64(i64 %a, i64 %b, i64 5)
  ret i64 %1
}
