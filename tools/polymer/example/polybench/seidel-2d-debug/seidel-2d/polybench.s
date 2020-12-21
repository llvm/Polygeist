	.text
	.file	"polybench.c"
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3                               # -- Begin function polybench_flush_cache
.LCPI0_0:
	.quad	0x4024000000000000              # double 10
	.text
	.globl	polybench_flush_cache
	.p2align	4, 0x90
	.type	polybench_flush_cache,@function
polybench_flush_cache:                  # @polybench_flush_cache
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset %rbx, -16
	movl	$8, %ebx
	movl	$4194560, %edi                  # imm = 0x400100
	movl	$8, %esi
	callq	calloc
	xorpd	%xmm0, %xmm0
	.p2align	4, 0x90
.LBB0_1:                                # %for.body.for.body_crit_edge
                                        # =>This Inner Loop Header: Depth=1
	addsd	-56(%rax,%rbx,8), %xmm0
	addsd	-48(%rax,%rbx,8), %xmm0
	addsd	-40(%rax,%rbx,8), %xmm0
	addsd	-32(%rax,%rbx,8), %xmm0
	addsd	-24(%rax,%rbx,8), %xmm0
	addsd	-16(%rax,%rbx,8), %xmm0
	addsd	-8(%rax,%rbx,8), %xmm0
	cmpq	$4194560, %rbx                  # imm = 0x400100
	je	.LBB0_2
# %bb.4:                                # %for.body.for.body_crit_edge.7
                                        #   in Loop: Header=BB0_1 Depth=1
	addsd	(%rax,%rbx,8), %xmm0
	addq	$8, %rbx
	jmp	.LBB0_1
.LBB0_2:                                # %for.end
	movsd	.LCPI0_0(%rip), %xmm1           # xmm1 = mem[0],zero
	ucomisd	%xmm0, %xmm1
	jb	.LBB0_5
# %bb.3:                                # %cond.end
	movq	%rax, %rdi
	popq	%rbx
	.cfi_def_cfa_offset 8
	jmp	free                            # TAILCALL
.LBB0_5:                                # %cond.false
	.cfi_def_cfa_offset 16
	movl	$.L.str, %edi
	movl	$.L.str.1, %esi
	movl	$.L__PRETTY_FUNCTION__.polybench_flush_cache, %ecx
	movl	$116, %edx
	callq	__assert_fail
.Lfunc_end0:
	.size	polybench_flush_cache, .Lfunc_end0-polybench_flush_cache
	.cfi_endproc
                                        # -- End function
	.globl	polybench_prepare_instruments   # -- Begin function polybench_prepare_instruments
	.p2align	4, 0x90
	.type	polybench_prepare_instruments,@function
polybench_prepare_instruments:          # @polybench_prepare_instruments
	.cfi_startproc
# %bb.0:                                # %entry
	retq
.Lfunc_end1:
	.size	polybench_prepare_instruments, .Lfunc_end1-polybench_prepare_instruments
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3                               # -- Begin function polybench_timer_start
.LCPI2_0:
	.quad	0x3eb0c6f7a0b5ed8d              # double 9.9999999999999995E-7
	.text
	.globl	polybench_timer_start
	.p2align	4, 0x90
	.type	polybench_timer_start,@function
polybench_timer_start:                  # @polybench_timer_start
	.cfi_startproc
# %bb.0:                                # %entry
	subq	$24, %rsp
	.cfi_def_cfa_offset 32
	leaq	8(%rsp), %rdi
	xorl	%esi, %esi
	callq	gettimeofday
	testl	%eax, %eax
	je	.LBB2_2
# %bb.1:                                # %if.then.i
	movl	$.L.str.3, %edi
	movl	%eax, %esi
	xorl	%eax, %eax
	callq	printf
.LBB2_2:                                # %rtclock.exit
	cvtsi2sdq	8(%rsp), %xmm0
	cvtsi2sdq	16(%rsp), %xmm1
	mulsd	.LCPI2_0(%rip), %xmm1
	addsd	%xmm0, %xmm1
	movsd	%xmm1, polybench_t_start(%rip)
	addq	$24, %rsp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end2:
	.size	polybench_timer_start, .Lfunc_end2-polybench_timer_start
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3                               # -- Begin function polybench_timer_stop
.LCPI3_0:
	.quad	0x3eb0c6f7a0b5ed8d              # double 9.9999999999999995E-7
	.text
	.globl	polybench_timer_stop
	.p2align	4, 0x90
	.type	polybench_timer_stop,@function
polybench_timer_stop:                   # @polybench_timer_stop
	.cfi_startproc
# %bb.0:                                # %entry
	subq	$24, %rsp
	.cfi_def_cfa_offset 32
	leaq	8(%rsp), %rdi
	xorl	%esi, %esi
	callq	gettimeofday
	testl	%eax, %eax
	je	.LBB3_2
# %bb.1:                                # %if.then.i
	movl	$.L.str.3, %edi
	movl	%eax, %esi
	xorl	%eax, %eax
	callq	printf
.LBB3_2:                                # %rtclock.exit
	cvtsi2sdq	8(%rsp), %xmm0
	cvtsi2sdq	16(%rsp), %xmm1
	mulsd	.LCPI3_0(%rip), %xmm1
	addsd	%xmm0, %xmm1
	movsd	%xmm1, polybench_t_end(%rip)
	addq	$24, %rsp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end3:
	.size	polybench_timer_stop, .Lfunc_end3-polybench_timer_stop
	.cfi_endproc
                                        # -- End function
	.globl	polybench_timer_print           # -- Begin function polybench_timer_print
	.p2align	4, 0x90
	.type	polybench_timer_print,@function
polybench_timer_print:                  # @polybench_timer_print
	.cfi_startproc
# %bb.0:                                # %entry
	movsd	polybench_t_end(%rip), %xmm0    # xmm0 = mem[0],zero
	subsd	polybench_t_start(%rip), %xmm0
	movl	$.L.str.2, %edi
	movb	$1, %al
	jmp	printf                          # TAILCALL
.Lfunc_end4:
	.size	polybench_timer_print, .Lfunc_end4-polybench_timer_print
	.cfi_endproc
                                        # -- End function
	.type	polybench_papi_counters_threadid,@object # @polybench_papi_counters_threadid
	.bss
	.globl	polybench_papi_counters_threadid
	.p2align	2
polybench_papi_counters_threadid:
	.long	0                               # 0x0
	.size	polybench_papi_counters_threadid, 4

	.type	polybench_program_total_flops,@object # @polybench_program_total_flops
	.globl	polybench_program_total_flops
	.p2align	3
polybench_program_total_flops:
	.quad	0x0000000000000000              # double 0
	.size	polybench_program_total_flops, 8

	.type	.L.str,@object                  # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"tmp <= 10.0"
	.size	.L.str, 12

	.type	.L.str.1,@object                # @.str.1
.L.str.1:
	.asciz	"/mnt/ccnas2/bdp/rz3515/projects/polymer/example/polybench/utilities/polybench.c"
	.size	.L.str.1, 80

	.type	.L__PRETTY_FUNCTION__.polybench_flush_cache,@object # @__PRETTY_FUNCTION__.polybench_flush_cache
.L__PRETTY_FUNCTION__.polybench_flush_cache:
	.asciz	"void polybench_flush_cache()"
	.size	.L__PRETTY_FUNCTION__.polybench_flush_cache, 29

	.type	polybench_t_start,@object       # @polybench_t_start
	.bss
	.globl	polybench_t_start
	.p2align	3
polybench_t_start:
	.quad	0x0000000000000000              # double 0
	.size	polybench_t_start, 8

	.type	polybench_t_end,@object         # @polybench_t_end
	.globl	polybench_t_end
	.p2align	3
polybench_t_end:
	.quad	0x0000000000000000              # double 0
	.size	polybench_t_end, 8

	.type	.L.str.2,@object                # @.str.2
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str.2:
	.asciz	"%0.6f\n"
	.size	.L.str.2, 7

	.type	polybench_c_start,@object       # @polybench_c_start
	.bss
	.globl	polybench_c_start
	.p2align	3
polybench_c_start:
	.quad	0                               # 0x0
	.size	polybench_c_start, 8

	.type	polybench_c_end,@object         # @polybench_c_end
	.globl	polybench_c_end
	.p2align	3
polybench_c_end:
	.quad	0                               # 0x0
	.size	polybench_c_end, 8

	.type	.L.str.3,@object                # @.str.3
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str.3:
	.asciz	"Error return from gettimeofday: %d"
	.size	.L.str.3, 35

	.ident	"clang version 12.0.0 (git@github.com:wsmoses/MLIR-GPU 1112d5451cea635029a160c950f14a85f31b2258)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
