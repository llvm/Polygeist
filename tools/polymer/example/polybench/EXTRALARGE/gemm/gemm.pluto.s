	.text
	.file	"gemm.pluto.c"
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3                               # -- Begin function main
.LCPI0_0:
	.quad	0x409f400000000000              # double 2000
.LCPI0_1:
	.quad	0x40a4500000000000              # double 2600
.LCPI0_2:
	.quad	0x40a1f80000000000              # double 2300
.LCPI0_3:
	.quad	0x3ff3333333333333              # double 1.2
.LCPI0_4:
	.quad	0x3ff8000000000000              # double 1.5
	.text
	.globl	main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$88, %rsp
	.cfi_def_cfa_offset 144
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rsi, 80(%rsp)                  # 8-byte Spill
	movl	%edi, 40(%rsp)                  # 4-byte Spill
	movl	$36800000, %edi                 # imm = 0x2318600
	callq	malloc
	movq	%rax, %rbp
	movl	$41600000, %edi                 # imm = 0x27AC400
	callq	malloc
	movq	%rax, %rbx
	movl	$47840000, %edi                 # imm = 0x2D9FB00
	callq	malloc
	movq	%rax, 24(%rsp)                  # 8-byte Spill
	xorl	%eax, %eax
	movsd	.LCPI0_0(%rip), %xmm0           # xmm0 = mem[0],zero
	movq	%rbp, 8(%rsp)                   # 8-byte Spill
	movq	%rbp, %rcx
	.p2align	4, 0x90
.LBB0_1:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_2 Depth 2
	movl	$1, %ebp
	xorl	%esi, %esi
	.p2align	4, 0x90
.LBB0_2:                                #   Parent Loop BB0_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movl	%ebp, %edi
	imulq	$274877907, %rdi, %rdi          # imm = 0x10624DD3
	shrq	$39, %rdi
	imull	$2000, %edi, %edi               # imm = 0x7D0
	movl	%ebp, %edx
	subl	%edi, %edx
	xorps	%xmm1, %xmm1
	cvtsi2sd	%edx, %xmm1
	divsd	%xmm0, %xmm1
	movsd	%xmm1, (%rcx,%rsi,8)
	addq	$1, %rsi
	addl	%eax, %ebp
	cmpq	$2300, %rsi                     # imm = 0x8FC
	jne	.LBB0_2
# %bb.3:                                #   in Loop: Header=BB0_1 Depth=1
	addq	$1, %rax
	addq	$18400, %rcx                    # imm = 0x47E0
	cmpq	$2000, %rax                     # imm = 0x7D0
	jne	.LBB0_1
# %bb.4:
	xorl	%eax, %eax
	movl	$3383112701, %r8d               # imm = 0xC9A633FD
	movsd	.LCPI0_1(%rip), %xmm0           # xmm0 = mem[0],zero
	movq	%rbx, %rdx
	.p2align	4, 0x90
.LBB0_5:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_6 Depth 2
	movl	%eax, %ecx
	xorl	%edi, %edi
	.p2align	4, 0x90
.LBB0_6:                                #   Parent Loop BB0_5 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movl	%ecx, %ebp
	imulq	%r8, %rbp
	shrq	$43, %rbp
	imull	$2600, %ebp, %ebp               # imm = 0xA28
	movl	%ecx, %esi
	subl	%ebp, %esi
	xorps	%xmm1, %xmm1
	cvtsi2sd	%esi, %xmm1
	divsd	%xmm0, %xmm1
	movsd	%xmm1, (%rdx,%rdi,8)
	addq	$1, %rdi
	addl	%eax, %ecx
	cmpq	$2600, %rdi                     # imm = 0xA28
	jne	.LBB0_6
# %bb.7:                                #   in Loop: Header=BB0_5 Depth=1
	addq	$1, %rax
	addq	$20800, %rdx                    # imm = 0x5140
	cmpq	$2000, %rax                     # imm = 0x7D0
	jne	.LBB0_5
# %bb.8:                                # %.preheader20.preheader
	xorl	%r8d, %r8d
	movl	$3824388271, %r9d               # imm = 0xE3F388AF
	movsd	.LCPI0_2(%rip), %xmm0           # xmm0 = mem[0],zero
	movq	24(%rsp), %rdx                  # 8-byte Reload
	xorl	%esi, %esi
	.p2align	4, 0x90
.LBB0_9:                                # %.preheader20
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_10 Depth 2
	movl	%r8d, %ecx
	xorl	%ebp, %ebp
	.p2align	4, 0x90
.LBB0_10:                               #   Parent Loop BB0_9 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movl	%ecx, %eax
	imulq	%r9, %rax
	shrq	$43, %rax
	imull	$2300, %eax, %eax               # imm = 0x8FC
	movl	%ecx, %edi
	subl	%eax, %edi
	xorps	%xmm1, %xmm1
	cvtsi2sd	%edi, %xmm1
	divsd	%xmm0, %xmm1
	movsd	%xmm1, (%rdx,%rbp,8)
	addq	$1, %rbp
	addl	%esi, %ecx
	cmpq	$2300, %rbp                     # imm = 0x8FC
	jne	.LBB0_10
# %bb.11:                               #   in Loop: Header=BB0_9 Depth=1
	addq	$1, %rsi
	addq	$18400, %rdx                    # imm = 0x47E0
	addl	$2, %r8d
	cmpq	$2600, %rsi                     # imm = 0xA28
	jne	.LBB0_9
# %bb.12:
	xorl	%r15d, %r15d
	xorl	%eax, %eax
	callq	polybench_timer_start
	movl	$31, %r10d
	movl	$2299, %r11d                    # imm = 0x8FB
	movsd	.LCPI0_3(%rip), %xmm0           # xmm0 = mem[0],zero
	movq	8(%rsp), %r9                    # 8-byte Reload
	xorl	%r14d, %r14d
	jmp	.LBB0_13
	.p2align	4, 0x90
.LBB0_21:                               # %.loopexit19
                                        #   in Loop: Header=BB0_13 Depth=1
	addl	$1, %r14d
	addq	$32, %r15
	addl	$32, %r10d
	addq	$588800, %r9                    # imm = 0x8FC00
	cmpl	$63, %r14d
	je	.LBB0_22
.LBB0_13:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_15 Depth 2
                                        #       Child Loop BB0_17 Depth 3
                                        #         Child Loop BB0_18 Depth 4
	cmpl	$1999, %r10d                    # imm = 0x7CF
	movl	$1999, %ecx                     # imm = 0x7CF
	cmovbl	%r10d, %ecx
	movl	%r14d, %eax
	shll	$5, %eax
	leal	31(%rax), %edx
	cmpl	$1999, %edx                     # imm = 0x7CF
	movl	$1999, %esi                     # imm = 0x7CF
	cmovael	%esi, %edx
	cmpl	%edx, %eax
	ja	.LBB0_21
# %bb.14:                               # %.preheader18.preheader
                                        #   in Loop: Header=BB0_13 Depth=1
	addl	$1, %ecx
	xorl	%esi, %esi
	movl	$31, %eax
	movl	$1, %r12d
	movq	%r9, %r13
	jmp	.LBB0_15
	.p2align	4, 0x90
.LBB0_20:                               # %.loopexit17
                                        #   in Loop: Header=BB0_15 Depth=2
	addl	$1, %esi
	addl	$32, %eax
	addq	$-32, %r12
	addq	$256, %r13                      # imm = 0x100
	cmpl	$72, %esi
	je	.LBB0_21
.LBB0_15:                               # %.preheader18
                                        #   Parent Loop BB0_13 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_17 Depth 3
                                        #         Child Loop BB0_18 Depth 4
	cmpl	$2299, %eax                     # imm = 0x8FB
	movl	$2299, %ebp                     # imm = 0x8FB
	cmovbl	%eax, %ebp
	movl	%esi, %edx
	shll	$5, %edx
	leal	31(%rdx), %edi
	cmpl	$2299, %edi                     # imm = 0x8FB
	cmovael	%r11d, %edi
	cmpl	%edi, %edx
	ja	.LBB0_20
# %bb.16:                               # %.preheader16.preheader
                                        #   in Loop: Header=BB0_15 Depth=2
	addq	%r12, %rbp
	movq	%r13, %rdi
	movq	%r15, %rdx
	.p2align	4, 0x90
.LBB0_17:                               # %.preheader16
                                        #   Parent Loop BB0_13 Depth=1
                                        #     Parent Loop BB0_15 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_18 Depth 4
	xorl	%r8d, %r8d
	.p2align	4, 0x90
.LBB0_18:                               #   Parent Loop BB0_13 Depth=1
                                        #     Parent Loop BB0_15 Depth=2
                                        #       Parent Loop BB0_17 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	movsd	(%rdi,%r8,8), %xmm1             # xmm1 = mem[0],zero
	mulsd	%xmm0, %xmm1
	movsd	%xmm1, (%rdi,%r8,8)
	addq	$1, %r8
	cmpq	%r8, %rbp
	jne	.LBB0_18
# %bb.19:                               #   in Loop: Header=BB0_17 Depth=3
	addq	$1, %rdx
	addq	$18400, %rdi                    # imm = 0x47E0
	cmpq	%rcx, %rdx
	jne	.LBB0_17
	jmp	.LBB0_20
.LBB0_22:                               # %.preheader15.preheader
	xorl	%eax, %eax
	movq	%rax, 16(%rsp)                  # 8-byte Spill
	movl	$31, %eax
	movsd	.LCPI0_4(%rip), %xmm0           # xmm0 = mem[0],zero
	movq	8(%rsp), %rcx                   # 8-byte Reload
	movq	%rcx, 56(%rsp)                  # 8-byte Spill
	xorl	%ecx, %ecx
	movl	$2599, %esi                     # imm = 0xA27
	jmp	.LBB0_23
	.p2align	4, 0x90
.LBB0_36:                               # %.us-lcssa.us
                                        #   in Loop: Header=BB0_23 Depth=1
	movl	44(%rsp), %ecx                  # 4-byte Reload
	addl	$1, %ecx
	addq	$32, 16(%rsp)                   # 8-byte Folded Spill
	movl	48(%rsp), %eax                  # 4-byte Reload
	addl	$32, %eax
	addq	$588800, 56(%rsp)               # 8-byte Folded Spill
                                        # imm = 0x8FC00
	cmpl	$63, %ecx
	je	.LBB0_37
.LBB0_23:                               # %.preheader15
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_25 Depth 2
                                        #       Child Loop BB0_27 Depth 3
                                        #         Child Loop BB0_29 Depth 4
                                        #           Child Loop BB0_30 Depth 5
                                        #             Child Loop BB0_31 Depth 6
	cmpl	$1999, %eax                     # imm = 0x7CF
	movl	$1999, %ebp                     # imm = 0x7CF
	movl	%eax, 48(%rsp)                  # 4-byte Spill
	cmovbl	%eax, %ebp
	movl	%ecx, 44(%rsp)                  # 4-byte Spill
	movl	%ecx, %eax
	shll	$5, %eax
	leal	31(%rax), %ecx
	cmpl	$1999, %ecx                     # imm = 0x7CF
	movl	$1999, %edx                     # imm = 0x7CF
	cmovael	%edx, %ecx
	cmpl	%ecx, %eax
	ja	.LBB0_36
# %bb.24:                               # %.split.preheader
                                        #   in Loop: Header=BB0_23 Depth=1
	addl	$1, %ebp
	xorl	%eax, %eax
	movl	$31, %ecx
	movl	$1, %edx
	movq	%rdx, 72(%rsp)                  # 8-byte Spill
	movq	24(%rsp), %rdx                  # 8-byte Reload
	movq	%rdx, 64(%rsp)                  # 8-byte Spill
	movq	56(%rsp), %rdi                  # 8-byte Reload
	jmp	.LBB0_25
	.p2align	4, 0x90
.LBB0_35:                               # %.loopexit14
                                        #   in Loop: Header=BB0_25 Depth=2
	movl	32(%rsp), %eax                  # 4-byte Reload
	addl	$1, %eax
	movl	52(%rsp), %ecx                  # 4-byte Reload
	addl	$32, %ecx
	addq	$-32, 72(%rsp)                  # 8-byte Folded Spill
	addq	$256, %rdi                      # imm = 0x100
	addq	$256, 64(%rsp)                  # 8-byte Folded Spill
                                        # imm = 0x100
	cmpl	$72, %eax
	je	.LBB0_36
.LBB0_25:                               # %.split
                                        #   Parent Loop BB0_23 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_27 Depth 3
                                        #         Child Loop BB0_29 Depth 4
                                        #           Child Loop BB0_30 Depth 5
                                        #             Child Loop BB0_31 Depth 6
	cmpl	$2299, %ecx                     # imm = 0x8FB
	movl	$2299, %r13d                    # imm = 0x8FB
	movl	%ecx, 52(%rsp)                  # 4-byte Spill
	cmovbl	%ecx, %r13d
	movl	%eax, 32(%rsp)                  # 4-byte Spill
                                        # kill: def $eax killed $eax def $rax
	shll	$5, %eax
	leal	31(%rax), %ecx
	cmpl	$2299, %ecx                     # imm = 0x8FB
	movl	$2299, %edx                     # imm = 0x8FB
	cmovael	%edx, %ecx
	cmpl	%ecx, %eax
	ja	.LBB0_35
# %bb.26:                               # %.preheader13.split.preheader
                                        #   in Loop: Header=BB0_25 Depth=2
	addq	72(%rsp), %r13                  # 8-byte Folded Reload
	xorl	%r12d, %r12d
	movl	$31, %r14d
	movq	64(%rsp), %r10                  # 8-byte Reload
	xorl	%r11d, %r11d
	jmp	.LBB0_27
	.p2align	4, 0x90
.LBB0_34:                               # %.loopexit
                                        #   in Loop: Header=BB0_27 Depth=3
	addl	$1, %r11d
	addq	$32, %r12
	addl	$32, %r14d
	addq	$588800, %r10                   # imm = 0x8FC00
	cmpl	$82, %r11d
	je	.LBB0_35
.LBB0_27:                               # %.preheader13.split
                                        #   Parent Loop BB0_23 Depth=1
                                        #     Parent Loop BB0_25 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_29 Depth 4
                                        #           Child Loop BB0_30 Depth 5
                                        #             Child Loop BB0_31 Depth 6
	cmpl	$2599, %r14d                    # imm = 0xA27
	movl	$2599, %edx                     # imm = 0xA27
	cmovbl	%r14d, %edx
	movl	%r11d, %eax
	shll	$5, %eax
	leal	31(%rax), %ecx
	cmpl	$2599, %ecx                     # imm = 0xA27
	cmovael	%esi, %ecx
	cmpl	%ecx, %eax
	ja	.LBB0_34
# %bb.28:                               # %.preheader.preheader
                                        #   in Loop: Header=BB0_27 Depth=3
	addl	$1, %edx
	movq	%rdi, %r9
	movq	16(%rsp), %rax                  # 8-byte Reload
	.p2align	4, 0x90
.LBB0_29:                               # %.preheader
                                        #   Parent Loop BB0_23 Depth=1
                                        #     Parent Loop BB0_25 Depth=2
                                        #       Parent Loop BB0_27 Depth=3
                                        # =>      This Loop Header: Depth=4
                                        #           Child Loop BB0_30 Depth 5
                                        #             Child Loop BB0_31 Depth 6
	movq	%r10, %r15
	movq	%r12, %r8
	.p2align	4, 0x90
.LBB0_30:                               #   Parent Loop BB0_23 Depth=1
                                        #     Parent Loop BB0_25 Depth=2
                                        #       Parent Loop BB0_27 Depth=3
                                        #         Parent Loop BB0_29 Depth=4
                                        # =>        This Loop Header: Depth=5
                                        #             Child Loop BB0_31 Depth 6
	imulq	$20800, %rax, %rcx              # imm = 0x5140
	addq	%rbx, %rcx
	movsd	(%rcx,%r8,8), %xmm1             # xmm1 = mem[0],zero
	mulsd	%xmm0, %xmm1
	xorl	%ecx, %ecx
	.p2align	4, 0x90
.LBB0_31:                               #   Parent Loop BB0_23 Depth=1
                                        #     Parent Loop BB0_25 Depth=2
                                        #       Parent Loop BB0_27 Depth=3
                                        #         Parent Loop BB0_29 Depth=4
                                        #           Parent Loop BB0_30 Depth=5
                                        # =>          This Inner Loop Header: Depth=6
	movsd	(%r15,%rcx,8), %xmm2            # xmm2 = mem[0],zero
	mulsd	%xmm1, %xmm2
	addsd	(%r9,%rcx,8), %xmm2
	movsd	%xmm2, (%r9,%rcx,8)
	addq	$1, %rcx
	cmpq	%rcx, %r13
	jne	.LBB0_31
# %bb.32:                               #   in Loop: Header=BB0_30 Depth=5
	addq	$1, %r8
	addq	$18400, %r15                    # imm = 0x47E0
	cmpq	%rdx, %r8
	jne	.LBB0_30
# %bb.33:                               #   in Loop: Header=BB0_29 Depth=4
	addq	$1, %rax
	addq	$18400, %r9                     # imm = 0x47E0
	cmpq	%rbp, %rax
	jne	.LBB0_29
	jmp	.LBB0_34
.LBB0_37:
	xorl	%eax, %eax
	callq	polybench_timer_stop
	xorl	%eax, %eax
	callq	polybench_timer_print
	cmpl	$43, 40(%rsp)                   # 4-byte Folded Reload
	jl	.LBB0_46
# %bb.38:
	movq	80(%rsp), %rax                  # 8-byte Reload
	movq	(%rax), %rax
	cmpb	$0, (%rax)
	je	.LBB0_39
.LBB0_46:
	movq	8(%rsp), %rdi                   # 8-byte Reload
	callq	free
	movq	%rbx, %rdi
	callq	free
	movq	24(%rsp), %rdi                  # 8-byte Reload
	callq	free
	xorl	%eax, %eax
	addq	$88, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB0_39:
	.cfi_def_cfa_offset 144
	movq	stderr(%rip), %rcx
	movl	$.L.str.1, %edi
	movl	$22, %esi
	movl	$1, %edx
	callq	fwrite@PLT
	movq	stderr(%rip), %rdi
	xorl	%ebp, %ebp
	movl	$.L.str.2, %esi
	movl	$.L.str.3, %edx
	xorl	%eax, %eax
	callq	fprintf
	xorl	%r12d, %r12d
	movq	8(%rsp), %r13                   # 8-byte Reload
	xorl	%eax, %eax
.LBB0_40:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_41 Depth 2
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	movq	%rbp, 16(%rsp)                  # 8-byte Spill
	movl	%ebp, %r14d
	xorl	%r15d, %r15d
	movl	$3435973837, %ebp               # imm = 0xCCCCCCCD
.LBB0_41:                               #   Parent Loop BB0_40 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movl	%r14d, %eax
	imulq	%rbp, %rax
	shrq	$36, %rax
	leal	(%rax,%rax,4), %eax
	leal	(%r12,%rax,4), %eax
	cmpl	%r15d, %eax
	jne	.LBB0_43
# %bb.42:                               #   in Loop: Header=BB0_41 Depth=2
	movq	stderr(%rip), %rsi
	movl	$10, %edi
	callq	fputc@PLT
.LBB0_43:                               #   in Loop: Header=BB0_41 Depth=2
	movq	stderr(%rip), %rdi
	movsd	(%r13,%r15,8), %xmm0            # xmm0 = mem[0],zero
	movl	$.L.str.5, %esi
	movb	$1, %al
	callq	fprintf
	addq	$1, %r15
	addl	$1, %r14d
	cmpq	$2300, %r15                     # imm = 0x8FC
	jne	.LBB0_41
# %bb.44:                               #   in Loop: Header=BB0_40 Depth=1
	movq	32(%rsp), %rax                  # 8-byte Reload
	addq	$1, %rax
	addq	$18400, %r13                    # imm = 0x47E0
	addl	$-2000, %r12d                   # imm = 0xF830
	movq	16(%rsp), %rbp                  # 8-byte Reload
	addl	$2000, %ebp                     # imm = 0x7D0
	cmpq	$2000, %rax                     # imm = 0x7D0
	jne	.LBB0_40
# %bb.45:
	movq	stderr(%rip), %rdi
	movl	$.L.str.6, %esi
	movl	$.L.str.3, %edx
	xorl	%eax, %eax
	callq	fprintf
	movq	stderr(%rip), %rcx
	movl	$.L.str.7, %edi
	movl	$22, %esi
	movl	$1, %edx
	callq	fwrite@PLT
	jmp	.LBB0_46
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
                                        # -- End function
	.type	.L.str.1,@object                # @.str.1
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str.1:
	.asciz	"==BEGIN DUMP_ARRAYS==\n"
	.size	.L.str.1, 23

	.type	.L.str.2,@object                # @.str.2
.L.str.2:
	.asciz	"begin dump: %s"
	.size	.L.str.2, 15

	.type	.L.str.3,@object                # @.str.3
.L.str.3:
	.asciz	"C"
	.size	.L.str.3, 2

	.type	.L.str.5,@object                # @.str.5
.L.str.5:
	.asciz	"%0.2lf "
	.size	.L.str.5, 8

	.type	.L.str.6,@object                # @.str.6
.L.str.6:
	.asciz	"\nend   dump: %s\n"
	.size	.L.str.6, 17

	.type	.L.str.7,@object                # @.str.7
.L.str.7:
	.asciz	"==END   DUMP_ARRAYS==\n"
	.size	.L.str.7, 23

	.ident	"clang version 12.0.0 (git@github.com:wsmoses/MLIR-GPU 525f08761a2245405ec538260c1b1db18dcc2bcf)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
