	.text
	.file	"2mm.pluto.c"
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3                               # -- Begin function main
.LCPI0_0:
	.quad	0x4099000000000000              # double 1600
.LCPI0_1:
	.quad	0x409c200000000000              # double 1800
.LCPI0_2:
	.quad	0x40a2c00000000000              # double 2400
.LCPI0_3:
	.quad	0x40a1300000000000              # double 2200
.LCPI0_4:
	.quad	0x3ff3333333333333              # double 1.2
.LCPI0_5:
	.quad	0x3ff8000000000000              # double 1.5
	.text
	.globl	main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# %bb.0:                                # %entry
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
	subq	$120, %rsp
	.cfi_def_cfa_offset 176
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rsi, 104(%rsp)                 # 8-byte Spill
	movl	%edi, 80(%rsp)                  # 4-byte Spill
	movl	$23040000, %edi                 # imm = 0x15F9000
	callq	malloc
	movq	%rax, %rbx
	movl	$28160000, %edi                 # imm = 0x1ADB000
	callq	malloc
	movq	%rax, %r14
	movl	$31680000, %edi                 # imm = 0x1E36600
	callq	malloc
	movq	%rax, 56(%rsp)                  # 8-byte Spill
	movl	$34560000, %edi                 # imm = 0x20F5800
	callq	malloc
	movq	%rax, 48(%rsp)                  # 8-byte Spill
	movl	$30720000, %edi                 # imm = 0x1D4C000
	callq	malloc
	movq	%rax, %r15
	xorl	%eax, %eax
	movsd	.LCPI0_0(%rip), %xmm0           # xmm0 = mem[0],zero
	movq	%r14, %rcx
	.p2align	4, 0x90
.LBB0_1:                                # %for.cond1.preheader.i
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_2 Depth 2
	movl	$1, %ebp
	xorl	%esi, %esi
	.p2align	4, 0x90
.LBB0_2:                                # %for.body3.i
                                        #   Parent Loop BB0_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movl	%ebp, %edi
	imulq	$1374389535, %rdi, %rdi         # imm = 0x51EB851F
	shrq	$41, %rdi
	imull	$1600, %edi, %edi               # imm = 0x640
	movl	%ebp, %edx
	subl	%edi, %edx
	xorps	%xmm1, %xmm1
	cvtsi2sd	%edx, %xmm1
	divsd	%xmm0, %xmm1
	movsd	%xmm1, (%rcx,%rsi,8)
	addq	$1, %rsi
	addl	%eax, %ebp
	cmpq	$2200, %rsi                     # imm = 0x898
	jne	.LBB0_2
# %bb.3:                                # %for.inc7.i
                                        #   in Loop: Header=BB0_1 Depth=1
	addq	$1, %rax
	addq	$17600, %rcx                    # imm = 0x44C0
	cmpq	$1600, %rax                     # imm = 0x640
	jne	.LBB0_1
# %bb.4:                                # %for.cond14.preheader.i.preheader
	xorl	%eax, %eax
	movl	$2443359173, %r8d               # imm = 0x91A2B3C5
	movsd	.LCPI0_1(%rip), %xmm0           # xmm0 = mem[0],zero
	movq	56(%rsp), %rdx                  # 8-byte Reload
	.p2align	4, 0x90
.LBB0_5:                                # %for.cond14.preheader.i
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_6 Depth 2
	movl	%eax, %ecx
	xorl	%edi, %edi
	.p2align	4, 0x90
.LBB0_6:                                # %for.body17.i
                                        #   Parent Loop BB0_5 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movl	%ecx, %ebp
	imulq	%r8, %rbp
	shrq	$42, %rbp
	imull	$1800, %ebp, %ebp               # imm = 0x708
	movl	%ecx, %esi
	subl	%ebp, %esi
	xorps	%xmm1, %xmm1
	cvtsi2sd	%esi, %xmm1
	divsd	%xmm0, %xmm1
	movsd	%xmm1, (%rdx,%rdi,8)
	addq	$1, %rdi
	addl	%eax, %ecx
	cmpq	$1800, %rdi                     # imm = 0x708
	jne	.LBB0_6
# %bb.7:                                # %for.inc31.i
                                        #   in Loop: Header=BB0_5 Depth=1
	addq	$1, %rax
	addq	$14400, %rdx                    # imm = 0x3840
	cmpq	$2200, %rax                     # imm = 0x898
	jne	.LBB0_5
# %bb.8:                                # %for.cond38.preheader.i.preheader
	movl	$1, %r8d
	xorl	%ecx, %ecx
	movsd	.LCPI0_2(%rip), %xmm0           # xmm0 = mem[0],zero
	movq	48(%rsp), %rdx                  # 8-byte Reload
	.p2align	4, 0x90
.LBB0_9:                                # %for.cond38.preheader.i
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_10 Depth 2
	movl	%r8d, %eax
	xorl	%edi, %edi
	.p2align	4, 0x90
.LBB0_10:                               # %for.body41.i
                                        #   Parent Loop BB0_9 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movl	%eax, %ebp
	imulq	$458129845, %rbp, %rbp          # imm = 0x1B4E81B5
	shrq	$40, %rbp
	imull	$2400, %ebp, %ebp               # imm = 0x960
	movl	%eax, %esi
	subl	%ebp, %esi
	xorps	%xmm1, %xmm1
	cvtsi2sd	%esi, %xmm1
	divsd	%xmm0, %xmm1
	movsd	%xmm1, (%rdx,%rdi,8)
	addq	$1, %rdi
	addl	%ecx, %eax
	cmpq	$2400, %rdi                     # imm = 0x960
	jne	.LBB0_10
# %bb.11:                               # %for.inc56.i
                                        #   in Loop: Header=BB0_9 Depth=1
	addq	$1, %rcx
	addq	$19200, %rdx                    # imm = 0x4B00
	addl	$3, %r8d
	cmpq	$1800, %rcx                     # imm = 0x708
	jne	.LBB0_9
# %bb.12:                               # %for.cond63.preheader.i.preheader
	xorl	%r8d, %r8d
	movsd	.LCPI0_3(%rip), %xmm0           # xmm0 = mem[0],zero
	movq	%r15, %rcx
	xorl	%edx, %edx
	.p2align	4, 0x90
.LBB0_13:                               # %for.cond63.preheader.i
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_14 Depth 2
	movl	%r8d, %eax
	xorl	%edi, %edi
	.p2align	4, 0x90
.LBB0_14:                               # %for.body66.i
                                        #   Parent Loop BB0_13 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movl	%eax, %ebp
	shrl	$3, %ebp
	imulq	$499778013, %rbp, %rbp          # imm = 0x1DCA01DD
	shrq	$37, %rbp
	imull	$2200, %ebp, %ebp               # imm = 0x898
	movl	%eax, %esi
	subl	%ebp, %esi
	xorps	%xmm1, %xmm1
	cvtsi2sd	%esi, %xmm1
	divsd	%xmm0, %xmm1
	movsd	%xmm1, (%rcx,%rdi,8)
	addq	$1, %rdi
	addl	%edx, %eax
	cmpq	$2400, %rdi                     # imm = 0x960
	jne	.LBB0_14
# %bb.15:                               # %for.inc80.i
                                        #   in Loop: Header=BB0_13 Depth=1
	addq	$1, %rdx
	addq	$19200, %rcx                    # imm = 0x4B00
	addl	$2, %r8d
	cmpq	$1600, %rdx                     # imm = 0x640
	jne	.LBB0_13
# %bb.16:                               # %init_array.exit
	xorl	%r12d, %r12d
	xorl	%eax, %eax
	callq	polybench_timer_start
	leaq	14336(%r15), %r11
	movq	%r15, %rax
	addq	$14400, %rax                    # imm = 0x3840
	movl	$32, %r13d
	xorpd	%xmm1, %xmm1
	movsd	.LCPI0_4(%rip), %xmm2           # xmm2 = mem[0],zero
	movq	%rbx, 16(%rsp)                  # 8-byte Spill
	movq	%r15, 40(%rsp)                  # 8-byte Spill
	movq	%r15, %r9
	movq	%rax, %r15
	xorl	%eax, %eax
	movq	%rax, (%rsp)                    # 8-byte Spill
	.p2align	4, 0x90
.LBB0_17:                               # %for.cond12.preheader.i
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_18 Depth 2
                                        #       Child Loop BB0_20 Depth 3
                                        #         Child Loop BB0_21 Depth 4
                                        #         Child Loop BB0_23 Depth 4
                                        #       Child Loop BB0_27 Depth 3
                                        #         Child Loop BB0_28 Depth 4
                                        #       Child Loop BB0_32 Depth 3
                                        #         Child Loop BB0_33 Depth 4
                                        #     Child Loop BB0_37 Depth 2
                                        #       Child Loop BB0_39 Depth 3
                                        #     Child Loop BB0_42 Depth 2
                                        #       Child Loop BB0_43 Depth 3
                                        #         Child Loop BB0_44 Depth 4
	imulq	$460800, %r12, %r8              # imm = 0x70800
	leaq	14336(%r8), %rcx
	movq	%r9, 24(%rsp)                   # 8-byte Spill
	xorl	%r10d, %r10d
	jmp	.LBB0_18
	.p2align	4, 0x90
.LBB0_35:                               # %for.inc527.i
                                        #   in Loop: Header=BB0_18 Depth=2
	addq	$1, %r10
	addq	$256, %r9                       # imm = 0x100
	cmpl	$132, %r10d
	je	.LBB0_36
.LBB0_18:                               # %cond.end188.i
                                        #   Parent Loop BB0_17 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_20 Depth 3
                                        #         Child Loop BB0_21 Depth 4
                                        #         Child Loop BB0_23 Depth 4
                                        #       Child Loop BB0_27 Depth 3
                                        #         Child Loop BB0_28 Depth 4
                                        #       Child Loop BB0_32 Depth 3
                                        #         Child Loop BB0_33 Depth 4
	cmpl	$56, %r10d
	jne	.LBB0_25
# %bb.19:                               # %for.cond224.preheader.i.preheader
                                        #   in Loop: Header=BB0_18 Depth=2
	movq	%r15, %rdi
	movq	%r11, %rbp
	xorl	%eax, %eax
	movq	(%rsp), %rdx                    # 8-byte Reload
	.p2align	4, 0x90
.LBB0_20:                               # %for.cond224.preheader.i
                                        #   Parent Loop BB0_17 Depth=1
                                        #     Parent Loop BB0_18 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_21 Depth 4
                                        #         Child Loop BB0_23 Depth 4
	imulq	$14400, %rax, %rsi              # imm = 0x3840
	addq	%rcx, %rsi
	movupd	%xmm1, 48(%rbx,%rsi)
	movupd	%xmm1, 32(%rbx,%rsi)
	movupd	%xmm1, 16(%rbx,%rsi)
	movupd	%xmm1, (%rbx,%rsi)
	xorl	%esi, %esi
	.p2align	4, 0x90
.LBB0_21:                               # %for.body227.i
                                        #   Parent Loop BB0_17 Depth=1
                                        #     Parent Loop BB0_18 Depth=2
                                        #       Parent Loop BB0_20 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	movsd	(%rbp,%rsi,8), %xmm0            # xmm0 = mem[0],zero
	mulsd	%xmm2, %xmm0
	movsd	%xmm0, (%rbp,%rsi,8)
	addq	$1, %rsi
	cmpq	$8, %rsi
	jne	.LBB0_21
# %bb.22:                               # %for.body253.i.preheader
                                        #   in Loop: Header=BB0_20 Depth=3
	xorl	%esi, %esi
	.p2align	4, 0x90
.LBB0_23:                               # %for.body253.i
                                        #   Parent Loop BB0_17 Depth=1
                                        #     Parent Loop BB0_18 Depth=2
                                        #       Parent Loop BB0_20 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	movsd	(%rdi,%rsi,8), %xmm0            # xmm0 = mem[0],zero
	mulsd	%xmm2, %xmm0
	movsd	%xmm0, (%rdi,%rsi,8)
	addq	$1, %rsi
	cmpq	$24, %rsi
	jne	.LBB0_23
# %bb.24:                               # %for.inc262.i
                                        #   in Loop: Header=BB0_20 Depth=3
	addq	$1, %rdx
	addq	$1, %rax
	addq	$19200, %rbp                    # imm = 0x4B00
	addq	$19200, %rdi                    # imm = 0x4B00
	cmpq	%r13, %rdx
	jne	.LBB0_20
.LBB0_25:                               # %cond.end345.i
                                        #   in Loop: Header=BB0_18 Depth=2
	cmpl	$55, %r10d
	ja	.LBB0_30
# %bb.26:                               # %for.body363.i.preheader
                                        #   in Loop: Header=BB0_18 Depth=2
	movq	%r10, %rax
	shlq	$8, %rax
	addq	%r8, %rax
	movq	%r9, %rdx
	xorl	%edi, %edi
	movq	(%rsp), %rbp                    # 8-byte Reload
	.p2align	4, 0x90
.LBB0_27:                               # %for.body363.i
                                        #   Parent Loop BB0_17 Depth=1
                                        #     Parent Loop BB0_18 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_28 Depth 4
	imulq	$14400, %rdi, %rsi              # imm = 0x3840
	addq	%rax, %rsi
	movupd	%xmm1, 240(%rbx,%rsi)
	movupd	%xmm1, 224(%rbx,%rsi)
	movupd	%xmm1, 208(%rbx,%rsi)
	movupd	%xmm1, 192(%rbx,%rsi)
	movupd	%xmm1, 176(%rbx,%rsi)
	movupd	%xmm1, 160(%rbx,%rsi)
	movupd	%xmm1, 144(%rbx,%rsi)
	movupd	%xmm1, 128(%rbx,%rsi)
	movupd	%xmm1, 112(%rbx,%rsi)
	movupd	%xmm1, 96(%rbx,%rsi)
	movupd	%xmm1, 80(%rbx,%rsi)
	movupd	%xmm1, 64(%rbx,%rsi)
	movupd	%xmm1, 48(%rbx,%rsi)
	movupd	%xmm1, 32(%rbx,%rsi)
	movupd	%xmm1, 16(%rbx,%rsi)
	movupd	%xmm1, (%rbx,%rsi)
	xorl	%esi, %esi
	.p2align	4, 0x90
.LBB0_28:                               # %for.body369.i
                                        #   Parent Loop BB0_17 Depth=1
                                        #     Parent Loop BB0_18 Depth=2
                                        #       Parent Loop BB0_27 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	movsd	(%rdx,%rsi,8), %xmm0            # xmm0 = mem[0],zero
	mulsd	%xmm2, %xmm0
	movsd	%xmm0, (%rdx,%rsi,8)
	addq	$1, %rsi
	cmpq	$32, %rsi
	jne	.LBB0_28
# %bb.29:                               # %for.inc382.i
                                        #   in Loop: Header=BB0_27 Depth=3
	addq	$1, %rbp
	addq	$1, %rdi
	addq	$19200, %rdx                    # imm = 0x4B00
	cmpq	%r13, %rbp
	jne	.LBB0_27
.LBB0_30:                               # %cond.end398.i
                                        #   in Loop: Header=BB0_18 Depth=2
	leal	-57(%r10), %eax
	cmpl	$17, %eax
	ja	.LBB0_35
# %bb.31:                               # %for.body499.i.preheader
                                        #   in Loop: Header=BB0_18 Depth=2
	movq	%r9, %rax
	movq	(%rsp), %rdx                    # 8-byte Reload
	.p2align	4, 0x90
.LBB0_32:                               # %for.body499.i
                                        #   Parent Loop BB0_17 Depth=1
                                        #     Parent Loop BB0_18 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_33 Depth 4
	xorl	%esi, %esi
	.p2align	4, 0x90
.LBB0_33:                               # %for.body514.i
                                        #   Parent Loop BB0_17 Depth=1
                                        #     Parent Loop BB0_18 Depth=2
                                        #       Parent Loop BB0_32 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	movsd	(%rax,%rsi,8), %xmm0            # xmm0 = mem[0],zero
	mulsd	%xmm2, %xmm0
	movsd	%xmm0, (%rax,%rsi,8)
	addq	$1, %rsi
	cmpq	$32, %rsi
	jne	.LBB0_33
# %bb.34:                               # %for.inc523.i
                                        #   in Loop: Header=BB0_32 Depth=3
	addq	$1, %rdx
	addq	$19200, %rax                    # imm = 0x4B00
	cmpq	%r13, %rdx
	jne	.LBB0_32
	jmp	.LBB0_35
	.p2align	4, 0x90
.LBB0_36:                               # %for.body549.i.preheader
                                        #   in Loop: Header=BB0_17 Depth=1
	movq	%r15, 32(%rsp)                  # 8-byte Spill
	movq	%r11, 64(%rsp)                  # 8-byte Spill
	movq	%r12, 72(%rsp)                  # 8-byte Spill
	movq	16(%rsp), %r12                  # 8-byte Reload
	xorl	%eax, %eax
	jmp	.LBB0_37
	.p2align	4, 0x90
.LBB0_40:                               # %for.inc590.i
                                        #   in Loop: Header=BB0_37 Depth=2
	movq	8(%rsp), %rax                   # 8-byte Reload
	addq	$1, %rax
	addq	$256, %r12                      # imm = 0x100
	cmpq	$57, %rax
	je	.LBB0_41
.LBB0_37:                               # %for.body549.i
                                        #   Parent Loop BB0_17 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_39 Depth 3
	movq	%rax, 8(%rsp)                   # 8-byte Spill
                                        # kill: def $eax killed $eax killed $rax def $rax
	shll	$5, %eax
	leal	31(%rax), %ecx
	cmpl	$1799, %ecx                     # imm = 0x707
	movl	$1799, %edx                     # imm = 0x707
	cmovael	%edx, %ecx
	subl	%ecx, %eax
	ja	.LBB0_40
# %bb.38:                               # %for.body564.i.preheader
                                        #   in Loop: Header=BB0_37 Depth=2
	negl	%eax
	leaq	8(,%rax,8), %rbp
	xorl	%r15d, %r15d
	.p2align	4, 0x90
.LBB0_39:                               # %for.body564.i
                                        #   Parent Loop BB0_17 Depth=1
                                        #     Parent Loop BB0_37 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	leaq	(%r12,%r15), %rdi
	xorl	%esi, %esi
	movq	%rbp, %rdx
	callq	memset
	addq	$14400, %r15                    # imm = 0x3840
	cmpq	$460800, %r15                   # imm = 0x70800
	jne	.LBB0_39
	jmp	.LBB0_40
	.p2align	4, 0x90
.LBB0_41:                               # %for.body612.i.preheader
                                        #   in Loop: Header=BB0_17 Depth=1
	xorl	%eax, %eax
	movq	24(%rsp), %r9                   # 8-byte Reload
	movq	%r9, %rcx
	movq	72(%rsp), %r12                  # 8-byte Reload
	movq	64(%rsp), %r11                  # 8-byte Reload
	movq	32(%rsp), %r15                  # 8-byte Reload
	xorpd	%xmm1, %xmm1
	movsd	.LCPI0_4(%rip), %xmm2           # xmm2 = mem[0],zero
	.p2align	4, 0x90
.LBB0_42:                               # %for.body612.i
                                        #   Parent Loop BB0_17 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_43 Depth 3
                                        #         Child Loop BB0_44 Depth 4
	movq	%rcx, %rdx
	movq	(%rsp), %rsi                    # 8-byte Reload
	.p2align	4, 0x90
.LBB0_43:                               # %for.body627.i
                                        #   Parent Loop BB0_17 Depth=1
                                        #     Parent Loop BB0_42 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_44 Depth 4
	xorl	%edi, %edi
	.p2align	4, 0x90
.LBB0_44:                               # %for.body642.i
                                        #   Parent Loop BB0_17 Depth=1
                                        #     Parent Loop BB0_42 Depth=2
                                        #       Parent Loop BB0_43 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	movsd	(%rdx,%rdi,8), %xmm0            # xmm0 = mem[0],zero
	mulsd	%xmm2, %xmm0
	movsd	%xmm0, (%rdx,%rdi,8)
	addq	$1, %rdi
	cmpq	$32, %rdi
	jne	.LBB0_44
# %bb.45:                               # %for.inc651.i
                                        #   in Loop: Header=BB0_43 Depth=3
	addq	$1, %rsi
	addq	$19200, %rdx                    # imm = 0x4B00
	cmpq	%r13, %rsi
	jne	.LBB0_43
# %bb.46:                               # %for.inc654.i
                                        #   in Loop: Header=BB0_42 Depth=2
	addl	$1, %eax
	addq	$256, %rcx                      # imm = 0x100
	cmpl	$75, %eax
	jne	.LBB0_42
# %bb.47:                               # %for.inc658.i
                                        #   in Loop: Header=BB0_17 Depth=1
	addq	$32, (%rsp)                     # 8-byte Folded Spill
	addq	$32, %r13
	addq	$1, %r12
	addq	$614400, %r11                   # imm = 0x96000
	addq	$614400, %r15                   # imm = 0x96000
	addq	$614400, %r9                    # imm = 0x96000
	addq	$460800, 16(%rsp)               # 8-byte Folded Spill
                                        # imm = 0x70800
	cmpq	$50, %r12
	jne	.LBB0_17
# %bb.48:                               # %for.cond680.preheader.i.preheader
	xorl	%r11d, %r11d
	movl	$32, %r12d
	movl	$1799, %r13d                    # imm = 0x707
	movsd	.LCPI0_5(%rip), %xmm0           # xmm0 = mem[0],zero
	movq	40(%rsp), %rax                  # 8-byte Reload
	movq	%rax, 88(%rsp)                  # 8-byte Spill
	movq	%rbx, %rax
	xorl	%ecx, %ecx
	jmp	.LBB0_49
	.p2align	4, 0x90
.LBB0_69:                               # %for.inc883.i
                                        #   in Loop: Header=BB0_49 Depth=1
	movl	84(%rsp), %ecx                  # 4-byte Reload
	addl	$1, %ecx
	addq	$32, %r11
	addq	$32, %r12
	movq	112(%rsp), %rax                 # 8-byte Reload
	addq	$460800, %rax                   # imm = 0x70800
	addq	$614400, 88(%rsp)               # 8-byte Folded Spill
                                        # imm = 0x96000
	cmpl	$50, %ecx
	je	.LBB0_70
.LBB0_49:                               # %for.cond680.preheader.i
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_50 Depth 2
                                        #       Child Loop BB0_52 Depth 3
                                        #         Child Loop BB0_54 Depth 4
                                        #           Child Loop BB0_55 Depth 5
                                        #             Child Loop BB0_56 Depth 6
                                        #       Child Loop BB0_61 Depth 3
                                        #         Child Loop BB0_62 Depth 4
                                        #           Child Loop BB0_63 Depth 5
                                        #             Child Loop BB0_64 Depth 6
	movl	%ecx, 84(%rsp)                  # 4-byte Spill
	movl	$31, %ecx
	movl	$1, %edx
	movq	%rdx, 32(%rsp)                  # 8-byte Spill
	movq	48(%rsp), %rdx                  # 8-byte Reload
	movq	%rdx, 24(%rsp)                  # 8-byte Spill
	movq	56(%rsp), %rdx                  # 8-byte Reload
	movq	%rdx, 96(%rsp)                  # 8-byte Spill
	movq	%rax, 112(%rsp)                 # 8-byte Spill
	movq	%rax, 8(%rsp)                   # 8-byte Spill
	xorl	%r15d, %r15d
	xorl	%eax, %eax
	movq	%r11, 16(%rsp)                  # 8-byte Spill
	jmp	.LBB0_50
	.p2align	4, 0x90
.LBB0_68:                               # %for.inc880.i
                                        #   in Loop: Header=BB0_50 Depth=2
	movl	64(%rsp), %eax                  # 4-byte Reload
	addl	$1, %eax
	addq	$32, %r15
	movl	72(%rsp), %ecx                  # 4-byte Reload
	addl	$32, %ecx
	addq	$-32, 32(%rsp)                  # 8-byte Folded Spill
	addq	$256, 8(%rsp)                   # 8-byte Folded Spill
                                        # imm = 0x100
	addq	$256, 96(%rsp)                  # 8-byte Folded Spill
                                        # imm = 0x100
	addq	$614400, 24(%rsp)               # 8-byte Folded Spill
                                        # imm = 0x96000
	cmpl	$57, %eax
	je	.LBB0_69
.LBB0_50:                               # %for.cond699.preheader.i
                                        #   Parent Loop BB0_49 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_52 Depth 3
                                        #         Child Loop BB0_54 Depth 4
                                        #           Child Loop BB0_55 Depth 5
                                        #             Child Loop BB0_56 Depth 6
                                        #       Child Loop BB0_61 Depth 3
                                        #         Child Loop BB0_62 Depth 4
                                        #           Child Loop BB0_63 Depth 5
                                        #             Child Loop BB0_64 Depth 6
	cmpl	$1799, %ecx                     # imm = 0x707
	movl	$1799, %r9d                     # imm = 0x707
	movl	%ecx, 72(%rsp)                  # 4-byte Spill
	cmovbl	%ecx, %r9d
	movl	%eax, 64(%rsp)                  # 4-byte Spill
                                        # kill: def $eax killed $eax def $rax
	shll	$5, %eax
	leal	31(%rax), %ecx
	cmpl	$1799, %ecx                     # imm = 0x707
	cmovael	%r13d, %ecx
	cmpl	%ecx, %eax
	ja	.LBB0_68
# %bb.51:                               # %for.body715.i.preheader
                                        #   in Loop: Header=BB0_50 Depth=2
	movq	32(%rsp), %rax                  # 8-byte Reload
	leaq	(%rax,%r9), %r10
	addl	$1, %r9d
	xorl	%r8d, %r8d
	movl	$31, %eax
	movq	96(%rsp), %r13                  # 8-byte Reload
	xorl	%ecx, %ecx
	jmp	.LBB0_52
	.p2align	4, 0x90
.LBB0_59:                               # %for.inc785.i
                                        #   in Loop: Header=BB0_52 Depth=3
	addl	$1, %ecx
	addq	$32, %r8
	movl	(%rsp), %eax                    # 4-byte Reload
	addl	$32, %eax
	addq	$460800, %r13                   # imm = 0x70800
	cmpl	$69, %ecx
	je	.LBB0_60
.LBB0_52:                               # %for.body715.i
                                        #   Parent Loop BB0_49 Depth=1
                                        #     Parent Loop BB0_50 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_54 Depth 4
                                        #           Child Loop BB0_55 Depth 5
                                        #             Child Loop BB0_56 Depth 6
	cmpl	$2199, %eax                     # imm = 0x897
	movl	$2199, %esi                     # imm = 0x897
	movl	%eax, (%rsp)                    # 4-byte Spill
	cmovbl	%eax, %esi
	movl	%ecx, %eax
	shll	$5, %eax
	leal	31(%rax), %edx
	cmpl	$2199, %edx                     # imm = 0x897
	movl	$2199, %edi                     # imm = 0x897
	cmovael	%edi, %edx
	cmpl	%edx, %eax
	ja	.LBB0_59
# %bb.53:                               # %for.body730.i.preheader
                                        #   in Loop: Header=BB0_52 Depth=3
	addl	$1, %esi
	movq	8(%rsp), %rdi                   # 8-byte Reload
	movq	16(%rsp), %rbp                  # 8-byte Reload
	.p2align	4, 0x90
.LBB0_54:                               # %for.body730.i
                                        #   Parent Loop BB0_49 Depth=1
                                        #     Parent Loop BB0_50 Depth=2
                                        #       Parent Loop BB0_52 Depth=3
                                        # =>      This Loop Header: Depth=4
                                        #           Child Loop BB0_55 Depth 5
                                        #             Child Loop BB0_56 Depth 6
	movq	%r13, %rax
	movq	%r8, %r11
	.p2align	4, 0x90
.LBB0_55:                               # %for.body745.i
                                        #   Parent Loop BB0_49 Depth=1
                                        #     Parent Loop BB0_50 Depth=2
                                        #       Parent Loop BB0_52 Depth=3
                                        #         Parent Loop BB0_54 Depth=4
                                        # =>        This Loop Header: Depth=5
                                        #             Child Loop BB0_56 Depth 6
	imulq	$17600, %rbp, %rdx              # imm = 0x44C0
	addq	%r14, %rdx
	movsd	(%rdx,%r11,8), %xmm1            # xmm1 = mem[0],zero
	mulsd	%xmm0, %xmm1
	xorl	%edx, %edx
	.p2align	4, 0x90
.LBB0_56:                               # %for.body760.i
                                        #   Parent Loop BB0_49 Depth=1
                                        #     Parent Loop BB0_50 Depth=2
                                        #       Parent Loop BB0_52 Depth=3
                                        #         Parent Loop BB0_54 Depth=4
                                        #           Parent Loop BB0_55 Depth=5
                                        # =>          This Inner Loop Header: Depth=6
	movsd	(%rax,%rdx,8), %xmm2            # xmm2 = mem[0],zero
	mulsd	%xmm1, %xmm2
	addsd	(%rdi,%rdx,8), %xmm2
	movsd	%xmm2, (%rdi,%rdx,8)
	addq	$1, %rdx
	cmpq	%rdx, %r10
	jne	.LBB0_56
# %bb.57:                               # %for.inc779.i
                                        #   in Loop: Header=BB0_55 Depth=5
	addq	$1, %r11
	addq	$14400, %rax                    # imm = 0x3840
	cmpq	%rsi, %r11
	jne	.LBB0_55
# %bb.58:                               # %for.inc782.i
                                        #   in Loop: Header=BB0_54 Depth=4
	addq	$1, %rbp
	addq	$14400, %rdi                    # imm = 0x3840
	cmpq	%r12, %rbp
	jne	.LBB0_54
	jmp	.LBB0_59
	.p2align	4, 0x90
.LBB0_60:                               # %for.body807.i.preheader
                                        #   in Loop: Header=BB0_50 Depth=2
	xorl	%r10d, %r10d
	movq	24(%rsp), %rax                  # 8-byte Reload
	movq	88(%rsp), %r8                   # 8-byte Reload
	movq	16(%rsp), %r11                  # 8-byte Reload
	movl	$1799, %r13d                    # imm = 0x707
	.p2align	4, 0x90
.LBB0_61:                               # %for.body807.i
                                        #   Parent Loop BB0_49 Depth=1
                                        #     Parent Loop BB0_50 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_62 Depth 4
                                        #           Child Loop BB0_63 Depth 5
                                        #             Child Loop BB0_64 Depth 6
	movq	%r8, %rdi
	movq	%r11, %rsi
	.p2align	4, 0x90
.LBB0_62:                               # %for.body822.i
                                        #   Parent Loop BB0_49 Depth=1
                                        #     Parent Loop BB0_50 Depth=2
                                        #       Parent Loop BB0_61 Depth=3
                                        # =>      This Loop Header: Depth=4
                                        #           Child Loop BB0_63 Depth 5
                                        #             Child Loop BB0_64 Depth 6
	movq	%rax, %rbp
	movq	%r15, %rdx
	.p2align	4, 0x90
.LBB0_63:                               # %for.body837.i
                                        #   Parent Loop BB0_49 Depth=1
                                        #     Parent Loop BB0_50 Depth=2
                                        #       Parent Loop BB0_61 Depth=3
                                        #         Parent Loop BB0_62 Depth=4
                                        # =>        This Loop Header: Depth=5
                                        #             Child Loop BB0_64 Depth 6
	imulq	$14400, %rsi, %rcx              # imm = 0x3840
	addq	%rbx, %rcx
	movsd	(%rcx,%rdx,8), %xmm1            # xmm1 = mem[0],zero
	xorl	%ecx, %ecx
	.p2align	4, 0x90
.LBB0_64:                               # %for.body852.i
                                        #   Parent Loop BB0_49 Depth=1
                                        #     Parent Loop BB0_50 Depth=2
                                        #       Parent Loop BB0_61 Depth=3
                                        #         Parent Loop BB0_62 Depth=4
                                        #           Parent Loop BB0_63 Depth=5
                                        # =>          This Inner Loop Header: Depth=6
	movsd	(%rbp,%rcx,8), %xmm2            # xmm2 = mem[0],zero
	mulsd	%xmm1, %xmm2
	addsd	(%rdi,%rcx,8), %xmm2
	movsd	%xmm2, (%rdi,%rcx,8)
	addq	$1, %rcx
	cmpq	$32, %rcx
	jne	.LBB0_64
# %bb.65:                               # %for.inc870.i
                                        #   in Loop: Header=BB0_63 Depth=5
	addq	$1, %rdx
	addq	$19200, %rbp                    # imm = 0x4B00
	cmpq	%r9, %rdx
	jne	.LBB0_63
# %bb.66:                               # %for.inc873.i
                                        #   in Loop: Header=BB0_62 Depth=4
	addq	$1, %rsi
	addq	$19200, %rdi                    # imm = 0x4B00
	cmpq	%r12, %rsi
	jne	.LBB0_62
# %bb.67:                               # %for.inc876.i
                                        #   in Loop: Header=BB0_61 Depth=3
	addl	$1, %r10d
	addq	$256, %r8                       # imm = 0x100
	addq	$256, %rax                      # imm = 0x100
	cmpl	$75, %r10d
	jne	.LBB0_61
	jmp	.LBB0_68
.LBB0_70:                               # %kernel_2mm.exit
	xorl	%eax, %eax
	callq	polybench_timer_stop
	xorl	%eax, %eax
	callq	polybench_timer_print
	cmpl	$43, 80(%rsp)                   # 4-byte Folded Reload
	jl	.LBB0_79
# %bb.71:                               # %land.lhs.true
	movq	104(%rsp), %rax                 # 8-byte Reload
	movq	(%rax), %rax
	cmpb	$0, (%rax)
	je	.LBB0_72
.LBB0_79:                               # %if.end
	movq	%rbx, %rdi
	callq	free
	movq	%r14, %rdi
	callq	free
	movq	56(%rsp), %rdi                  # 8-byte Reload
	callq	free
	movq	48(%rsp), %rdi                  # 8-byte Reload
	callq	free
	movq	40(%rsp), %rdi                  # 8-byte Reload
	callq	free
	xorl	%eax, %eax
	addq	$120, %rsp
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
.LBB0_72:                               # %if.then
	.cfi_def_cfa_offset 176
	movq	stderr(%rip), %rcx
	movl	$.L.str.1, %edi
	movl	$22, %esi
	movl	$1, %edx
	callq	fwrite
	movq	stderr(%rip), %rdi
	xorl	%ebp, %ebp
	movl	$.L.str.2, %esi
	movl	$.L.str.3, %edx
	xorl	%eax, %eax
	callq	fprintf
	xorl	%ecx, %ecx
	movq	40(%rsp), %r13                  # 8-byte Reload
	xorl	%eax, %eax
.LBB0_73:                               # %for.cond2.preheader.i
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_74 Depth 2
	movq	%rax, 8(%rsp)                   # 8-byte Spill
	movq	%rbp, (%rsp)                    # 8-byte Spill
	movl	%ebp, %r12d
	xorl	%r15d, %r15d
.LBB0_74:                               # %for.body4.i
                                        #   Parent Loop BB0_73 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movl	%r12d, %eax
	movl	$3435973837, %edx               # imm = 0xCCCCCCCD
	imulq	%rdx, %rax
	shrq	$36, %rax
	leal	(%rax,%rax,4), %eax
	movq	%rcx, %rbp
	leal	(%rcx,%rax,4), %eax
	cmpl	%r15d, %eax
	jne	.LBB0_76
# %bb.75:                               # %if.then.i
                                        #   in Loop: Header=BB0_74 Depth=2
	movq	stderr(%rip), %rsi
	movl	$10, %edi
	callq	fputc
.LBB0_76:                               # %if.end.i
                                        #   in Loop: Header=BB0_74 Depth=2
	movq	stderr(%rip), %rdi
	movsd	(%r13,%r15,8), %xmm0            # xmm0 = mem[0],zero
	movl	$.L.str.5, %esi
	movb	$1, %al
	callq	fprintf
	addq	$1, %r15
	addl	$1, %r12d
	cmpq	$2400, %r15                     # imm = 0x960
	movq	%rbp, %rcx
	jne	.LBB0_74
# %bb.77:                               # %for.inc10.i
                                        #   in Loop: Header=BB0_73 Depth=1
	movq	8(%rsp), %rax                   # 8-byte Reload
	addq	$1, %rax
	addq	$19200, %r13                    # imm = 0x4B00
	addl	$-1600, %ecx                    # imm = 0xF9C0
	movq	(%rsp), %rbp                    # 8-byte Reload
	addl	$1600, %ebp                     # imm = 0x640
	cmpq	$1600, %rax                     # imm = 0x640
	jne	.LBB0_73
# %bb.78:                               # %print_array.exit
	movq	stderr(%rip), %rdi
	movl	$.L.str.6, %esi
	movl	$.L.str.3, %edx
	xorl	%eax, %eax
	callq	fprintf
	movq	stderr(%rip), %rcx
	movl	$.L.str.7, %edi
	movl	$22, %esi
	movl	$1, %edx
	callq	fwrite
	jmp	.LBB0_79
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
	.asciz	"D"
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

	.ident	"clang version 12.0.0 (https://github.com/wsmoses/MLIR-GPU 1112d5451cea635029a160c950f14a85f31b2258)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
