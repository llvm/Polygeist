	.text
	.file	"2mm.pluto.i64.c"
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
	subq	$136, %rsp
	.cfi_def_cfa_offset 192
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rsi, 104(%rsp)                 # 8-byte Spill
	movl	%edi, 84(%rsp)                  # 4-byte Spill
	movl	$23040000, %edi                 # imm = 0x15F9000
	callq	malloc
	movq	%rax, %rbx
	movl	$28160000, %edi                 # imm = 0x1ADB000
	callq	malloc
	movq	%rax, %r14
	movl	$31680000, %edi                 # imm = 0x1E36600
	callq	malloc
	movq	%rax, 64(%rsp)                  # 8-byte Spill
	movl	$34560000, %edi                 # imm = 0x20F5800
	callq	malloc
	movq	%rax, 56(%rsp)                  # 8-byte Spill
	movl	$30720000, %edi                 # imm = 0x1D4C000
	callq	malloc
	movq	%rax, 8(%rsp)                   # 8-byte Spill
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
	movq	64(%rsp), %rdx                  # 8-byte Reload
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
	movq	56(%rsp), %rdx                  # 8-byte Reload
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
	movq	8(%rsp), %rcx                   # 8-byte Reload
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
	xorl	%r15d, %r15d
	xorl	%eax, %eax
	callq	polybench_timer_start
	movq	8(%rsp), %rax                   # 8-byte Reload
	leaq	14336(%rax), %r11
	movq	%rax, %r12
	addq	$14400, %r12                    # imm = 0x3840
	movl	$32, %r13d
	xorpd	%xmm1, %xmm1
	movsd	.LCPI0_4(%rip), %xmm2           # xmm2 = mem[0],zero
	movq	%rbx, 16(%rsp)                  # 8-byte Spill
	movq	%rax, %r9
	.p2align	4, 0x90
.LBB0_17:                               # %cond.end214.preheader.i
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
	imulq	$460800, %r15, %r8              # imm = 0x70800
	leaq	14336(%r8), %rcx
	movq	%r15, %rax
	shlq	$5, %rax
	movq	%rax, (%rsp)                    # 8-byte Spill
	movq	%r9, 32(%rsp)                   # 8-byte Spill
	xorl	%r10d, %r10d
	jmp	.LBB0_18
	.p2align	4, 0x90
.LBB0_35:                               # %for.inc600.i
                                        #   in Loop: Header=BB0_18 Depth=2
	addq	$1, %r10
	addq	$256, %r9                       # imm = 0x100
	cmpq	$132, %r10
	je	.LBB0_36
.LBB0_18:                               # %cond.end214.i
                                        #   Parent Loop BB0_17 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_20 Depth 3
                                        #         Child Loop BB0_21 Depth 4
                                        #         Child Loop BB0_23 Depth 4
                                        #       Child Loop BB0_27 Depth 3
                                        #         Child Loop BB0_28 Depth 4
                                        #       Child Loop BB0_32 Depth 3
                                        #         Child Loop BB0_33 Depth 4
	cmpq	$56, %r10
	jne	.LBB0_25
# %bb.19:                               # %for.cond259.preheader.i.preheader
                                        #   in Loop: Header=BB0_18 Depth=2
	movq	%r12, %rdi
	movq	%r11, %rbp
	xorl	%eax, %eax
	movq	(%rsp), %rdx                    # 8-byte Reload
	.p2align	4, 0x90
.LBB0_20:                               # %for.cond259.preheader.i
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
.LBB0_21:                               # %for.body264.i
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
# %bb.22:                               # %for.body291.i.preheader
                                        #   in Loop: Header=BB0_20 Depth=3
	xorl	%esi, %esi
	.p2align	4, 0x90
.LBB0_23:                               # %for.body291.i
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
# %bb.24:                               # %for.inc298.i
                                        #   in Loop: Header=BB0_20 Depth=3
	addq	$1, %rdx
	addq	$1, %rax
	addq	$19200, %rbp                    # imm = 0x4B00
	addq	$19200, %rdi                    # imm = 0x4B00
	cmpq	%r13, %rdx
	jne	.LBB0_20
.LBB0_25:                               # %cond.end391.i
                                        #   in Loop: Header=BB0_18 Depth=2
	cmpq	$55, %r10
	ja	.LBB0_30
# %bb.26:                               # %for.body415.lr.ph.i
                                        #   in Loop: Header=BB0_18 Depth=2
	movq	%r10, %rax
	shlq	$8, %rax
	addq	%r8, %rax
	movq	%r9, %rdx
	xorl	%edi, %edi
	movq	(%rsp), %rbp                    # 8-byte Reload
	.p2align	4, 0x90
.LBB0_27:                               # %for.body415.i
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
.LBB0_28:                               # %for.body422.i
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
# %bb.29:                               # %for.inc431.i
                                        #   in Loop: Header=BB0_27 Depth=3
	addq	$1, %rbp
	addq	$1, %rdi
	addq	$19200, %rdx                    # imm = 0x4B00
	cmpq	%r13, %rbp
	jne	.LBB0_27
.LBB0_30:                               # %cond.end448.i
                                        #   in Loop: Header=BB0_18 Depth=2
	leaq	-57(%r10), %rax
	cmpq	$17, %rax
	ja	.LBB0_35
# %bb.31:                               # %for.body570.lr.ph.i
                                        #   in Loop: Header=BB0_18 Depth=2
	movq	%r9, %rax
	movq	(%rsp), %rdx                    # 8-byte Reload
	.p2align	4, 0x90
.LBB0_32:                               # %for.body570.i
                                        #   Parent Loop BB0_17 Depth=1
                                        #     Parent Loop BB0_18 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_33 Depth 4
	xorl	%esi, %esi
	.p2align	4, 0x90
.LBB0_33:                               # %for.body589.i
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
# %bb.34:                               # %for.inc596.i
                                        #   in Loop: Header=BB0_32 Depth=3
	addq	$1, %rdx
	addq	$19200, %rax                    # imm = 0x4B00
	cmpq	%r13, %rdx
	jne	.LBB0_32
	jmp	.LBB0_35
	.p2align	4, 0x90
.LBB0_36:                               # %for.body626.i.preheader
                                        #   in Loop: Header=BB0_17 Depth=1
	movq	%r12, 40(%rsp)                  # 8-byte Spill
	movq	%r11, 72(%rsp)                  # 8-byte Spill
	movq	%r15, 48(%rsp)                  # 8-byte Spill
	movq	16(%rsp), %r12                  # 8-byte Reload
	xorl	%eax, %eax
	jmp	.LBB0_37
	.p2align	4, 0x90
.LBB0_40:                               # %for.inc673.i
                                        #   in Loop: Header=BB0_37 Depth=2
	movq	24(%rsp), %rax                  # 8-byte Reload
	addq	$1, %rax
	addq	$256, %r12                      # imm = 0x100
	cmpq	$57, %rax
	je	.LBB0_41
.LBB0_37:                               # %for.body626.i
                                        #   Parent Loop BB0_17 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_39 Depth 3
	movq	%rax, 24(%rsp)                  # 8-byte Spill
	shlq	$5, %rax
	leaq	31(%rax), %rdx
	cmpq	$1799, %rdx                     # imm = 0x707
	movl	$1799, %ecx                     # imm = 0x707
	cmovaeq	%rcx, %rdx
	cmpq	%rax, %rdx
	movq	%rax, %rcx
	cmovaq	%rdx, %rcx
	cmpq	%rdx, %rax
	ja	.LBB0_40
# %bb.38:                               # %for.body645.i.preheader
                                        #   in Loop: Header=BB0_37 Depth=2
	subq	%rax, %rcx
	leaq	8(,%rcx,8), %rbp
	xorl	%r15d, %r15d
	.p2align	4, 0x90
.LBB0_39:                               # %for.body645.i
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
.LBB0_41:                               # %for.body718.preheader.i.preheader
                                        #   in Loop: Header=BB0_17 Depth=1
	movq	32(%rsp), %r9                   # 8-byte Reload
	movq	%r9, %rax
	xorl	%ecx, %ecx
	movq	48(%rsp), %r15                  # 8-byte Reload
	movq	72(%rsp), %r11                  # 8-byte Reload
	movq	40(%rsp), %r12                  # 8-byte Reload
	xorpd	%xmm1, %xmm1
	movsd	.LCPI0_4(%rip), %xmm2           # xmm2 = mem[0],zero
	.p2align	4, 0x90
.LBB0_42:                               # %for.body718.preheader.i
                                        #   Parent Loop BB0_17 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_43 Depth 3
                                        #         Child Loop BB0_44 Depth 4
	movq	%rax, %rdx
	movq	(%rsp), %rsi                    # 8-byte Reload
	.p2align	4, 0x90
.LBB0_43:                               # %for.body718.i
                                        #   Parent Loop BB0_17 Depth=1
                                        #     Parent Loop BB0_42 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_44 Depth 4
	xorl	%edi, %edi
	.p2align	4, 0x90
.LBB0_44:                               # %for.body737.i
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
# %bb.45:                               # %for.inc744.i
                                        #   in Loop: Header=BB0_43 Depth=3
	addq	$1, %rsi
	addq	$19200, %rdx                    # imm = 0x4B00
	cmpq	%r13, %rsi
	jne	.LBB0_43
# %bb.46:                               # %for.inc747.i
                                        #   in Loop: Header=BB0_42 Depth=2
	addq	$1, %rcx
	addq	$256, %rax                      # imm = 0x100
	cmpq	$75, %rcx
	jne	.LBB0_42
# %bb.47:                               # %for.inc751.i
                                        #   in Loop: Header=BB0_17 Depth=1
	addq	$1, %r15
	addq	$32, %r13
	addq	$614400, %r11                   # imm = 0x96000
	addq	$614400, %r12                   # imm = 0x96000
	addq	$614400, %r9                    # imm = 0x96000
	addq	$460800, 16(%rsp)               # 8-byte Folded Spill
                                        # imm = 0x70800
	cmpq	$50, %r15
	jne	.LBB0_17
# %bb.48:                               # %for.cond777.preheader.i.preheader
	movl	$32, %r9d
	xorl	%eax, %eax
	movl	$1799, %r15d                    # imm = 0x707
	movsd	.LCPI0_5(%rip), %xmm0           # xmm0 = mem[0],zero
	movq	8(%rsp), %rcx                   # 8-byte Reload
	movq	%rcx, 88(%rsp)                  # 8-byte Spill
	movq	%rbx, %rcx
	jmp	.LBB0_49
	.p2align	4, 0x90
.LBB0_69:                               # %for.inc1003.i
                                        #   in Loop: Header=BB0_49 Depth=1
	movq	120(%rsp), %rax                 # 8-byte Reload
	addq	$1, %rax
	addq	$32, %r9
	movq	112(%rsp), %rcx                 # 8-byte Reload
	addq	$460800, %rcx                   # imm = 0x70800
	addq	$614400, 88(%rsp)               # 8-byte Folded Spill
                                        # imm = 0x96000
	cmpq	$50, %rax
	je	.LBB0_70
.LBB0_49:                               # %for.cond777.preheader.i
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
	movq	%rax, 120(%rsp)                 # 8-byte Spill
	movq	%rax, %r12
	shlq	$5, %r12
	movl	$31, %r13d
	movl	$1, %eax
	movq	%rax, 40(%rsp)                  # 8-byte Spill
	movq	56(%rsp), %rax                  # 8-byte Reload
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	movq	64(%rsp), %rax                  # 8-byte Reload
	movq	%rax, 96(%rsp)                  # 8-byte Spill
	movq	%rcx, 112(%rsp)                 # 8-byte Spill
	movq	%rcx, 16(%rsp)                  # 8-byte Spill
	xorl	%eax, %eax
	movq	%r12, 48(%rsp)                  # 8-byte Spill
	jmp	.LBB0_50
	.p2align	4, 0x90
.LBB0_68:                               # %for.inc1000.i
                                        #   in Loop: Header=BB0_50 Depth=2
	movq	72(%rsp), %rax                  # 8-byte Reload
	addq	$1, %rax
	addq	$32, %r13
	addq	$-32, 40(%rsp)                  # 8-byte Folded Spill
	addq	$256, 16(%rsp)                  # 8-byte Folded Spill
                                        # imm = 0x100
	addq	$256, 96(%rsp)                  # 8-byte Folded Spill
                                        # imm = 0x100
	addq	$614400, 32(%rsp)               # 8-byte Folded Spill
                                        # imm = 0x96000
	cmpq	$57, %rax
	je	.LBB0_69
.LBB0_50:                               # %for.cond800.preheader.i
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
	cmpq	$1799, %r13                     # imm = 0x707
	movl	$1799, %r8d                     # imm = 0x707
	cmovbq	%r13, %r8
	movq	%rax, 72(%rsp)                  # 8-byte Spill
	movq	%rax, %rcx
	shlq	$5, %rcx
	leaq	31(%rcx), %rax
	cmpq	$1799, %rax                     # imm = 0x707
	cmovaeq	%r15, %rax
	movq	%rcx, (%rsp)                    # 8-byte Spill
	cmpq	%rax, %rcx
	ja	.LBB0_68
# %bb.51:                               # %for.body838.lr.ph.i.preheader
                                        #   in Loop: Header=BB0_50 Depth=2
	movq	%r13, 128(%rsp)                 # 8-byte Spill
	movq	40(%rsp), %rax                  # 8-byte Reload
	leaq	(%r8,%rax), %r10
	movl	$31, %eax
	movq	96(%rsp), %r15                  # 8-byte Reload
	xorl	%edx, %edx
	jmp	.LBB0_52
	.p2align	4, 0x90
.LBB0_59:                               # %for.inc895.i
                                        #   in Loop: Header=BB0_52 Depth=3
	addq	$1, %rdx
	movq	24(%rsp), %rax                  # 8-byte Reload
	addq	$32, %rax
	addq	$460800, %r15                   # imm = 0x70800
	cmpq	$69, %rdx
	je	.LBB0_60
.LBB0_52:                               # %for.body838.lr.ph.i
                                        #   Parent Loop BB0_49 Depth=1
                                        #     Parent Loop BB0_50 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_54 Depth 4
                                        #           Child Loop BB0_55 Depth 5
                                        #             Child Loop BB0_56 Depth 6
	cmpq	$2199, %rax                     # imm = 0x897
	movl	$2199, %esi                     # imm = 0x897
	movq	%rax, 24(%rsp)                  # 8-byte Spill
	cmovbq	%rax, %rsi
	movq	%rdx, %r12
	shlq	$5, %r12
	leaq	31(%r12), %rax
	cmpq	$2199, %rax                     # imm = 0x897
	movl	$2199, %ecx                     # imm = 0x897
	cmovaeq	%rcx, %rax
	cmpq	%rax, %r12
	ja	.LBB0_59
# %bb.53:                               # %for.body838.i.preheader
                                        #   in Loop: Header=BB0_52 Depth=3
	movq	16(%rsp), %rbp                  # 8-byte Reload
	movq	48(%rsp), %rcx                  # 8-byte Reload
	.p2align	4, 0x90
.LBB0_54:                               # %for.body838.i
                                        #   Parent Loop BB0_49 Depth=1
                                        #     Parent Loop BB0_50 Depth=2
                                        #       Parent Loop BB0_52 Depth=3
                                        # =>      This Loop Header: Depth=4
                                        #           Child Loop BB0_55 Depth 5
                                        #             Child Loop BB0_56 Depth 6
	movq	%r15, %r13
	movq	%r12, %rdi
	.p2align	4, 0x90
.LBB0_55:                               # %for.body857.i
                                        #   Parent Loop BB0_49 Depth=1
                                        #     Parent Loop BB0_50 Depth=2
                                        #       Parent Loop BB0_52 Depth=3
                                        #         Parent Loop BB0_54 Depth=4
                                        # =>        This Loop Header: Depth=5
                                        #             Child Loop BB0_56 Depth 6
	imulq	$17600, %rcx, %rax              # imm = 0x44C0
	addq	%r14, %rax
	movsd	(%rax,%rdi,8), %xmm1            # xmm1 = mem[0],zero
	mulsd	%xmm0, %xmm1
	xorl	%r11d, %r11d
	.p2align	4, 0x90
.LBB0_56:                               # %for.body876.i
                                        #   Parent Loop BB0_49 Depth=1
                                        #     Parent Loop BB0_50 Depth=2
                                        #       Parent Loop BB0_52 Depth=3
                                        #         Parent Loop BB0_54 Depth=4
                                        #           Parent Loop BB0_55 Depth=5
                                        # =>          This Inner Loop Header: Depth=6
	movsd	(%r13,%r11,8), %xmm2            # xmm2 = mem[0],zero
	mulsd	%xmm1, %xmm2
	addsd	(%rbp,%r11,8), %xmm2
	movsd	%xmm2, (%rbp,%r11,8)
	addq	$1, %r11
	cmpq	%r11, %r10
	jne	.LBB0_56
# %bb.57:                               # %for.inc889.i
                                        #   in Loop: Header=BB0_55 Depth=5
	leaq	1(%rdi), %rax
	addq	$14400, %r13                    # imm = 0x3840
	cmpq	%rsi, %rdi
	movq	%rax, %rdi
	jne	.LBB0_55
# %bb.58:                               # %for.inc892.i
                                        #   in Loop: Header=BB0_54 Depth=4
	addq	$1, %rcx
	addq	$14400, %rbp                    # imm = 0x3840
	cmpq	%r9, %rcx
	jne	.LBB0_54
	jmp	.LBB0_59
	.p2align	4, 0x90
.LBB0_60:                               # %for.body940.lr.ph.i.preheader
                                        #   in Loop: Header=BB0_50 Depth=2
	movq	32(%rsp), %rdi                  # 8-byte Reload
	movq	88(%rsp), %r10                  # 8-byte Reload
	xorl	%r11d, %r11d
	movl	$1799, %r15d                    # imm = 0x707
	movq	48(%rsp), %r12                  # 8-byte Reload
	movq	128(%rsp), %r13                 # 8-byte Reload
	.p2align	4, 0x90
.LBB0_61:                               # %for.body940.lr.ph.i
                                        #   Parent Loop BB0_49 Depth=1
                                        #     Parent Loop BB0_50 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_62 Depth 4
                                        #           Child Loop BB0_63 Depth 5
                                        #             Child Loop BB0_64 Depth 6
	movq	%r10, %rcx
	movq	%r12, %rbp
	.p2align	4, 0x90
.LBB0_62:                               # %for.body940.i
                                        #   Parent Loop BB0_49 Depth=1
                                        #     Parent Loop BB0_50 Depth=2
                                        #       Parent Loop BB0_61 Depth=3
                                        # =>      This Loop Header: Depth=4
                                        #           Child Loop BB0_63 Depth 5
                                        #             Child Loop BB0_64 Depth 6
	movq	%rdi, %rsi
	movq	(%rsp), %rdx                    # 8-byte Reload
	.p2align	4, 0x90
.LBB0_63:                               # %for.body959.i
                                        #   Parent Loop BB0_49 Depth=1
                                        #     Parent Loop BB0_50 Depth=2
                                        #       Parent Loop BB0_61 Depth=3
                                        #         Parent Loop BB0_62 Depth=4
                                        # =>        This Loop Header: Depth=5
                                        #             Child Loop BB0_64 Depth 6
	imulq	$14400, %rbp, %rax              # imm = 0x3840
	addq	%rbx, %rax
	movsd	(%rax,%rdx,8), %xmm1            # xmm1 = mem[0],zero
	xorl	%eax, %eax
	.p2align	4, 0x90
.LBB0_64:                               # %for.body978.i
                                        #   Parent Loop BB0_49 Depth=1
                                        #     Parent Loop BB0_50 Depth=2
                                        #       Parent Loop BB0_61 Depth=3
                                        #         Parent Loop BB0_62 Depth=4
                                        #           Parent Loop BB0_63 Depth=5
                                        # =>          This Inner Loop Header: Depth=6
	movsd	(%rsi,%rax,8), %xmm2            # xmm2 = mem[0],zero
	mulsd	%xmm1, %xmm2
	addsd	(%rcx,%rax,8), %xmm2
	movsd	%xmm2, (%rcx,%rax,8)
	addq	$1, %rax
	cmpq	$32, %rax
	jne	.LBB0_64
# %bb.65:                               # %for.inc990.i
                                        #   in Loop: Header=BB0_63 Depth=5
	leaq	1(%rdx), %rax
	addq	$19200, %rsi                    # imm = 0x4B00
	cmpq	%r8, %rdx
	movq	%rax, %rdx
	jne	.LBB0_63
# %bb.66:                               # %for.inc993.i
                                        #   in Loop: Header=BB0_62 Depth=4
	addq	$1, %rbp
	addq	$19200, %rcx                    # imm = 0x4B00
	cmpq	%r9, %rbp
	jne	.LBB0_62
# %bb.67:                               # %for.inc996.i
                                        #   in Loop: Header=BB0_61 Depth=3
	addq	$1, %r11
	addq	$256, %r10                      # imm = 0x100
	addq	$256, %rdi                      # imm = 0x100
	cmpq	$75, %r11
	jne	.LBB0_61
	jmp	.LBB0_68
.LBB0_70:                               # %kernel_2mm.exit
	xorl	%eax, %eax
	callq	polybench_timer_stop
	xorl	%eax, %eax
	callq	polybench_timer_print
	cmpl	$43, 84(%rsp)                   # 4-byte Folded Reload
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
	movq	64(%rsp), %rdi                  # 8-byte Reload
	callq	free
	movq	56(%rsp), %rdi                  # 8-byte Reload
	callq	free
	movq	8(%rsp), %rdi                   # 8-byte Reload
	callq	free
	xorl	%eax, %eax
	addq	$136, %rsp
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
	.cfi_def_cfa_offset 192
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
	movq	8(%rsp), %r13                   # 8-byte Reload
	xorl	%eax, %eax
.LBB0_73:                               # %for.cond2.preheader.i
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_74 Depth 2
	movq	%rax, 24(%rsp)                  # 8-byte Spill
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
	movq	24(%rsp), %rax                  # 8-byte Reload
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
