	.text
	.file	"seidel-2d.pluto.1d.c"
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3                               # -- Begin function main
.LCPI0_0:
	.quad	0x4000000000000000              # double 2
.LCPI0_1:
	.quad	0x40af400000000000              # double 4000
.LCPI0_2:
	.quad	0x4022000000000000              # double 9
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
	subq	$312, %rsp                      # imm = 0x138
	.cfi_def_cfa_offset 368
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rsi, 128(%rsp)                 # 8-byte Spill
	movl	%edi, 44(%rsp)                  # 4-byte Spill
	movl	$128000000, %edi                # imm = 0x7A12000
	callq	malloc
	movq	%rax, %r12
	addq	$8, %rax
	xorl	%ecx, %ecx
	movsd	.LCPI0_0(%rip), %xmm0           # xmm0 = mem[0],zero
	movsd	.LCPI0_1(%rip), %xmm1           # xmm1 = mem[0],zero
	.p2align	4, 0x90
.LBB0_1:                                # %for.cond1.preheader.i
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_2 Depth 2
	xorps	%xmm2, %xmm2
	cvtsi2sd	%ecx, %xmm2
	xorl	%edx, %edx
	.p2align	4, 0x90
.LBB0_2:                                # %for.body3.i
                                        #   Parent Loop BB0_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	leaq	2(%rdx), %rsi
	xorps	%xmm3, %xmm3
	cvtsi2sd	%esi, %xmm3
	mulsd	%xmm2, %xmm3
	addsd	%xmm0, %xmm3
	divsd	%xmm1, %xmm3
	movsd	%xmm3, -8(%rax,%rdx,8)
	leal	3(%rdx), %edi
	xorps	%xmm3, %xmm3
	cvtsi2sd	%edi, %xmm3
	mulsd	%xmm2, %xmm3
	addsd	%xmm0, %xmm3
	divsd	%xmm1, %xmm3
	movsd	%xmm3, (%rax,%rdx,8)
	movq	%rsi, %rdx
	cmpq	$4000, %rsi                     # imm = 0xFA0
	jne	.LBB0_2
# %bb.3:                                # %for.inc9.i
                                        #   in Loop: Header=BB0_1 Depth=1
	addq	$1, %rcx
	addq	$32000, %rax                    # imm = 0x7D00
	cmpq	$4000, %rcx                     # imm = 0xFA0
	jne	.LBB0_1
# %bb.4:                                # %init_array.exit
	xorl	%ebx, %ebx
	xorl	%eax, %eax
	callq	polybench_timer_start
	movl	$126, %eax
	movq	%rax, 48(%rsp)                  # 8-byte Spill
	movq	$-3998, %rax                    # imm = 0xF062
	movq	$-4029, %rcx                    # imm = 0xF043
	movsd	.LCPI0_2(%rip), %xmm0           # xmm0 = mem[0],zero
	xorl	%esi, %esi
	movabsq	$-4294967296, %r11              # imm = 0xFFFFFFFF00000000
	jmp	.LBB0_5
	.p2align	4, 0x90
.LBB0_22:                               # %for.inc1555.i
                                        #   in Loop: Header=BB0_5 Depth=1
	addq	$1, %rsi
	addq	$1, 48(%rsp)                    # 8-byte Folded Spill
	addq	$32, %rbx
	movq	136(%rsp), %rax                 # 8-byte Reload
	addq	$32, %rax
	addq	$-32, %rcx
	cmpq	$32, %rsi
	je	.LBB0_23
.LBB0_5:                                # %for.body90.lr.ph.i
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_6 Depth 2
                                        #       Child Loop BB0_11 Depth 3
                                        #         Child Loop BB0_14 Depth 4
                                        #           Child Loop BB0_16 Depth 5
                                        #             Child Loop BB0_18 Depth 6
	movq	%rcx, 184(%rsp)                 # 8-byte Spill
	movq	%rsi, %rdi
	shlq	$5, %rdi
	leaq	4029(%rdi), %rbp
	shrq	$4, %rbp
	cmpq	$312, %rbp                      # imm = 0x138
	movl	$312, %edx                      # imm = 0x138
	cmovaeq	%rdx, %rbp
	movq	%rbp, 160(%rsp)                 # 8-byte Spill
	leaq	4060(%rdi), %rdx
	movq	%rdx, 152(%rsp)                 # 8-byte Spill
	movq	%rdi, 168(%rsp)                 # 8-byte Spill
	addq	$31, %rdi
	cmpq	$999, %rdi                      # imm = 0x3E7
	movl	$999, %edx                      # imm = 0x3E7
	cmovaeq	%rdx, %rdi
	movq	%rdi, 144(%rsp)                 # 8-byte Spill
	movq	%rcx, 56(%rsp)                  # 8-byte Spill
	movq	%rax, 136(%rsp)                 # 8-byte Spill
	movq	%rax, 64(%rsp)                  # 8-byte Spill
	movq	%rbx, 80(%rsp)                  # 8-byte Spill
	movq	%rsi, %rdi
	movq	%rsi, 176(%rsp)                 # 8-byte Spill
	movq	%rbx, 200(%rsp)                 # 8-byte Spill
	jmp	.LBB0_6
	.p2align	4, 0x90
.LBB0_21:                               # %for.inc1552.i
                                        #   in Loop: Header=BB0_6 Depth=2
	movq	192(%rsp), %rdi                 # 8-byte Reload
	addq	$1, %rdi
	addq	$32, 80(%rsp)                   # 8-byte Folded Spill
	addq	$32, 64(%rsp)                   # 8-byte Folded Spill
	addq	$-32, 56(%rsp)                  # 8-byte Folded Spill
	cmpq	48(%rsp), %rdi                  # 8-byte Folded Reload
	movq	184(%rsp), %rcx                 # 8-byte Reload
	movq	176(%rsp), %rsi                 # 8-byte Reload
	je	.LBB0_22
.LBB0_6:                                # %for.body90.i
                                        #   Parent Loop BB0_5 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_11 Depth 3
                                        #         Child Loop BB0_14 Depth 4
                                        #           Child Loop BB0_16 Depth 5
                                        #             Child Loop BB0_18 Depth 6
	movq	%rdi, %rax
	shlq	$6, %rax
	cmpq	$4027, %rax                     # imm = 0xFBB
	ja	.LBB0_8
# %bb.7:                                # %cond.end109.i
                                        #   in Loop: Header=BB0_6 Depth=2
	movl	$4028, %ecx                     # imm = 0xFBC
	subl	%eax, %ecx
	movl	$4059, %edx                     # imm = 0xFDB
	subl	%eax, %edx
	testw	%cx, %cx
	cmovnsl	%ecx, %edx
	movswl	%dx, %ecx
	sarl	$5, %ecx
	negl	%ecx
	movswq	%cx, %rcx
	leaq	(%rdi,%rsi), %rbp
	cmpq	%rcx, %rbp
	cmovlq	%rcx, %rbp
	jmp	.LBB0_9
	.p2align	4, 0x90
.LBB0_8:                                # %cond.end109.thread.i
                                        #   in Loop: Header=BB0_6 Depth=2
	leaq	-3997(%rax), %rcx
	leaq	-3966(%rax), %rbp
	testq	%rcx, %rcx
	cmovnsq	%rcx, %rbp
	sarq	$5, %rbp
	leaq	(%rdi,%rsi), %rcx
	cmpq	%rcx, %rbp
	cmovleq	%rcx, %rbp
.LBB0_9:                                # %cond.end136.i
                                        #   in Loop: Header=BB0_6 Depth=2
	addq	$4059, %rax                     # imm = 0xFDB
	shrq	$5, %rax
	movq	160(%rsp), %rcx                 # 8-byte Reload
	cmpq	%rax, %rcx
	cmovbq	%rcx, %rax
	movq	%rdi, 192(%rsp)                 # 8-byte Spill
	shlq	$5, %rdi
	movq	152(%rsp), %rcx                 # 8-byte Reload
	leaq	(%rcx,%rdi), %rdx
	shrq	$5, %rdx
	leaq	5028(%rdi), %rcx
	shrq	$5, %rcx
	cmpq	%rcx, %rax
	movq	%rdx, %r8
	cmovaeq	%rcx, %r8
	cmpq	%rdx, %rax
	movq	%r8, %rsi
	cmovbq	%rax, %rsi
	cmpq	%rcx, %rax
	cmovaeq	%r8, %rsi
	cmpq	%rcx, %rdx
	cmovbq	%rdx, %rcx
	cmpq	%rdx, %rax
	cmovbq	%rsi, %rcx
	leaq	-3998(%rdi), %rdx
	movq	168(%rsp), %rax                 # 8-byte Reload
	cmpq	%rdx, %rax
	cmovgq	%rax, %rdx
	movq	%rdx, 232(%rsp)                 # 8-byte Spill
	movq	%rdi, 88(%rsp)                  # 8-byte Spill
	leaq	30(%rdi), %rdx
	movq	144(%rsp), %rax                 # 8-byte Reload
	cmpq	%rdx, %rax
	cmovbq	%rax, %rdx
	movq	%rdx, 224(%rsp)                 # 8-byte Spill
	cmpq	%rcx, %rbp
	jg	.LBB0_21
# %bb.10:                               # %for.body1117.i.preheader
                                        #   in Loop: Header=BB0_6 Depth=2
	movq	%rcx, %rdx
	movq	88(%rsp), %rax                  # 8-byte Reload
	movq	%rax, %rcx
	negq	%rcx
	movq	%rcx, 208(%rsp)                 # 8-byte Spill
	addq	$31, %rax
	movq	%rax, 264(%rsp)                 # 8-byte Spill
	cmpq	%rbp, %rdx
	cmovleq	%rbp, %rdx
	movq	%rdx, 216(%rsp)                 # 8-byte Spill
	movq	%rbp, %rcx
	shlq	$5, %rcx
	leaq	-3998(%rcx), %rax
	movq	%rax, 72(%rsp)                  # 8-byte Spill
	movq	56(%rsp), %rax                  # 8-byte Reload
	movq	%rcx, 112(%rsp)                 # 8-byte Spill
	leaq	(%rax,%rcx), %rdx
	movq	%rbp, %rsi
	shlq	$4, %rsi
	addq	$-3998, %rsi                    # imm = 0xF062
	jmp	.LBB0_11
	.p2align	4, 0x90
.LBB0_20:                               # %for.inc1549.i
                                        #   in Loop: Header=BB0_11 Depth=3
	movq	256(%rsp), %rcx                 # 8-byte Reload
	leaq	1(%rcx), %rbp
	addq	$32, 112(%rsp)                  # 8-byte Folded Spill
	addq	$32, 72(%rsp)                   # 8-byte Folded Spill
	movq	248(%rsp), %rdx                 # 8-byte Reload
	addq	$32, %rdx
	movq	240(%rsp), %rsi                 # 8-byte Reload
	addq	$16, %rsi
	cmpq	216(%rsp), %rcx                 # 8-byte Folded Reload
	movq	200(%rsp), %rbx                 # 8-byte Reload
	je	.LBB0_21
.LBB0_11:                               # %for.body1117.i
                                        #   Parent Loop BB0_5 Depth=1
                                        #     Parent Loop BB0_6 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_14 Depth 4
                                        #           Child Loop BB0_16 Depth 5
                                        #             Child Loop BB0_18 Depth 6
	movq	64(%rsp), %rax                  # 8-byte Reload
	cmpq	%rax, %rbx
	cmovgq	%rbx, %rax
	cmpq	%rdx, %rax
	movq	%rdx, 248(%rsp)                 # 8-byte Spill
	cmovleq	%rdx, %rax
	cmpq	%rsi, %rax
	movq	%rsi, 240(%rsp)                 # 8-byte Spill
	cmovleq	%rsi, %rax
	movq	%rax, %rdx
	movq	%rbp, %rax
	shlq	$4, %rax
	leaq	-3998(%rax), %rdi
	movq	232(%rsp), %rcx                 # 8-byte Reload
	cmpq	%rdi, %rcx
	cmovgq	%rcx, %rdi
	movq	%rbp, 256(%rsp)                 # 8-byte Spill
	shlq	$5, %rbp
	movq	208(%rsp), %rcx                 # 8-byte Reload
	leaq	(%rcx,%rbp), %rsi
	movq	%rbp, 16(%rsp)                  # 8-byte Spill
	addq	%rbp, %rcx
	addq	$-4029, %rcx                    # imm = 0xF043
	cmpq	%rcx, %rdi
	cmovleq	%rcx, %rdi
	orq	$14, %rax
	movq	224(%rsp), %rcx                 # 8-byte Reload
	cmpq	%rax, %rcx
	cmovlq	%rcx, %rax
	orq	$30, %rsi
	cmpq	%rsi, %rax
	cmovlq	%rax, %rsi
	movq	%rsi, 272(%rsp)                 # 8-byte Spill
	cmpq	%rsi, %rdi
	jg	.LBB0_20
# %bb.12:                               # %for.body1364.lr.ph.i
                                        #   in Loop: Header=BB0_11 Depth=3
	movq	72(%rsp), %rcx                  # 8-byte Reload
	subq	%rdx, %rcx
	leaq	1(%rdx), %rsi
	negq	%rdx
	movq	%rdx, 96(%rsp)                  # 8-byte Spill
	movq	16(%rsp), %rax                  # 8-byte Reload
	orq	$31, %rax
	movq	%rax, 296(%rsp)                 # 8-byte Spill
	jmp	.LBB0_14
	.p2align	4, 0x90
.LBB0_13:                               # %for.cond1207.loopexit.i
                                        #   in Loop: Header=BB0_14 Depth=4
	movq	280(%rsp), %rcx                 # 8-byte Reload
	addq	$-1, %rcx
	movq	104(%rsp), %rsi                 # 8-byte Reload
	addq	$1, %rsi
	addq	$-1, 96(%rsp)                   # 8-byte Folded Spill
	movq	24(%rsp), %rax                  # 8-byte Reload
	cmpq	272(%rsp), %rax                 # 8-byte Folded Reload
	movq	288(%rsp), %rdi                 # 8-byte Reload
	jge	.LBB0_20
.LBB0_14:                               # %for.body1364.i
                                        #   Parent Loop BB0_5 Depth=1
                                        #     Parent Loop BB0_6 Depth=2
                                        #       Parent Loop BB0_11 Depth=3
                                        # =>      This Loop Header: Depth=4
                                        #           Child Loop BB0_16 Depth 5
                                        #             Child Loop BB0_18 Depth 6
	movq	80(%rsp), %rax                  # 8-byte Reload
	cmpq	%rcx, %rax
	movq	%rcx, 280(%rsp)                 # 8-byte Spill
	movq	%rcx, %rdx
	cmovgq	%rax, %rdx
	cmpq	%rsi, %rdx
	movq	%rsi, 104(%rsp)                 # 8-byte Spill
	cmovleq	%rsi, %rdx
	leaq	1(%rdi), %rcx
	movq	88(%rsp), %rax                  # 8-byte Reload
	cmpq	%rcx, %rax
	movq	%rcx, 288(%rsp)                 # 8-byte Spill
	cmovgq	%rax, %rcx
	movq	16(%rsp), %rsi                  # 8-byte Reload
	subq	%rdi, %rsi
	leaq	-3998(%rsi), %rax
	cmpq	%rax, %rcx
	cmovleq	%rax, %rcx
	addq	$30, %rsi
	movq	264(%rsp), %rax                 # 8-byte Reload
	cmpq	%rsi, %rax
	cmovlq	%rax, %rsi
	movq	%rdi, 24(%rsp)                  # 8-byte Spill
	leaq	3998(%rdi), %rax
	cmpq	%rax, %rsi
	cmovgeq	%rax, %rsi
	movq	%rsi, 304(%rsp)                 # 8-byte Spill
	cmpq	%rsi, %rcx
	jg	.LBB0_13
# %bb.15:                               # %for.body1437.i.preheader
                                        #   in Loop: Header=BB0_14 Depth=4
	movq	96(%rsp), %rax                  # 8-byte Reload
	subq	%rdx, %rax
	movq	%rax, 120(%rsp)                 # 8-byte Spill
	addq	104(%rsp), %rdx                 # 8-byte Folded Reload
	jmp	.LBB0_16
	.p2align	4, 0x90
.LBB0_19:                               # %for.inc1543.i
                                        #   in Loop: Header=BB0_16 Depth=5
	movq	8(%rsp), %rcx                   # 8-byte Reload
	leaq	1(%rcx), %rax
	movq	32(%rsp), %rdx                  # 8-byte Reload
	addq	$1, %rdx
	addq	$-1, 120(%rsp)                  # 8-byte Folded Spill
	cmpq	304(%rsp), %rcx                 # 8-byte Folded Reload
	movq	%rax, %rcx
	jge	.LBB0_13
.LBB0_16:                               # %for.body1437.i
                                        #   Parent Loop BB0_5 Depth=1
                                        #     Parent Loop BB0_6 Depth=2
                                        #       Parent Loop BB0_11 Depth=3
                                        #         Parent Loop BB0_14 Depth=4
                                        # =>        This Loop Header: Depth=5
                                        #             Child Loop BB0_18 Depth 6
	movq	112(%rsp), %rax                 # 8-byte Reload
	cmpq	%rdx, %rax
	movq	%rdx, 32(%rsp)                  # 8-byte Spill
	movq	%rdx, %rdi
	cmovgq	%rax, %rdi
	movq	24(%rsp), %rdx                  # 8-byte Reload
	leaq	(%rcx,%rdx), %rsi
	addq	$1, %rsi
	movq	16(%rsp), %rax                  # 8-byte Reload
	cmpq	%rsi, %rax
	cmovgq	%rax, %rsi
	movq	%rcx, 8(%rsp)                   # 8-byte Spill
	leaq	(%rcx,%rdx), %r10
	addq	$3998, %r10                     # imm = 0xF9E
	movq	296(%rsp), %rax                 # 8-byte Reload
	cmpq	%r10, %rax
	cmovlq	%rax, %r10
	cmpq	%r10, %rsi
	movabsq	$4294967296, %r8                # imm = 0x100000000
	jg	.LBB0_19
# %bb.17:                               # %for.body1466.lr.ph.i
                                        #   in Loop: Header=BB0_16 Depth=5
	leaq	-1(%rdi), %rsi
	movq	120(%rsp), %rax                 # 8-byte Reload
	leal	(%rax,%rdi), %r14d
	shlq	$32, %r14
	movq	8(%rsp), %rax                   # 8-byte Reload
	subq	24(%rsp), %rax                  # 8-byte Folded Reload
	movslq	%eax, %rdi
	shlq	$32, %rax
	leaq	(%rax,%r11), %rcx
	sarq	$32, %rcx
	imulq	$4000, %rcx, %rcx               # imm = 0xFA0
	imulq	$4000, %rdi, %r9                # imm = 0xFA0
	addq	%r8, %rax
	sarq	$32, %rax
	imulq	$4000, %rax, %rdi               # imm = 0xFA0
	.p2align	4, 0x90
.LBB0_18:                               # %for.body1466.i
                                        #   Parent Loop BB0_5 Depth=1
                                        #     Parent Loop BB0_6 Depth=2
                                        #       Parent Loop BB0_11 Depth=3
                                        #         Parent Loop BB0_14 Depth=4
                                        #           Parent Loop BB0_16 Depth=5
                                        # =>          This Inner Loop Header: Depth=6
	leaq	(%r14,%r11), %rax
	sarq	$32, %rax
	leaq	(%rax,%rcx), %r15
	movq	%r14, %r13
	sarq	$32, %r13
	movsd	(%r12,%r15,8), %xmm1            # xmm1 = mem[0],zero
	leaq	(%rcx,%r13), %rbp
	addsd	(%r12,%rbp,8), %xmm1
	addq	%r8, %r14
	movq	%r14, %rbp
	sarq	$32, %rbp
	leaq	(%rcx,%rbp), %rdx
	addsd	(%r12,%rdx,8), %xmm1
	leaq	(%rax,%r9), %rdx
	addsd	(%r12,%rdx,8), %xmm1
	leaq	(%r13,%r9), %rdx
	addsd	(%r12,%rdx,8), %xmm1
	leaq	(%r9,%rbp), %rbx
	addsd	(%r12,%rbx,8), %xmm1
	addq	%rdi, %rax
	addsd	(%r12,%rax,8), %xmm1
	addq	%rdi, %r13
	addsd	(%r12,%r13,8), %xmm1
	addq	%rdi, %rbp
	addsd	(%r12,%rbp,8), %xmm1
	divsd	%xmm0, %xmm1
	movsd	%xmm1, (%r12,%rdx,8)
	addq	$1, %rsi
	cmpq	%r10, %rsi
	jl	.LBB0_18
	jmp	.LBB0_19
.LBB0_23:                               # %kernel_seidel_2d.exit
	xorl	%eax, %eax
	callq	polybench_timer_stop
	xorl	%eax, %eax
	callq	polybench_timer_print
	cmpl	$43, 44(%rsp)                   # 4-byte Folded Reload
	jl	.LBB0_32
# %bb.24:                               # %if.end131
	movq	128(%rsp), %rax                 # 8-byte Reload
	movq	(%rax), %rax
	cmpb	$0, (%rax)
	je	.LBB0_25
.LBB0_32:                               # %if.end146
	movq	%r12, %rdi
	callq	free
	xorl	%eax, %eax
	addq	$312, %rsp                      # imm = 0x138
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
.LBB0_25:                               # %if.then144
	.cfi_def_cfa_offset 368
	movq	stderr(%rip), %rcx
	movl	$.L.str.1, %edi
	movl	$22, %esi
	movl	$1, %edx
	callq	fwrite
	movq	stderr(%rip), %rdi
	xorl	%ebx, %ebx
	movl	$.L.str.2, %esi
	movl	$.L.str.3, %edx
	xorl	%eax, %eax
	callq	fprintf
	xorl	%r13d, %r13d
	movq	%r12, %rbp
	xorl	%eax, %eax
.LBB0_26:                               # %for.cond2.preheader.i
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_27 Depth 2
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	movq	%rbx, 8(%rsp)                   # 8-byte Spill
	movl	%ebx, %r14d
	xorl	%r15d, %r15d
	movl	$3435973837, %ebx               # imm = 0xCCCCCCCD
.LBB0_27:                               # %for.body4.i
                                        #   Parent Loop BB0_26 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movl	%r14d, %eax
	imulq	%rbx, %rax
	shrq	$36, %rax
	leal	(%rax,%rax,4), %eax
	leal	(%r13,%rax,4), %eax
	cmpl	%r15d, %eax
	jne	.LBB0_29
# %bb.28:                               # %if.then.i
                                        #   in Loop: Header=BB0_27 Depth=2
	movq	stderr(%rip), %rsi
	movl	$10, %edi
	callq	fputc
.LBB0_29:                               # %if.end.i
                                        #   in Loop: Header=BB0_27 Depth=2
	movq	stderr(%rip), %rdi
	movsd	(%rbp,%r15,8), %xmm0            # xmm0 = mem[0],zero
	movl	$.L.str.5, %esi
	movb	$1, %al
	callq	fprintf
	addq	$1, %r15
	addl	$1, %r14d
	cmpq	$4000, %r15                     # imm = 0xFA0
	jne	.LBB0_27
# %bb.30:                               # %for.inc10.i
                                        #   in Loop: Header=BB0_26 Depth=1
	movq	32(%rsp), %rax                  # 8-byte Reload
	addq	$1, %rax
	addq	$32000, %rbp                    # imm = 0x7D00
	addl	$-4000, %r13d                   # imm = 0xF060
	movq	8(%rsp), %rbx                   # 8-byte Reload
	addl	$4000, %ebx                     # imm = 0xFA0
	cmpq	$4000, %rax                     # imm = 0xFA0
	jne	.LBB0_26
# %bb.31:                               # %print_array.exit
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
	jmp	.LBB0_32
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
	.asciz	"A"
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

	.ident	"clang version 12.0.0 (git@github.com:wsmoses/MLIR-GPU 1112d5451cea635029a160c950f14a85f31b2258)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
