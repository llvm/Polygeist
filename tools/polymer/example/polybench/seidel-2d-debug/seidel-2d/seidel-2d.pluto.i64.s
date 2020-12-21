	.text
	.file	"seidel-2d.pluto.i64.c"
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
	subq	$376, %rsp                      # imm = 0x178
	.cfi_def_cfa_offset 432
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rsi, 136(%rsp)                 # 8-byte Spill
	movl	%edi, 28(%rsp)                  # 4-byte Spill
	movl	$128000000, %edi                # imm = 0x7A12000
	callq	malloc
	movq	%rax, 8(%rsp)                   # 8-byte Spill
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
	xorl	%r15d, %r15d
	xorl	%eax, %eax
	callq	polybench_timer_start
	movl	$126, %eax
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	movq	$-3998, %rax                    # imm = 0xF062
	movq	$-4029, %rcx                    # imm = 0xF043
	movsd	.LCPI0_2(%rip), %xmm0           # xmm0 = mem[0],zero
	xorl	%esi, %esi
	jmp	.LBB0_5
	.p2align	4, 0x90
.LBB0_22:                               # %for.inc1836.i
                                        #   in Loop: Header=BB0_5 Depth=1
	addq	$1, %rsi
	addq	$1, 32(%rsp)                    # 8-byte Folded Spill
	addq	$32, %r15
	movq	144(%rsp), %rax                 # 8-byte Reload
	addq	$32, %rax
	addq	$-32, %rcx
	cmpq	$32, %rsi
	je	.LBB0_23
.LBB0_5:                                # %for.body105.lr.ph.i
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_6 Depth 2
                                        #       Child Loop BB0_11 Depth 3
                                        #         Child Loop BB0_14 Depth 4
                                        #           Child Loop BB0_16 Depth 5
                                        #             Child Loop BB0_18 Depth 6
	movq	%rcx, 176(%rsp)                 # 8-byte Spill
	movq	%rsi, %rdi
	shlq	$5, %rdi
	leaq	4029(%rdi), %rbp
	shrq	$4, %rbp
	cmpq	$312, %rbp                      # imm = 0x138
	movl	$312, %edx                      # imm = 0x138
	cmovaeq	%rdx, %rbp
	movq	%rbp, 168(%rsp)                 # 8-byte Spill
	leaq	4060(%rdi), %rdx
	movq	%rdx, 160(%rsp)                 # 8-byte Spill
	movq	%rdi, 56(%rsp)                  # 8-byte Spill
	addq	$31, %rdi
	cmpq	$999, %rdi                      # imm = 0x3E7
	movl	$999, %edx                      # imm = 0x3E7
	cmovaeq	%rdx, %rdi
	movq	%rdi, 152(%rsp)                 # 8-byte Spill
	movq	%rcx, 48(%rsp)                  # 8-byte Spill
	movq	%rax, 144(%rsp)                 # 8-byte Spill
	movq	%rax, 72(%rsp)                  # 8-byte Spill
	movq	%r15, 88(%rsp)                  # 8-byte Spill
	xorl	%eax, %eax
	movq	%rax, 40(%rsp)                  # 8-byte Spill
	movq	%rsi, %rdi
	movq	%rsi, 64(%rsp)                  # 8-byte Spill
	movq	%r15, 192(%rsp)                 # 8-byte Spill
	jmp	.LBB0_6
	.p2align	4, 0x90
.LBB0_21:                               # %for.inc1833.i
                                        #   in Loop: Header=BB0_6 Depth=2
	movq	184(%rsp), %rdi                 # 8-byte Reload
	addq	$1, %rdi
	addq	$1, 40(%rsp)                    # 8-byte Folded Spill
	addq	$32, 88(%rsp)                   # 8-byte Folded Spill
	addq	$32, 72(%rsp)                   # 8-byte Folded Spill
	addq	$-32, 48(%rsp)                  # 8-byte Folded Spill
	cmpq	32(%rsp), %rdi                  # 8-byte Folded Reload
	movq	176(%rsp), %rcx                 # 8-byte Reload
	movq	64(%rsp), %rsi                  # 8-byte Reload
	je	.LBB0_22
.LBB0_6:                                # %for.body105.i
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
# %bb.7:                                # %cond.end128.i
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
.LBB0_8:                                # %cond.end128.thread.i
                                        #   in Loop: Header=BB0_6 Depth=2
	leaq	-3997(%rax), %rcx
	leaq	-3966(%rax), %rbp
	testq	%rcx, %rcx
	cmovnsq	%rcx, %rbp
	sarq	$5, %rbp
	leaq	(%rdi,%rsi), %rcx
	cmpq	%rcx, %rbp
	cmovleq	%rcx, %rbp
.LBB0_9:                                # %cond.end160.i
                                        #   in Loop: Header=BB0_6 Depth=2
	addq	$4059, %rax                     # imm = 0xFDB
	shrq	$5, %rax
	movq	168(%rsp), %rcx                 # 8-byte Reload
	cmpq	%rax, %rcx
	cmovbq	%rcx, %rax
	movq	%rdi, 184(%rsp)                 # 8-byte Spill
	shlq	$5, %rdi
	movq	160(%rsp), %rcx                 # 8-byte Reload
	addq	%rdi, %rcx
	shrq	$5, %rcx
	leaq	5028(%rdi), %rdx
	shrq	$5, %rdx
	cmpq	%rdx, %rax
	movq	%rcx, %r8
	cmovaeq	%rdx, %r8
	cmpq	%rcx, %rax
	movq	%r8, %rsi
	cmovbq	%rax, %rsi
	cmpq	%rdx, %rax
	cmovaeq	%r8, %rsi
	cmpq	%rdx, %rcx
	cmovbq	%rcx, %rdx
	cmpq	%rcx, %rax
	cmovbq	%rsi, %rdx
	leaq	-3998(%rdi), %rcx
	movq	56(%rsp), %rax                  # 8-byte Reload
	cmpq	%rcx, %rax
	cmovgq	%rax, %rcx
	movq	%rcx, 256(%rsp)                 # 8-byte Spill
	movq	%rdi, 96(%rsp)                  # 8-byte Spill
	leaq	30(%rdi), %rcx
	movq	152(%rsp), %rax                 # 8-byte Reload
	cmpq	%rcx, %rax
	cmovbq	%rax, %rcx
	movq	%rcx, 248(%rsp)                 # 8-byte Spill
	cmpq	%rdx, %rbp
	jg	.LBB0_21
# %bb.10:                               # %for.body1315.i.preheader
                                        #   in Loop: Header=BB0_6 Depth=2
	movq	%rdx, %rsi
	movq	64(%rsp), %rax                  # 8-byte Reload
	movq	40(%rsp), %rcx                  # 8-byte Reload
	addq	%rax, %rcx
	shlq	$5, %rcx
	leaq	-3998(%rcx), %rdx
	movq	96(%rsp), %rax                  # 8-byte Reload
	movq	%rax, %rdi
	negq	%rdi
	movq	%rdi, 216(%rsp)                 # 8-byte Spill
	addq	$31, %rax
	movq	%rax, 304(%rsp)                 # 8-byte Spill
	cmpq	%rbp, %rsi
	cmovleq	%rbp, %rsi
	movq	%rsi, 232(%rsp)                 # 8-byte Spill
	movq	%rbp, %rsi
	shlq	$4, %rsi
	addq	$-3998, %rsi                    # imm = 0xF062
	movq	56(%rsp), %rax                  # 8-byte Reload
	cmpq	%rdx, %rax
	cmovgq	%rax, %rdx
	movq	%rdx, 224(%rsp)                 # 8-byte Spill
	movq	%rcx, 312(%rsp)                 # 8-byte Spill
	negq	%rcx
	movq	%rcx, 200(%rsp)                 # 8-byte Spill
	movq	%rbp, %r14
	shlq	$5, %r14
	leaq	-3998(%r14), %rax
	movq	%rax, 80(%rsp)                  # 8-byte Spill
	movq	48(%rsp), %rax                  # 8-byte Reload
	leaq	(%rax,%r14), %rcx
	movq	%rsi, 208(%rsp)                 # 8-byte Spill
	xorl	%edx, %edx
	movq	%rbp, %r8
	movq	%rbp, 240(%rsp)                 # 8-byte Spill
	jmp	.LBB0_11
	.p2align	4, 0x90
.LBB0_20:                               # %for.inc1830.i
                                        #   in Loop: Header=BB0_11 Depth=3
	movq	264(%rsp), %rdi                 # 8-byte Reload
	leaq	1(%rdi), %r8
	movq	272(%rsp), %rdx                 # 8-byte Reload
	addq	$1, %rdx
	addq	$32, %r14
	addq	$32, 80(%rsp)                   # 8-byte Folded Spill
	movq	288(%rsp), %rcx                 # 8-byte Reload
	addq	$32, %rcx
	movq	280(%rsp), %rsi                 # 8-byte Reload
	addq	$16, %rsi
	cmpq	232(%rsp), %rdi                 # 8-byte Folded Reload
	movq	192(%rsp), %r15                 # 8-byte Reload
	movq	240(%rsp), %rbp                 # 8-byte Reload
	je	.LBB0_21
.LBB0_11:                               # %for.body1315.i
                                        #   Parent Loop BB0_5 Depth=1
                                        #     Parent Loop BB0_6 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_14 Depth 4
                                        #           Child Loop BB0_16 Depth 5
                                        #             Child Loop BB0_18 Depth 6
	movq	72(%rsp), %rax                  # 8-byte Reload
	cmpq	%rax, %r15
	movq	%rax, %rbx
	cmovgq	%r15, %rbx
	cmpq	%rcx, %rbx
	movq	%rcx, 288(%rsp)                 # 8-byte Spill
	cmovleq	%rcx, %rbx
	cmpq	%rsi, %rbx
	movq	%rsi, 280(%rsp)                 # 8-byte Spill
	cmovleq	%rsi, %rbx
	leaq	(%rdx,%rbp), %rcx
	shlq	$5, %rcx
	movq	200(%rsp), %rax                 # 8-byte Reload
	movq	%rcx, 128(%rsp)                 # 8-byte Spill
	addq	%rcx, %rax
	addq	$-4029, %rax                    # imm = 0xF043
	movq	224(%rsp), %rcx                 # 8-byte Reload
	cmpq	%rax, %rcx
	cmovgq	%rcx, %rax
	movq	%rdx, 272(%rsp)                 # 8-byte Spill
	shlq	$4, %rdx
	addq	208(%rsp), %rdx                 # 8-byte Folded Reload
	cmpq	%rdx, %rax
	cmovgq	%rax, %rdx
	movq	%rdx, 16(%rsp)                  # 8-byte Spill
	movq	%r8, %rax
	shlq	$4, %rax
	leaq	-3998(%rax), %r11
	movq	256(%rsp), %rcx                 # 8-byte Reload
	cmpq	%r11, %rcx
	cmovgq	%rcx, %r11
	movq	%r8, 264(%rsp)                  # 8-byte Spill
	shlq	$5, %r8
	movq	216(%rsp), %rcx                 # 8-byte Reload
	leaq	(%r8,%rcx), %rdx
	addq	$-4029, %rdx                    # imm = 0xF043
	cmpq	%rdx, %r11
	cmovleq	%rdx, %r11
	leaq	(%r8,%rcx), %rdx
	orq	$14, %rax
	movq	248(%rsp), %rcx                 # 8-byte Reload
	cmpq	%rax, %rcx
	cmovlq	%rcx, %rax
	orq	$30, %rdx
	cmpq	%rdx, %rax
	cmovlq	%rax, %rdx
	movq	%rdx, 328(%rsp)                 # 8-byte Spill
	cmpq	%rdx, %r11
	jg	.LBB0_20
# %bb.12:                               # %for.body1615.lr.ph.i
                                        #   in Loop: Header=BB0_11 Depth=3
	movq	80(%rsp), %rdx                  # 8-byte Reload
	subq	%rbx, %rdx
	imulq	$-32008, %rbx, %rax             # imm = 0x82F8
	addq	$1, %rbx
	addq	8(%rsp), %rax                   # 8-byte Folded Reload
	movq	%rax, 104(%rsp)                 # 8-byte Spill
	movq	16(%rsp), %rax                  # 8-byte Reload
	movq	%rax, %rcx
	notq	%rcx
	movq	%rcx, 296(%rsp)                 # 8-byte Spill
	movq	%r8, %r10
	orq	$31, %r10
	movq	128(%rsp), %rcx                 # 8-byte Reload
	subq	%rax, %rcx
	addq	$-3998, %rcx                    # imm = 0xF062
	movq	%rcx, 320(%rsp)                 # 8-byte Spill
	xorl	%ecx, %ecx
	jmp	.LBB0_14
	.p2align	4, 0x90
.LBB0_13:                               # %for.cond1426.loopexit.i
                                        #   in Loop: Header=BB0_14 Depth=4
	movq	112(%rsp), %rcx                 # 8-byte Reload
	addq	$1, %rcx
	movq	336(%rsp), %rdx                 # 8-byte Reload
	addq	$-1, %rdx
	movq	120(%rsp), %rbx                 # 8-byte Reload
	addq	$1, %rbx
	addq	$-32008, 104(%rsp)              # 8-byte Folded Spill
                                        # imm = 0x82F8
	cmpq	328(%rsp), %r11                 # 8-byte Folded Reload
	movq	344(%rsp), %r11                 # 8-byte Reload
	jge	.LBB0_20
.LBB0_14:                               # %for.body1615.i
                                        #   Parent Loop BB0_5 Depth=1
                                        #     Parent Loop BB0_6 Depth=2
                                        #       Parent Loop BB0_11 Depth=3
                                        # =>      This Loop Header: Depth=4
                                        #           Child Loop BB0_16 Depth 5
                                        #             Child Loop BB0_18 Depth 6
	movq	88(%rsp), %rax                  # 8-byte Reload
	cmpq	%rdx, %rax
	movq	%rdx, 336(%rsp)                 # 8-byte Spill
	cmovgq	%rax, %rdx
	cmpq	%rbx, %rdx
	movq	%rbx, 120(%rsp)                 # 8-byte Spill
	cmovleq	%rbx, %rdx
	movq	320(%rsp), %rsi                 # 8-byte Reload
	subq	%rcx, %rsi
	movq	312(%rsp), %rax                 # 8-byte Reload
	cmpq	%rsi, %rax
	cmovgq	%rax, %rsi
	movq	16(%rsp), %rax                  # 8-byte Reload
	movq	%rcx, 112(%rsp)                 # 8-byte Spill
	addq	%rcx, %rax
	addq	$1, %rax
	cmpq	%rax, %rsi
	cmovleq	%rax, %rsi
	leaq	1(%r11), %rbp
	movq	96(%rsp), %rax                  # 8-byte Reload
	cmpq	%rbp, %rax
	movq	%rbp, 344(%rsp)                 # 8-byte Spill
	cmovgq	%rax, %rbp
	movq	%r8, %r13
	subq	%r11, %r13
	leaq	-3998(%r13), %rax
	cmpq	%rax, %rbp
	cmovleq	%rax, %rbp
	addq	$30, %r13
	movq	304(%rsp), %rax                 # 8-byte Reload
	cmpq	%r13, %rax
	cmovlq	%rax, %r13
	leaq	3998(%r11), %rax
	cmpq	%rax, %r13
	cmovgeq	%rax, %r13
	cmpq	%r13, %rbp
	jg	.LBB0_13
# %bb.15:                               # %for.body1699.i.preheader
                                        #   in Loop: Header=BB0_14 Depth=4
	imulq	$31992, %rdx, %r12              # imm = 0x7CF8
	addq	120(%rsp), %rdx                 # 8-byte Folded Reload
	addq	104(%rsp), %r12                 # 8-byte Folded Reload
	movq	16(%rsp), %rax                  # 8-byte Reload
	movq	112(%rsp), %rcx                 # 8-byte Reload
	addq	%rcx, %rax
	movq	%rsi, %rdi
	subq	%rax, %rdi
	movq	%rdi, 368(%rsp)                 # 8-byte Spill
	addq	%rsi, %rax
	addq	$1, %rax
	movq	%rax, 360(%rsp)                 # 8-byte Spill
	movq	296(%rsp), %rax                 # 8-byte Reload
	subq	%rcx, %rax
	subq	%rsi, %rax
	movq	%rax, 352(%rsp)                 # 8-byte Spill
	xorl	%esi, %esi
	jmp	.LBB0_16
	.p2align	4, 0x90
.LBB0_19:                               # %for.inc1824.i
                                        #   in Loop: Header=BB0_16 Depth=5
	leaq	1(%rbp), %rax
	addq	$1, %rsi
	addq	$1, %rdx
	addq	$31992, %r12                    # imm = 0x7CF8
	cmpq	%r13, %rbp
	movq	%rax, %rbp
	jge	.LBB0_13
.LBB0_16:                               # %for.body1699.i
                                        #   Parent Loop BB0_5 Depth=1
                                        #     Parent Loop BB0_6 Depth=2
                                        #       Parent Loop BB0_11 Depth=3
                                        #         Parent Loop BB0_14 Depth=4
                                        # =>        This Loop Header: Depth=5
                                        #             Child Loop BB0_18 Depth 6
	cmpq	%rdx, %r14
	movq	%rdx, %r15
	cmovgq	%r14, %r15
	leaq	(%r11,%rbp), %rcx
	addq	$1, %rcx
	cmpq	%rcx, %r8
	cmovgq	%r8, %rcx
	leaq	(%r11,%rbp), %rdi
	addq	$3998, %rdi                     # imm = 0xF9E
	cmpq	%rdi, %r10
	cmovlq	%r10, %rdi
	cmpq	%rdi, %rcx
	jg	.LBB0_19
# %bb.17:                               # %for.body1733.lr.ph.i
                                        #   in Loop: Header=BB0_16 Depth=5
	addq	$-1, %r15
	movq	360(%rsp), %rax                 # 8-byte Reload
	leaq	(%rax,%rsi), %rcx
	movq	368(%rsp), %rax                 # 8-byte Reload
	leaq	(%rax,%rsi), %r9
	imulq	$32000, %r9, %rbx               # imm = 0x7D00
	addq	8(%rsp), %rbx                   # 8-byte Folded Reload
	movq	128(%rsp), %rax                 # 8-byte Reload
	cmpq	%rcx, %rax
	cmovgq	%rax, %rcx
	movq	352(%rsp), %rax                 # 8-byte Reload
	subq	%rsi, %rax
	addq	%rcx, %rax
	movsd	(%rbx,%rax,8), %xmm1            # xmm1 = mem[0],zero
	.p2align	4, 0x90
.LBB0_18:                               # %for.body1733.i
                                        #   Parent Loop BB0_5 Depth=1
                                        #     Parent Loop BB0_6 Depth=2
                                        #       Parent Loop BB0_11 Depth=3
                                        #         Parent Loop BB0_14 Depth=4
                                        #           Parent Loop BB0_16 Depth=5
                                        # =>          This Inner Loop Header: Depth=6
	movsd	-32000(%r12,%r15,8), %xmm2      # xmm2 = mem[0],zero
	addsd	-31992(%r12,%r15,8), %xmm2
	addsd	-31984(%r12,%r15,8), %xmm2
	addsd	%xmm1, %xmm2
	addsd	8(%r12,%r15,8), %xmm2
	addsd	16(%r12,%r15,8), %xmm2
	addsd	32000(%r12,%r15,8), %xmm2
	addsd	32008(%r12,%r15,8), %xmm2
	addsd	32016(%r12,%r15,8), %xmm2
	divsd	%xmm0, %xmm2
	movsd	%xmm2, 8(%r12,%r15,8)
	addq	$1, %r15
	movapd	%xmm2, %xmm1
	cmpq	%rdi, %r15
	jl	.LBB0_18
	jmp	.LBB0_19
.LBB0_23:                               # %kernel_seidel_2d.exit
	xorl	%eax, %eax
	callq	polybench_timer_stop
	xorl	%eax, %eax
	callq	polybench_timer_print
	cmpl	$43, 28(%rsp)                   # 4-byte Folded Reload
	jl	.LBB0_32
# %bb.24:                               # %if.end123
	movq	136(%rsp), %rax                 # 8-byte Reload
	movq	(%rax), %rax
	cmpb	$0, (%rax)
	je	.LBB0_25
.LBB0_32:                               # %if.end138
	movq	8(%rsp), %rdi                   # 8-byte Reload
	callq	free
	xorl	%eax, %eax
	addq	$376, %rsp                      # imm = 0x178
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
.LBB0_25:                               # %if.then136
	.cfi_def_cfa_offset 432
	movq	stderr(%rip), %rcx
	movl	$.L.str.1, %edi
	movl	$22, %esi
	movl	$1, %edx
	callq	fwrite
	movq	stderr(%rip), %rdi
	xorl	%r15d, %r15d
	movl	$.L.str.2, %esi
	movl	$.L.str.3, %edx
	xorl	%eax, %eax
	callq	fprintf
	xorl	%r13d, %r13d
	movq	8(%rsp), %rbx                   # 8-byte Reload
	xorl	%r12d, %r12d
.LBB0_26:                               # %for.cond2.preheader.i
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_27 Depth 2
	movl	%r15d, %ebp
	xorl	%r14d, %r14d
.LBB0_27:                               # %for.body4.i
                                        #   Parent Loop BB0_26 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movl	%ebp, %eax
	movl	$3435973837, %ecx               # imm = 0xCCCCCCCD
	imulq	%rcx, %rax
	shrq	$36, %rax
	leal	(%rax,%rax,4), %eax
	leal	(%r13,%rax,4), %eax
	cmpl	%r14d, %eax
	jne	.LBB0_29
# %bb.28:                               # %if.then.i
                                        #   in Loop: Header=BB0_27 Depth=2
	movq	stderr(%rip), %rsi
	movl	$10, %edi
	callq	fputc
.LBB0_29:                               # %if.end.i
                                        #   in Loop: Header=BB0_27 Depth=2
	movq	stderr(%rip), %rdi
	movsd	(%rbx,%r14,8), %xmm0            # xmm0 = mem[0],zero
	movl	$.L.str.5, %esi
	movb	$1, %al
	callq	fprintf
	addq	$1, %r14
	addl	$1, %ebp
	cmpq	$4000, %r14                     # imm = 0xFA0
	jne	.LBB0_27
# %bb.30:                               # %for.inc10.i
                                        #   in Loop: Header=BB0_26 Depth=1
	addq	$1, %r12
	addq	$32000, %rbx                    # imm = 0x7D00
	addl	$-4000, %r13d                   # imm = 0xF060
	addl	$4000, %r15d                    # imm = 0xFA0
	cmpq	$4000, %r12                     # imm = 0xFA0
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
