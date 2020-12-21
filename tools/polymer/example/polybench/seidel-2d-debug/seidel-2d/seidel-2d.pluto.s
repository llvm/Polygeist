	.text
	.file	"seidel-2d.pluto.c"
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
	subq	$328, %rsp                      # imm = 0x148
	.cfi_def_cfa_offset 384
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rsi, 168(%rsp)                 # 8-byte Spill
	movl	%edi, 56(%rsp)                  # 4-byte Spill
	movl	$128000000, %edi                # imm = 0x7A12000
	callq	malloc
	leaq	8(%rax), %rbp
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
	movsd	%xmm3, -8(%rbp,%rdx,8)
	leal	3(%rdx), %edi
	xorps	%xmm3, %xmm3
	cvtsi2sd	%edi, %xmm3
	mulsd	%xmm2, %xmm3
	addsd	%xmm0, %xmm3
	divsd	%xmm1, %xmm3
	movsd	%xmm3, (%rbp,%rdx,8)
	movq	%rsi, %rdx
	cmpq	$4000, %rsi                     # imm = 0xFA0
	jne	.LBB0_2
# %bb.3:                                # %for.inc9.i
                                        #   in Loop: Header=BB0_1 Depth=1
	addq	$1, %rcx
	addq	$32000, %rbp                    # imm = 0x7D00
	cmpq	$4000, %rcx                     # imm = 0xFA0
	jne	.LBB0_1
# %bb.4:                                # %init_array.exit
	movq	%rax, 40(%rsp)                  # 8-byte Spill
	xorl	%r14d, %r14d
	xorl	%eax, %eax
	callq	polybench_timer_start
	movl	$-3998, %r8d                    # imm = 0xF062
	movl	$-4029, %ecx                    # imm = 0xF043
	movl	$126, %edx
	movq	%rdx, 120(%rsp)                 # 8-byte Spill
	movl	$30, %edx
	movl	$31, %esi
	movq	%rsi, 112(%rsp)                 # 8-byte Spill
	movl	$30, %esi
	movsd	.LCPI0_2(%rip), %xmm0           # xmm0 = mem[0],zero
	xorl	%r10d, %r10d
	jmp	.LBB0_5
	.p2align	4, 0x90
.LBB0_22:                               # %for.inc1588.i
                                        #   in Loop: Header=BB0_5 Depth=1
	addq	$1, %r14
	addl	$32, %r10d
	movl	60(%rsp), %r8d                  # 4-byte Reload
	addl	$32, %r8d
	addl	$-32, %ecx
	addq	$1, 120(%rsp)                   # 8-byte Folded Spill
	addl	$-32, %edx
	addq	$32, 112(%rsp)                  # 8-byte Folded Spill
	addq	$32, %rsi
	cmpq	$32, %r14
	je	.LBB0_23
.LBB0_5:                                # %for.body90.lr.ph.i
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_6 Depth 2
                                        #       Child Loop BB0_11 Depth 3
                                        #         Child Loop BB0_14 Depth 4
                                        #           Child Loop BB0_16 Depth 5
                                        #             Child Loop BB0_18 Depth 6
	movq	%rsi, 208(%rsp)                 # 8-byte Spill
	movl	%edx, 68(%rsp)                  # 4-byte Spill
	movl	%ecx, 72(%rsp)                  # 4-byte Spill
	movq	%r14, %rax
	shlq	$5, %rax
	leaq	4029(%rax), %r9
	shrq	$4, %r9
	cmpq	$963, %rax                      # imm = 0x3C3
	movl	$312, %edi                      # imm = 0x138
	cmovael	%edi, %r9d
	movq	%r9, 192(%rsp)                  # 8-byte Spill
	leaq	4060(%rax), %rdi
	movq	%rdi, 184(%rsp)                 # 8-byte Spill
	movq	%rax, 200(%rsp)                 # 8-byte Spill
	addq	$31, %rax
	cmpq	$999, %rax                      # imm = 0x3E7
	movl	$999, %edi                      # imm = 0x3E7
	cmovaeq	%rdi, %rax
	movq	%rax, 176(%rsp)                 # 8-byte Spill
	movq	%rsi, %rdi
	movl	%edx, %eax
	movq	%rax, 16(%rsp)                  # 8-byte Spill
	movq	%r14, %rbp
	movl	%ecx, %r11d
	movl	%r8d, 60(%rsp)                  # 4-byte Spill
	movl	%r8d, %r15d
	movl	%r10d, (%rsp)                   # 4-byte Spill
	movq	%r14, 216(%rsp)                 # 8-byte Spill
	movl	%r10d, 64(%rsp)                 # 4-byte Spill
	jmp	.LBB0_6
	.p2align	4, 0x90
.LBB0_21:                               # %for.inc1585.i
                                        #   in Loop: Header=BB0_6 Depth=2
	movq	128(%rsp), %rbp                 # 8-byte Reload
	addq	$1, %rbp
	addl	$32, (%rsp)                     # 4-byte Folded Spill
	movl	76(%rsp), %r15d                 # 4-byte Reload
	addl	$32, %r15d
	movq	224(%rsp), %r11                 # 8-byte Reload
	addl	$-32, %r11d
	movq	16(%rsp), %rax                  # 8-byte Reload
	addl	$-32, %eax
	movq	%rax, 16(%rsp)                  # 8-byte Spill
	movq	232(%rsp), %rdi                 # 8-byte Reload
	addq	$32, %rdi
	cmpq	120(%rsp), %rbp                 # 8-byte Folded Reload
	movq	216(%rsp), %r14                 # 8-byte Reload
	movl	72(%rsp), %ecx                  # 4-byte Reload
	movl	68(%rsp), %edx                  # 4-byte Reload
	movq	208(%rsp), %rsi                 # 8-byte Reload
	movl	64(%rsp), %r10d                 # 4-byte Reload
	je	.LBB0_22
.LBB0_6:                                # %for.body90.i
                                        #   Parent Loop BB0_5 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_11 Depth 3
                                        #         Child Loop BB0_14 Depth 4
                                        #           Child Loop BB0_16 Depth 5
                                        #             Child Loop BB0_18 Depth 6
	movq	112(%rsp), %rax                 # 8-byte Reload
	cmpq	%rdi, %rax
	movq	%rdi, %rcx
	cmovbq	%rax, %rcx
	cmpq	$999, %rcx                      # imm = 0x3E7
	movl	$999, %eax                      # imm = 0x3E7
	cmovael	%eax, %ecx
	movq	%rcx, 272(%rsp)                 # 8-byte Spill
	movq	%rbp, %rax
	shlq	$6, %rax
	cmpq	$4027, %rax                     # imm = 0xFBB
	movq	%rdi, 232(%rsp)                 # 8-byte Spill
	ja	.LBB0_8
# %bb.7:                                # %cond.end109.i
                                        #   in Loop: Header=BB0_6 Depth=2
	movl	$4028, %ecx                     # imm = 0xFBC
	subl	%eax, %ecx
	movl	$4059, %edx                     # imm = 0xFDB
	subl	%eax, %edx
	testw	%cx, %cx
	cmovnsl	%ecx, %edx
	movswl	%dx, %edx
	sarl	$5, %edx
	negl	%edx
	movswl	%dx, %r9d
	leaq	(%r14,%rbp), %r12
	movswq	%dx, %rdx
	jmp	.LBB0_9
	.p2align	4, 0x90
.LBB0_8:                                # %cond.end109.thread.i
                                        #   in Loop: Header=BB0_6 Depth=2
	leal	-3997(%rax), %edx
	leal	-3966(%rax), %r9d
	testl	%edx, %edx
	cmovnsl	%edx, %r9d
	sarl	$5, %r9d
	leaq	(%r14,%rbp), %r12
	movslq	%r9d, %rdx
.LBB0_9:                                # %cond.end136.i
                                        #   in Loop: Header=BB0_6 Depth=2
	cmpq	%rdx, %r12
	setl	%r8b
	addq	$4059, %rax                     # imm = 0xFDB
	shrq	$5, %rax
	movq	192(%rsp), %rcx                 # 8-byte Reload
	cmpq	%rcx, %rax
	cmoval	%ecx, %eax
	movq	%rbp, %r14
	shlq	$5, %r14
	movq	%rbp, 128(%rsp)                 # 8-byte Spill
	movl	%ebp, %edx
	shll	$5, %edx
	movq	184(%rsp), %rcx                 # 8-byte Reload
	leaq	(%r14,%rcx), %rdi
	addl	%ecx, %edx
	shrq	$5, %rdi
	shrl	$5, %edx
	leaq	5028(%r14), %rbp
	shrq	$5, %rbp
	cmpq	%rax, %rbp
	movl	%edx, %esi
	cmovbel	%ebp, %esi
	cmpq	%rax, %rdi
	movl	%eax, %ecx
	cmovbel	%esi, %ecx
	cmpq	%rax, %rbp
	cmovbel	%esi, %ecx
	cmpq	%rbp, %rdi
	cmovael	%ebp, %edx
	cmpq	%rax, %rdi
	cmoval	%ecx, %edx
	testb	%r8b, %r8b
	cmovnel	%r9d, %r12d
	leaq	-3998(%r14), %rcx
	movq	200(%rsp), %rax                 # 8-byte Reload
	cmpq	%rcx, %rax
	cmovgl	%eax, %ecx
	movq	%rcx, 264(%rsp)                 # 8-byte Spill
	movq	%r14, 136(%rsp)                 # 8-byte Spill
	leaq	30(%r14), %rcx
	movq	176(%rsp), %rax                 # 8-byte Reload
	cmpq	%rcx, %rax
	cmovbl	%eax, %ecx
	movq	%rcx, 256(%rsp)                 # 8-byte Spill
	movl	%r12d, %eax
	shll	$5, %eax
	movq	%r11, 224(%rsp)                 # 8-byte Spill
	movq	%rax, 8(%rsp)                   # 8-byte Spill
	addl	%eax, %r11d
	movl	%r12d, %edi
	shll	$4, %edi
	leal	-3998(%rdi), %ebp
	cmpl	%r15d, %r10d
	movl	%r15d, 76(%rsp)                 # 4-byte Spill
	cmovgl	%r10d, %r15d
	movslq	%edx, %rax
	cmpl	%r11d, %r15d
	movl	%r11d, %r8d
	movl	%r15d, 80(%rsp)                 # 4-byte Spill
	cmovgl	%r15d, %r8d
	cmpl	%ebp, %r8d
	cmovlel	%ebp, %r8d
	cmpq	%rax, %r12
	jg	.LBB0_21
# %bb.10:                               # %for.body1117.i.preheader
                                        #   in Loop: Header=BB0_6 Depth=2
	movq	%rax, %rdx
	movq	136(%rsp), %rax                 # 8-byte Reload
	addq	$31, %rax
	movq	%rax, 296(%rsp)                 # 8-byte Spill
	movq	8(%rsp), %rcx                   # 8-byte Reload
	leal	-3998(%rcx), %esi
	movq	128(%rsp), %rax                 # 8-byte Reload
                                        # kill: def $eax killed $eax killed $rax def $rax
	shll	$5, %eax
	negl	%eax
	movq	%rax, 240(%rsp)                 # 8-byte Spill
	movl	%esi, %ebx
	subl	%r8d, %ebx
	movq	16(%rsp), %rax                  # 8-byte Reload
	addl	%eax, %ecx
	orl	$14, %edi
	cmpq	%r12, %rdx
	cmovleq	%r12, %rdx
	movq	%rdx, 248(%rsp)                 # 8-byte Spill
	jmp	.LBB0_11
	.p2align	4, 0x90
.LBB0_20:                               # %for.inc1582.i
                                        #   in Loop: Header=BB0_11 Depth=3
	movq	8(%rsp), %rax                   # 8-byte Reload
	addl	$32, %eax
	movq	%rax, 8(%rsp)                   # 8-byte Spill
	movl	88(%rsp), %esi                  # 4-byte Reload
	addl	$32, %esi
	movl	96(%rsp), %r11d                 # 4-byte Reload
	addl	$32, %r11d
	movl	92(%rsp), %ebp                  # 4-byte Reload
	addl	$16, %ebp
	movl	80(%rsp), %eax                  # 4-byte Reload
	cmpl	%r11d, %eax
	movl	%r11d, %edx
	cmovgl	%eax, %edx
	movq	288(%rsp), %rax                 # 8-byte Reload
	leaq	1(%rax), %r12
	cmpl	%ebp, %edx
	cmovlel	%ebp, %edx
	movl	%esi, %ebx
	movq	%rdx, %r8
	subl	%edx, %ebx
	movl	84(%rsp), %ecx                  # 4-byte Reload
	addl	$32, %ecx
	movq	280(%rsp), %rdi                 # 8-byte Reload
	addl	$16, %edi
	cmpq	248(%rsp), %rax                 # 8-byte Folded Reload
	je	.LBB0_21
.LBB0_11:                               # %for.body1117.i
                                        #   Parent Loop BB0_5 Depth=1
                                        #     Parent Loop BB0_6 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_14 Depth 4
                                        #           Child Loop BB0_16 Depth 5
                                        #             Child Loop BB0_18 Depth 6
	movl	%esi, 88(%rsp)                  # 4-byte Spill
	movl	%ebp, 92(%rsp)                  # 4-byte Spill
	movl	%r11d, 96(%rsp)                 # 4-byte Spill
	cmpl	%edi, %ecx
	movq	%rdi, 280(%rsp)                 # 8-byte Spill
	movl	%edi, %eax
	movl	%ecx, 84(%rsp)                  # 4-byte Spill
	cmovll	%ecx, %eax
	movq	272(%rsp), %rcx                 # 8-byte Reload
	cmpl	%ecx, %eax
	cmovgel	%ecx, %eax
	movl	%r12d, %ecx
	shll	$4, %ecx
	leal	-3998(%rcx), %edx
	movq	264(%rsp), %rsi                 # 8-byte Reload
	cmpl	%esi, %edx
	cmovll	%esi, %edx
	movq	%r12, 288(%rsp)                 # 8-byte Spill
                                        # kill: def $r12d killed $r12d killed $r12 def $r12
	shll	$5, %r12d
	movq	240(%rsp), %rdi                 # 8-byte Reload
	leal	(%r12,%rdi), %esi
	movq	%r12, 144(%rsp)                 # 8-byte Spill
	addl	%r12d, %edi
	addl	$-4029, %edi                    # imm = 0xF043
	cmpl	%edi, %edx
	cmovlel	%edi, %edx
	orl	$14, %ecx
	movq	256(%rsp), %rdi                 # 8-byte Reload
	cmpl	%edi, %ecx
	cmovgl	%edi, %ecx
	orl	$30, %esi
	cmpl	%esi, %ecx
	cmovll	%ecx, %esi
	cmpl	%esi, %edx
	jg	.LBB0_20
# %bb.12:                               # %for.body1364.lr.ph.i
                                        #   in Loop: Header=BB0_11 Depth=3
	movslq	%eax, %rdx
	movl	%r8d, %esi
	movq	144(%rsp), %rcx                 # 8-byte Reload
	movl	%ecx, %eax
	orl	$31, %eax
	movslq	%ecx, %rcx
	movq	%rcx, 152(%rsp)                 # 8-byte Spill
	cmpq	%rsi, %rdx
	movq	%rsi, 24(%rsp)                  # 8-byte Spill
	cmovleq	%rsi, %rdx
	movq	%rdx, 304(%rsp)                 # 8-byte Spill
	cltq
	movq	%rax, 312(%rsp)                 # 8-byte Spill
	movl	%r8d, %eax
	negl	%eax
	movl	%eax, 4(%rsp)                   # 4-byte Spill
	leal	1(%r8), %ecx
	notl	%r8d
	movq	%r8, 32(%rsp)                   # 8-byte Spill
	jmp	.LBB0_14
	.p2align	4, 0x90
.LBB0_13:                               # %for.cond1207.loopexit.i
                                        #   in Loop: Header=BB0_14 Depth=4
	movl	104(%rsp), %ebx                 # 4-byte Reload
	addl	$-1, %ebx
	addl	$-1, 4(%rsp)                    # 4-byte Folded Spill
	movl	100(%rsp), %ecx                 # 4-byte Reload
	addl	$1, %ecx
	movq	32(%rsp), %rax                  # 8-byte Reload
	addl	$-1, %eax
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	movq	48(%rsp), %rax                  # 8-byte Reload
	cmpq	304(%rsp), %rax                 # 8-byte Folded Reload
	je	.LBB0_20
.LBB0_14:                               # %for.body1364.i
                                        #   Parent Loop BB0_5 Depth=1
                                        #     Parent Loop BB0_6 Depth=2
                                        #       Parent Loop BB0_11 Depth=3
                                        # =>      This Loop Header: Depth=4
                                        #           Child Loop BB0_16 Depth 5
                                        #             Child Loop BB0_18 Depth 6
	movl	(%rsp), %eax                    # 4-byte Reload
	cmpl	%ebx, %eax
	movl	%ebx, 104(%rsp)                 # 4-byte Spill
	movl	%ebx, %edi
	cmovgl	%eax, %edi
	movq	24(%rsp), %rbp                  # 8-byte Reload
	cmpl	%ecx, %edi
	movl	%ecx, 100(%rsp)                 # 4-byte Spill
	movl	%ecx, %r8d
	cmovgl	%edi, %r8d
	leaq	1(%rbp), %rax
	cmpl	%eax, %edi
	cmovlel	%eax, %edi
	movq	136(%rsp), %rcx                 # 8-byte Reload
	cmpq	%rax, %rcx
	movq	%rax, 24(%rsp)                  # 8-byte Spill
	cmovaq	%rcx, %rax
	movq	152(%rsp), %rsi                 # 8-byte Reload
	subq	%rbp, %rsi
	leaq	-3998(%rsi), %rdx
	cltq
	cmpq	%rdx, %rax
	cmovlel	%edx, %eax
	addq	$30, %rsi
	movq	296(%rsp), %rcx                 # 8-byte Reload
	cmpq	%rsi, %rcx
	cmovlq	%rcx, %rsi
	movq	%rbp, 48(%rsp)                  # 8-byte Spill
	leaq	3998(%rbp), %rdx
	movslq	%esi, %rbp
	cmpq	%rdx, %rbp
	cmovgeq	%rdx, %rsi
	cmpl	%esi, %eax
	jg	.LBB0_13
# %bb.15:                               # %for.body1437.lr.ph.i
                                        #   in Loop: Header=BB0_14 Depth=4
	movl	4(%rsp), %eax                   # 4-byte Reload
                                        # kill: def $eax killed $eax def $rax
	subl	%r8d, %eax
	movq	32(%rsp), %r15                  # 8-byte Reload
                                        # kill: def $r15d killed $r15d killed $r15 def $r15
	subl	%r8d, %r15d
	movl	%edi, %edx
	movq	24(%rsp), %rcx                  # 8-byte Reload
	addl	%ecx, %edi
	movslq	%esi, %rcx
	movq	%rcx, 320(%rsp)                 # 8-byte Spill
	jmp	.LBB0_16
	.p2align	4, 0x90
.LBB0_19:                               # %for.inc1576.i
                                        #   in Loop: Header=BB0_16 Depth=5
	movq	160(%rsp), %rdx                 # 8-byte Reload
	leaq	1(%rdx), %rcx
	movl	108(%rsp), %edi                 # 4-byte Reload
	addl	$1, %edi
	addq	$-1, %rax
	addq	$-1, %r15
	cmpq	320(%rsp), %rdx                 # 8-byte Folded Reload
	movq	%rcx, %rdx
	jge	.LBB0_13
.LBB0_16:                               # %for.body1437.i
                                        #   Parent Loop BB0_5 Depth=1
                                        #     Parent Loop BB0_6 Depth=2
                                        #       Parent Loop BB0_11 Depth=3
                                        #         Parent Loop BB0_14 Depth=4
                                        # =>        This Loop Header: Depth=5
                                        #             Child Loop BB0_18 Depth 6
	movq	8(%rsp), %rcx                   # 8-byte Reload
	cmpl	%edi, %ecx
	movl	%edi, 108(%rsp)                 # 4-byte Spill
	movl	%edi, %esi
	cmovgl	%ecx, %esi
	movq	48(%rsp), %rcx                  # 8-byte Reload
	leaq	(%rdx,%rcx), %rdi
	addq	$1, %rdi
	cmpq	152(%rsp), %rdi                 # 8-byte Folded Reload
	cmovll	144(%rsp), %edi                 # 4-byte Folded Reload
	movq	%rdx, 160(%rsp)                 # 8-byte Spill
	addq	%rdx, %rcx
	addq	$3998, %rcx                     # imm = 0xF9E
	movq	312(%rsp), %rdx                 # 8-byte Reload
	cmpq	%rdx, %rcx
	movl	%edx, %ebp
	cmovlel	%ecx, %ebp
	cmpl	%ebp, %edi
	movq	40(%rsp), %r11                  # 8-byte Reload
	jg	.LBB0_19
# %bb.17:                               # %for.body1466.lr.ph.i
                                        #   in Loop: Header=BB0_16 Depth=5
	movslq	%esi, %rdi
	addq	$-1, %rdi
	movq	160(%rsp), %r13                 # 8-byte Reload
	subq	48(%rsp), %r13                  # 8-byte Folded Reload
	leaq	-1(%r13), %r8
	leaq	1(%r13), %r14
	movslq	%ebp, %r12
	.p2align	4, 0x90
.LBB0_18:                               # %for.body1466.i
                                        #   Parent Loop BB0_5 Depth=1
                                        #     Parent Loop BB0_6 Depth=2
                                        #       Parent Loop BB0_11 Depth=3
                                        #         Parent Loop BB0_14 Depth=4
                                        #           Parent Loop BB0_16 Depth=5
                                        # =>          This Inner Loop Header: Depth=6
	leal	(%rax,%rdi), %ecx
	addl	$1, %ecx
	leal	(%r15,%rdi), %esi
	addl	$1, %esi
	movslq	%esi, %rsi
	imulq	$32000, %r8, %r10               # imm = 0x7D00
	addq	%r11, %r10
	movsd	(%r10,%rsi,8), %xmm1            # xmm1 = mem[0],zero
	movslq	%ecx, %rcx
	addsd	(%r10,%rcx,8), %xmm1
	leal	(%rax,%rdi), %r9d
	addl	$2, %r9d
	movslq	%r9d, %rbx
	addsd	(%r10,%rbx,8), %xmm1
	imulq	$32000, %r13, %rdx              # imm = 0x7D00
	addq	%r11, %rdx
	addsd	(%rdx,%rsi,8), %xmm1
	addsd	(%rdx,%rcx,8), %xmm1
	addsd	(%rdx,%rbx,8), %xmm1
	imulq	$32000, %r14, %rbp              # imm = 0x7D00
	addq	%r11, %rbp
	addsd	(%rbp,%rsi,8), %xmm1
	addsd	(%rbp,%rcx,8), %xmm1
	addsd	(%rbp,%rbx,8), %xmm1
	divsd	%xmm0, %xmm1
	movsd	%xmm1, (%rdx,%rcx,8)
	addq	$1, %rdi
	cmpq	%r12, %rdi
	jl	.LBB0_18
	jmp	.LBB0_19
.LBB0_23:                               # %kernel_seidel_2d.exit
	xorl	%eax, %eax
	callq	polybench_timer_stop
	xorl	%eax, %eax
	callq	polybench_timer_print
	cmpl	$43, 56(%rsp)                   # 4-byte Folded Reload
	jl	.LBB0_32
# %bb.24:                               # %if.end123
	movq	168(%rsp), %rax                 # 8-byte Reload
	movq	(%rax), %rax
	cmpb	$0, (%rax)
	je	.LBB0_25
.LBB0_32:                               # %if.end138
	movq	40(%rsp), %rdi                  # 8-byte Reload
	callq	free
	xorl	%eax, %eax
	addq	$328, %rsp                      # imm = 0x148
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
	.cfi_def_cfa_offset 384
	movq	stderr(%rip), %rcx
	movl	$.L.str.1, %edi
	movl	$22, %esi
	movl	$1, %edx
	callq	fwrite
	movq	stderr(%rip), %rdi
	xorl	%r12d, %r12d
	movl	$.L.str.2, %esi
	movl	$.L.str.3, %edx
	xorl	%eax, %eax
	callq	fprintf
	xorl	%ebx, %ebx
	movq	40(%rsp), %rbp                  # 8-byte Reload
	xorl	%r13d, %r13d
.LBB0_26:                               # %for.cond2.preheader.i
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_27 Depth 2
	movl	%r12d, %r14d
	xorl	%r15d, %r15d
.LBB0_27:                               # %for.body4.i
                                        #   Parent Loop BB0_26 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movl	%r14d, %eax
	movl	$3435973837, %ecx               # imm = 0xCCCCCCCD
	imulq	%rcx, %rax
	shrq	$36, %rax
	leal	(%rax,%rax,4), %eax
	leal	(%rbx,%rax,4), %eax
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
	addq	$1, %r13
	addq	$32000, %rbp                    # imm = 0x7D00
	addl	$-4000, %ebx                    # imm = 0xF060
	addl	$4000, %r12d                    # imm = 0xFA0
	cmpq	$4000, %r13                     # imm = 0xFA0
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
