	.text
	.file	"3mm.pluto.c"
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3                               # -- Begin function main
.LCPI0_0:
	.quad	0x40bf400000000000              # double 8000
.LCPI0_1:
	.quad	0x40c1940000000000              # double 9000
.LCPI0_2:
	.quad	0x40c57c0000000000              # double 11000
.LCPI0_3:
	.quad	0x40c3880000000000              # double 1.0E+4
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
	subq	$184, %rsp
	.cfi_def_cfa_offset 240
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rsi, 152(%rsp)                 # 8-byte Spill
	movl	%edi, 140(%rsp)                 # 4-byte Spill
	movl	$23040000, %edi                 # imm = 0x15F9000
	callq	malloc
	movq	%rax, %r14
	movl	$25600000, %edi                 # imm = 0x186A000
	callq	malloc
	movq	%rax, %r13
	movl	$28800000, %edi                 # imm = 0x1B77400
	callq	malloc
	movq	%rax, 120(%rsp)                 # 8-byte Spill
	movl	$31680000, %edi                 # imm = 0x1E36600
	callq	malloc
	movq	%rax, 96(%rsp)                  # 8-byte Spill
	movl	$34560000, %edi                 # imm = 0x20F5800
	callq	malloc
	movq	%rax, %rbx
	movl	$42240000, %edi                 # imm = 0x2848800
	callq	malloc
	movq	%rax, 112(%rsp)                 # 8-byte Spill
	movl	$28160000, %edi                 # imm = 0x1ADB000
	callq	malloc
	movq	%rax, 72(%rsp)                  # 8-byte Spill
	xorl	%eax, %eax
	movsd	.LCPI0_0(%rip), %xmm0           # xmm0 = mem[0],zero
	movq	%r13, %rcx
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
	cmpq	$2000, %rsi                     # imm = 0x7D0
	jne	.LBB0_2
# %bb.3:                                # %for.inc8.i
                                        #   in Loop: Header=BB0_1 Depth=1
	addq	$1, %rax
	addq	$16000, %rcx                    # imm = 0x3E80
	cmpq	$1600, %rax                     # imm = 0x640
	jne	.LBB0_1
# %bb.4:                                # %for.cond15.preheader.i.preheader
	movl	$2, %r8d
	xorl	%ecx, %ecx
	movl	$2443359173, %r9d               # imm = 0x91A2B3C5
	movsd	.LCPI0_1(%rip), %xmm0           # xmm0 = mem[0],zero
	movq	120(%rsp), %rsi                 # 8-byte Reload
	.p2align	4, 0x90
.LBB0_5:                                # %for.cond15.preheader.i
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_6 Depth 2
	movl	%r8d, %edx
	xorl	%ebp, %ebp
	.p2align	4, 0x90
.LBB0_6:                                # %for.body18.i
                                        #   Parent Loop BB0_5 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movl	%edx, %eax
	imulq	%r9, %rax
	shrq	$42, %rax
	imull	$1800, %eax, %eax               # imm = 0x708
	movl	%edx, %edi
	subl	%eax, %edi
	xorps	%xmm1, %xmm1
	cvtsi2sd	%edi, %xmm1
	divsd	%xmm0, %xmm1
	movsd	%xmm1, (%rsi,%rbp,8)
	addq	$1, %rbp
	addl	%ecx, %edx
	cmpq	$1800, %rbp                     # imm = 0x708
	jne	.LBB0_6
# %bb.7:                                # %for.inc34.i
                                        #   in Loop: Header=BB0_5 Depth=1
	addq	$1, %rcx
	addq	$14400, %rsi                    # imm = 0x3840
	addl	$1, %r8d
	cmpq	$2000, %rcx                     # imm = 0x7D0
	jne	.LBB0_5
# %bb.8:                                # %for.cond41.preheader.i.preheader
	xorl	%r8d, %r8d
	movsd	.LCPI0_2(%rip), %xmm0           # xmm0 = mem[0],zero
	movq	%rbx, %rcx
	xorl	%edx, %edx
	.p2align	4, 0x90
.LBB0_9:                                # %for.cond41.preheader.i
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_10 Depth 2
	movl	%r8d, %eax
	xorl	%edi, %edi
	.p2align	4, 0x90
.LBB0_10:                               # %for.body44.i
                                        #   Parent Loop BB0_9 Depth=1
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
	jne	.LBB0_10
# %bb.11:                               # %for.inc59.i
                                        #   in Loop: Header=BB0_9 Depth=1
	addq	$1, %rdx
	addq	$19200, %rcx                    # imm = 0x4B00
	addl	$3, %r8d
	cmpq	$1800, %rdx                     # imm = 0x708
	jne	.LBB0_9
# %bb.12:                               # %for.cond66.preheader.i.preheader
	movl	$2, %r8d
	xorl	%ecx, %ecx
	movsd	.LCPI0_3(%rip), %xmm0           # xmm0 = mem[0],zero
	movq	112(%rsp), %rdx                 # 8-byte Reload
	.p2align	4, 0x90
.LBB0_13:                               # %for.cond66.preheader.i
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_14 Depth 2
	movl	%r8d, %eax
	xorl	%edi, %edi
	.p2align	4, 0x90
.LBB0_14:                               # %for.body69.i
                                        #   Parent Loop BB0_13 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movl	%eax, %ebp
	imulq	$274877907, %rbp, %rbp          # imm = 0x10624DD3
	shrq	$39, %rbp
	imull	$2000, %ebp, %ebp               # imm = 0x7D0
	movl	%eax, %esi
	subl	%ebp, %esi
	xorps	%xmm1, %xmm1
	cvtsi2sd	%esi, %xmm1
	divsd	%xmm0, %xmm1
	movsd	%xmm1, (%rdx,%rdi,8)
	addq	$1, %rdi
	addl	%ecx, %eax
	cmpq	$2200, %rdi                     # imm = 0x898
	jne	.LBB0_14
# %bb.15:                               # %for.inc85.i
                                        #   in Loop: Header=BB0_13 Depth=1
	addq	$1, %rcx
	addq	$17600, %rdx                    # imm = 0x44C0
	addl	$2, %r8d
	cmpq	$2400, %rcx                     # imm = 0x960
	jne	.LBB0_13
# %bb.16:                               # %init_array.exit
	xorl	%ebp, %ebp
	xorl	%eax, %eax
	callq	polybench_timer_start
	movl	$1, %r9d
	movl	$31, %edi
	movq	72(%rsp), %r12                  # 8-byte Reload
	movq	96(%rsp), %rax                  # 8-byte Reload
	xorl	%ecx, %ecx
	jmp	.LBB0_17
	.p2align	4, 0x90
.LBB0_56:                               # %for.inc181.i
                                        #   in Loop: Header=BB0_17 Depth=1
	movl	144(%rsp), %ecx                 # 4-byte Reload
	addl	$1, %ecx
	movq	168(%rsp), %r9                  # 8-byte Reload
	addq	$-32, %r9
	movl	148(%rsp), %edi                 # 4-byte Reload
	addl	$32, %edi
	movq	80(%rsp), %rax                  # 8-byte Reload
	addq	$563200, %rax                   # imm = 0x89800
	movq	160(%rsp), %r12                 # 8-byte Reload
	addq	$563200, %r12                   # imm = 0x89800
	movq	176(%rsp), %rbp                 # 8-byte Reload
	addl	$32, %ebp
	cmpl	$107, %ecx
	je	.LBB0_57
.LBB0_17:                               # %for.cond15.preheader.i54
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_28 Depth 2
                                        #       Child Loop BB0_30 Depth 3
                                        #       Child Loop BB0_32 Depth 3
                                        #     Child Loop BB0_36 Depth 2
                                        #       Child Loop BB0_38 Depth 3
                                        #       Child Loop BB0_40 Depth 3
                                        #     Child Loop BB0_20 Depth 2
                                        #       Child Loop BB0_22 Depth 3
                                        #     Child Loop BB0_44 Depth 2
                                        #       Child Loop BB0_46 Depth 3
                                        #     Child Loop BB0_52 Depth 2
	movq	%rax, 80(%rsp)                  # 8-byte Spill
	cmpl	$1800, %ebp                     # imm = 0x708
	movl	$1800, %eax                     # imm = 0x708
	cmoval	%ebp, %eax
	movq	%rax, (%rsp)                    # 8-byte Spill
	cmpl	$1600, %ebp                     # imm = 0x640
	movl	$1600, %eax                     # imm = 0x640
	cmoval	%ebp, %eax
	imulq	$17600, %rax, %r15              # imm = 0x44C0
	addq	96(%rsp), %r15                  # 8-byte Folded Reload
	addl	$-1, %eax
	movq	%rax, 64(%rsp)                  # 8-byte Spill
	cmpl	$1599, %edi                     # imm = 0x63F
	movl	$1599, %r10d                    # imm = 0x63F
	cmovbl	%edi, %r10d
	movl	%ecx, %edx
	shll	$5, %edx
	cmpl	$1800, %edx                     # imm = 0x708
	movl	$1800, %eax                     # imm = 0x708
	cmoval	%edx, %eax
	movq	%rax, 40(%rsp)                  # 8-byte Spill
	cmpl	$1600, %edx                     # imm = 0x640
	movl	$1600, %r11d                    # imm = 0x640
	cmoval	%edx, %r11d
	leal	31(%rdx), %esi
	cmpl	$1599, %esi                     # imm = 0x63F
	movl	$1599, %r8d                     # imm = 0x63F
	cmovbl	%esi, %r8d
	cmpl	$1799, %esi                     # imm = 0x707
	movl	$1799, %eax                     # imm = 0x707
	cmovael	%eax, %esi
	movl	%r8d, 24(%rsp)                  # 4-byte Spill
	cmpl	%r8d, %edx
	movl	%esi, 16(%rsp)                  # 4-byte Spill
	movq	%rbp, 176(%rsp)                 # 8-byte Spill
	movq	%r9, 168(%rsp)                  # 8-byte Spill
	movl	%edi, 148(%rsp)                 # 4-byte Spill
	movq	%r12, 160(%rsp)                 # 8-byte Spill
	movl	%ecx, 144(%rsp)                 # 4-byte Spill
	jbe	.LBB0_18
# %bb.42:                               # %for.body31.i.us.preheader
                                        #   in Loop: Header=BB0_17 Depth=1
	movq	(%rsp), %rax                    # 8-byte Reload
	cmpl	%esi, %r11d
	jle	.LBB0_43
# %bb.51:                               # %for.body31.i.us.us.preheader
                                        #   in Loop: Header=BB0_17 Depth=1
	imulq	$17600, %rax, %rax              # imm = 0x44C0
	addq	72(%rsp), %rax                  # 8-byte Folded Reload
	movq	%rax, 16(%rsp)                  # 8-byte Spill
	movl	$31, %ebp
	xorl	%r12d, %r12d
	xorl	%r15d, %r15d
	movq	40(%rsp), %rsi                  # 8-byte Reload
	jmp	.LBB0_52
	.p2align	4, 0x90
.LBB0_55:                               # %for.inc178.i.us.us
                                        #   in Loop: Header=BB0_52 Depth=2
	addq	$32, %r15
	addl	$32, %ebp
	addl	$-32, %r12d
	cmpq	$2208, %r15                     # imm = 0x8A0
	je	.LBB0_56
.LBB0_52:                               # %for.body31.i.us.us
                                        #   Parent Loop BB0_17 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	cmpl	$2199, %ebp                     # imm = 0x897
	movl	$2199, %eax                     # imm = 0x897
	cmovbl	%ebp, %eax
	cmpl	%r15d, %eax
	cmovlel	%r15d, %eax
	cmpl	24(%rsp), %esi                  # 4-byte Folded Reload
	jg	.LBB0_55
# %bb.53:                               # %for.body152.lr.ph.i.us.us
                                        #   in Loop: Header=BB0_52 Depth=2
	leal	31(%r15), %ecx
	cmpl	$2199, %ecx                     # imm = 0x897
	movl	$2199, %edx                     # imm = 0x897
	cmovael	%edx, %ecx
	cmpl	%ecx, %r15d
	ja	.LBB0_55
# %bb.54:                               # %for.inc178.loopexit308.i.us.us
                                        #   in Loop: Header=BB0_52 Depth=2
	addl	%r12d, %eax
	leaq	8(,%rax,8), %rdx
	movq	16(%rsp), %rax                  # 8-byte Reload
	leaq	(%rax,%r15,8), %rdi
	xorl	%esi, %esi
	callq	memset
	movq	40(%rsp), %rsi                  # 8-byte Reload
	jmp	.LBB0_55
	.p2align	4, 0x90
.LBB0_18:                               # %for.body31.i.preheader
                                        #   in Loop: Header=BB0_17 Depth=1
	addq	%r9, %r10
	cmpl	%esi, %r11d
	movq	%r10, %rsi
	movq	%r10, 88(%rsp)                  # 8-byte Spill
	jle	.LBB0_26
# %bb.19:                               # %for.body31.i.us4.preheader
                                        #   in Loop: Header=BB0_17 Depth=1
	movq	%r12, (%rsp)                    # 8-byte Spill
	movq	80(%rsp), %rcx                  # 8-byte Reload
	xorl	%r15d, %r15d
	movq	%rsi, %rbp
	jmp	.LBB0_20
	.p2align	4, 0x90
.LBB0_25:                               # %for.inc178.i.us20
                                        #   in Loop: Header=BB0_20 Depth=2
	addq	$1, %r15
	addq	$256, %rcx                      # imm = 0x100
	addq	$256, (%rsp)                    # 8-byte Folded Spill
                                        # imm = 0x100
	cmpq	$69, %r15
	je	.LBB0_56
.LBB0_20:                               # %for.body31.i.us4
                                        #   Parent Loop BB0_17 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_22 Depth 3
	movq	%r15, %rdi
	shlq	$5, %rdi
	leal	31(%rdi), %eax
	cmpl	$2199, %eax                     # imm = 0x897
	movl	$2199, %r8d                     # imm = 0x897
	cmovael	%r8d, %eax
	cmpl	%edi, %eax
	movl	%edi, %r9d
	cmovgl	%eax, %r9d
	movl	%r15d, %edx
	shll	$5, %edx
	leal	31(%rdx), %esi
	cmpl	$2199, %esi                     # imm = 0x897
	cmovael	%r8d, %esi
	cmpl	%esi, %edx
	ja	.LBB0_25
# %bb.21:                               # %for.body61.i.us.preheader
                                        #   in Loop: Header=BB0_20 Depth=2
	movq	%r15, 56(%rsp)                  # 8-byte Spill
	movq	%rdi, 48(%rsp)                  # 8-byte Spill
	subq	%rdi, %rax
	leaq	8(,%rax,8), %rax
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	subl	%edx, %r9d
	leaq	8(,%r9,8), %rax
	movq	%rax, 64(%rsp)                  # 8-byte Spill
	xorl	%r15d, %r15d
	movq	%rcx, 8(%rsp)                   # 8-byte Spill
	.p2align	4, 0x90
.LBB0_22:                               # %for.body61.i.us
                                        #   Parent Loop BB0_17 Depth=1
                                        #     Parent Loop BB0_20 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movq	(%rsp), %rax                    # 8-byte Reload
	addq	%r15, %rax
	movq	%rax, 16(%rsp)                  # 8-byte Spill
	leaq	(%rcx,%r15), %rdi
	xorl	%esi, %esi
	movq	32(%rsp), %r12                  # 8-byte Reload
	movq	%r12, %rdx
	callq	memset
	movq	16(%rsp), %rdi                  # 8-byte Reload
	xorl	%esi, %esi
	movq	%r12, %rdx
	callq	memset
	movq	8(%rsp), %rcx                   # 8-byte Reload
	addq	$17600, %r15                    # imm = 0x44C0
	addq	$-1, %rbp
	jne	.LBB0_22
# %bb.23:                               # %for.end131.i.us16
                                        #   in Loop: Header=BB0_20 Depth=2
	movq	40(%rsp), %rax                  # 8-byte Reload
	cmpl	24(%rsp), %eax                  # 4-byte Folded Reload
	movq	88(%rsp), %rbp                  # 8-byte Reload
	movq	56(%rsp), %r15                  # 8-byte Reload
	movq	48(%rsp), %rdx                  # 8-byte Reload
	jg	.LBB0_25
# %bb.24:                               # %for.inc178.loopexit308.i.us17
                                        #   in Loop: Header=BB0_20 Depth=2
	imulq	$17600, %rax, %rax              # imm = 0x44C0
	addq	72(%rsp), %rax                  # 8-byte Folded Reload
	leaq	(%rax,%rdx,8), %rdi
	xorl	%esi, %esi
	movq	64(%rsp), %rdx                  # 8-byte Reload
	callq	memset
	movq	8(%rsp), %rcx                   # 8-byte Reload
	jmp	.LBB0_25
	.p2align	4, 0x90
.LBB0_43:                               # %for.body31.i.us.preheader83
                                        #   in Loop: Header=BB0_17 Depth=1
	xorl	%ebp, %ebp
	jmp	.LBB0_44
	.p2align	4, 0x90
.LBB0_50:                               # %for.inc178.i.us
                                        #   in Loop: Header=BB0_44 Depth=2
	addq	$1, %rbp
	addq	$256, %r15                      # imm = 0x100
	cmpq	$69, %rbp
	je	.LBB0_56
.LBB0_44:                               # %for.body31.i.us
                                        #   Parent Loop BB0_17 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_46 Depth 3
	movq	%rbp, %rdx
	shlq	$5, %rdx
	leal	31(%rdx), %eax
	cmpl	$2199, %eax                     # imm = 0x897
	movl	$2199, %ecx                     # imm = 0x897
	cmovael	%ecx, %eax
	cmpl	%edx, %eax
	cmovlel	%edx, %eax
	movl	%ebp, %esi
	shll	$5, %esi
	leal	31(%rsi), %edi
	cmpl	$2199, %edi                     # imm = 0x897
	cmovael	%ecx, %edi
	cmpl	%edi, %esi
	ja	.LBB0_50
# %bb.45:                               # %for.body106.i.us.preheader
                                        #   in Loop: Header=BB0_44 Depth=2
	movl	%edi, 56(%rsp)                  # 4-byte Spill
	movq	%rdx, 8(%rsp)                   # 8-byte Spill
	movq	%rbp, (%rsp)                    # 8-byte Spill
	movq	%rsi, 32(%rsp)                  # 8-byte Spill
	subl	%esi, %eax
	leaq	8(,%rax,8), %rdx
	movq	64(%rsp), %rax                  # 8-byte Reload
	movl	%eax, %r12d
	movq	%r15, 48(%rsp)                  # 8-byte Spill
	movq	%r15, %rbp
	.p2align	4, 0x90
.LBB0_46:                               # %for.body106.i.us
                                        #   Parent Loop BB0_17 Depth=1
                                        #     Parent Loop BB0_44 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movq	%rbp, %rdi
	xorl	%esi, %esi
	movq	%rdx, %r15
	callq	memset
	movq	%r15, %rdx
	addq	$17600, %rbp                    # imm = 0x44C0
	addl	$1, %r12d
	cmpl	16(%rsp), %r12d                 # 4-byte Folded Reload
	jb	.LBB0_46
# %bb.47:                               # %for.end131.i.us
                                        #   in Loop: Header=BB0_44 Depth=2
	movq	40(%rsp), %rax                  # 8-byte Reload
	cmpl	24(%rsp), %eax                  # 4-byte Folded Reload
	movq	48(%rsp), %r15                  # 8-byte Reload
	movq	(%rsp), %rbp                    # 8-byte Reload
	movq	8(%rsp), %rcx                   # 8-byte Reload
	movq	32(%rsp), %rsi                  # 8-byte Reload
	movl	56(%rsp), %edi                  # 4-byte Reload
	jg	.LBB0_50
# %bb.48:                               # %for.end131.i.us
                                        #   in Loop: Header=BB0_44 Depth=2
	cmpl	%edi, %esi
	ja	.LBB0_50
# %bb.49:                               # %for.inc178.loopexit308.i.us
                                        #   in Loop: Header=BB0_44 Depth=2
	imulq	$17600, %rax, %rax              # imm = 0x44C0
	addq	72(%rsp), %rax                  # 8-byte Folded Reload
	leaq	(%rax,%rcx,8), %rdi
	xorl	%esi, %esi
	callq	memset
	jmp	.LBB0_50
	.p2align	4, 0x90
.LBB0_26:                               # %for.body31.i.preheader.split
                                        #   in Loop: Header=BB0_17 Depth=1
	movl	24(%rsp), %eax                  # 4-byte Reload
	cmpl	%eax, 40(%rsp)                  # 4-byte Folded Reload
	jle	.LBB0_27
# %bb.35:                               # %for.body31.i.us24.preheader
                                        #   in Loop: Header=BB0_17 Depth=1
	movq	80(%rsp), %rax                  # 8-byte Reload
	xorl	%ecx, %ecx
	jmp	.LBB0_36
	.p2align	4, 0x90
.LBB0_41:                               # %for.inc178.i.us53
                                        #   in Loop: Header=BB0_36 Depth=2
	movq	56(%rsp), %rcx                  # 8-byte Reload
	addq	$1, %rcx
	movq	32(%rsp), %rax                  # 8-byte Reload
	addq	$256, %rax                      # imm = 0x100
	movq	8(%rsp), %r12                   # 8-byte Reload
	addq	$256, %r12                      # imm = 0x100
	movq	48(%rsp), %r15                  # 8-byte Reload
	addq	$256, %r15                      # imm = 0x100
	cmpq	$69, %rcx
	je	.LBB0_56
.LBB0_36:                               # %for.body31.i.us24
                                        #   Parent Loop BB0_17 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_38 Depth 3
                                        #       Child Loop BB0_40 Depth 3
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	movq	%r12, 8(%rsp)                   # 8-byte Spill
	movq	%r15, 48(%rsp)                  # 8-byte Spill
	movq	%rcx, 56(%rsp)                  # 8-byte Spill
	movl	%ecx, %eax
	shll	$5, %eax
	leal	31(%rax), %ecx
	cmpl	$2199, %ecx                     # imm = 0x897
	movl	$2199, %edx                     # imm = 0x897
	cmovael	%edx, %ecx
	cmpl	%eax, %ecx
	movl	%eax, %edx
	cmovgl	%ecx, %edx
	cmpl	%ecx, %eax
	ja	.LBB0_41
# %bb.37:                               # %for.body61.i.us33.preheader
                                        #   in Loop: Header=BB0_36 Depth=2
	movq	56(%rsp), %rsi                  # 8-byte Reload
	shlq	$5, %rsi
	subq	%rsi, %rcx
	leaq	8(,%rcx,8), %rcx
	movq	%rcx, 24(%rsp)                  # 8-byte Spill
	subl	%eax, %edx
	leaq	8(,%rdx,8), %rax
	movq	%rax, 40(%rsp)                  # 8-byte Spill
	xorl	%r15d, %r15d
	movq	88(%rsp), %r12                  # 8-byte Reload
	.p2align	4, 0x90
.LBB0_38:                               # %for.body61.i.us33
                                        #   Parent Loop BB0_17 Depth=1
                                        #     Parent Loop BB0_36 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movq	8(%rsp), %rax                   # 8-byte Reload
	addq	%r15, %rax
	movq	%rax, (%rsp)                    # 8-byte Spill
	movq	32(%rsp), %rax                  # 8-byte Reload
	leaq	(%rax,%r15), %rdi
	xorl	%esi, %esi
	movq	24(%rsp), %rbp                  # 8-byte Reload
	movq	%rbp, %rdx
	callq	memset
	movq	(%rsp), %rdi                    # 8-byte Reload
	xorl	%esi, %esi
	movq	%rbp, %rdx
	callq	memset
	addq	$17600, %r15                    # imm = 0x44C0
	addq	$-1, %r12
	jne	.LBB0_38
# %bb.39:                               # %for.body106.i.us40.preheader
                                        #   in Loop: Header=BB0_36 Depth=2
	movq	64(%rsp), %rax                  # 8-byte Reload
	movl	%eax, %r15d
	movq	48(%rsp), %rbp                  # 8-byte Reload
	movq	40(%rsp), %r12                  # 8-byte Reload
	.p2align	4, 0x90
.LBB0_40:                               # %for.body106.i.us40
                                        #   Parent Loop BB0_17 Depth=1
                                        #     Parent Loop BB0_36 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movq	%rbp, %rdi
	xorl	%esi, %esi
	movq	%r12, %rdx
	callq	memset
	addq	$17600, %rbp                    # imm = 0x44C0
	addl	$1, %r15d
	cmpl	16(%rsp), %r15d                 # 4-byte Folded Reload
	jb	.LBB0_40
	jmp	.LBB0_41
.LBB0_27:                               # %for.body31.i.preheader87
                                        #   in Loop: Header=BB0_17 Depth=1
	movq	%r12, %rcx
	movq	80(%rsp), %rax                  # 8-byte Reload
	xorl	%edx, %edx
	movq	%rsi, %r12
	jmp	.LBB0_28
	.p2align	4, 0x90
.LBB0_34:                               # %for.inc178.i
                                        #   in Loop: Header=BB0_28 Depth=2
	addq	$1, %rdx
	addq	$256, %rax                      # imm = 0x100
	addq	$256, %rcx                      # imm = 0x100
	addq	$256, %r15                      # imm = 0x100
	cmpq	$69, %rdx
	je	.LBB0_56
.LBB0_28:                               # %for.body31.i
                                        #   Parent Loop BB0_17 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_30 Depth 3
                                        #       Child Loop BB0_32 Depth 3
	movq	%rdx, %rbp
	shlq	$5, %rbp
	leal	31(%rbp), %edi
	cmpl	$2199, %edi                     # imm = 0x897
	movl	$2199, %r8d                     # imm = 0x897
	cmovael	%r8d, %edi
	cmpl	%ebp, %edi
	movl	%ebp, %r9d
	cmovgl	%edi, %r9d
	movl	%edx, %r10d
	shll	$5, %r10d
	leal	31(%r10), %esi
	cmpl	$2199, %esi                     # imm = 0x897
	cmovael	%r8d, %esi
	cmpl	%esi, %r10d
	ja	.LBB0_34
# %bb.29:                               # %for.body61.i.preheader
                                        #   in Loop: Header=BB0_28 Depth=2
	movq	%rdx, 104(%rsp)                 # 8-byte Spill
	movq	%r15, 48(%rsp)                  # 8-byte Spill
	movq	%rbp, %rdx
	movq	%rbp, 128(%rsp)                 # 8-byte Spill
	subq	%rbp, %rdi
	leaq	8(,%rdi,8), %rdx
	movq	%rdx, 56(%rsp)                  # 8-byte Spill
	subl	%r10d, %r9d
	leaq	8(,%r9,8), %rdx
	movq	%rdx, 24(%rsp)                  # 8-byte Spill
	xorl	%r15d, %r15d
	movq	%rcx, 8(%rsp)                   # 8-byte Spill
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	.p2align	4, 0x90
.LBB0_30:                               # %for.body61.i
                                        #   Parent Loop BB0_17 Depth=1
                                        #     Parent Loop BB0_28 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	addq	%r15, %rcx
	movq	%rcx, (%rsp)                    # 8-byte Spill
	leaq	(%rax,%r15), %rdi
	xorl	%esi, %esi
	movq	56(%rsp), %rbp                  # 8-byte Reload
	movq	%rbp, %rdx
	callq	memset
	movq	(%rsp), %rdi                    # 8-byte Reload
	xorl	%esi, %esi
	movq	%rbp, %rdx
	callq	memset
	movq	32(%rsp), %rax                  # 8-byte Reload
	movq	8(%rsp), %rcx                   # 8-byte Reload
	addq	$17600, %r15                    # imm = 0x44C0
	addq	$-1, %r12
	jne	.LBB0_30
# %bb.31:                               # %for.body106.i.preheader
                                        #   in Loop: Header=BB0_28 Depth=2
	movq	64(%rsp), %rax                  # 8-byte Reload
	movl	%eax, %r15d
	movq	48(%rsp), %rbp                  # 8-byte Reload
	movl	16(%rsp), %r12d                 # 4-byte Reload
	.p2align	4, 0x90
.LBB0_32:                               # %for.body106.i
                                        #   Parent Loop BB0_17 Depth=1
                                        #     Parent Loop BB0_28 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movq	%rbp, %rdi
	xorl	%esi, %esi
	movq	24(%rsp), %rdx                  # 8-byte Reload
	callq	memset
	addq	$17600, %rbp                    # imm = 0x44C0
	addl	$1, %r15d
	cmpl	%r12d, %r15d
	jb	.LBB0_32
# %bb.33:                               # %for.inc178.loopexit308.i
                                        #   in Loop: Header=BB0_28 Depth=2
	imulq	$17600, 40(%rsp), %rax          # 8-byte Folded Reload
                                        # imm = 0x44C0
	addq	72(%rsp), %rax                  # 8-byte Folded Reload
	movq	128(%rsp), %rcx                 # 8-byte Reload
	leaq	(%rax,%rcx,8), %rdi
	xorl	%esi, %esi
	movq	24(%rsp), %rdx                  # 8-byte Reload
	callq	memset
	movq	88(%rsp), %r12                  # 8-byte Reload
	movq	48(%rsp), %r15                  # 8-byte Reload
	movq	8(%rsp), %rcx                   # 8-byte Reload
	movq	32(%rsp), %rax                  # 8-byte Reload
	movq	104(%rsp), %rdx                 # 8-byte Reload
	jmp	.LBB0_34
.LBB0_57:                               # %for.cond375.preheader.i.preheader
	xorl	%eax, %eax
	movq	%rax, (%rsp)                    # 8-byte Spill
	movl	$31, %eax
	movq	96(%rsp), %rcx                  # 8-byte Reload
	movq	%rcx, 64(%rsp)                  # 8-byte Spill
	xorl	%ecx, %ecx
	jmp	.LBB0_58
	.p2align	4, 0x90
.LBB0_70:                               # %for.inc483.i
                                        #   in Loop: Header=BB0_58 Depth=1
	movl	40(%rsp), %ecx                  # 4-byte Reload
	addl	$1, %ecx
	addq	$32, (%rsp)                     # 8-byte Folded Spill
	movl	48(%rsp), %eax                  # 4-byte Reload
	addl	$32, %eax
	addq	$563200, 64(%rsp)               # 8-byte Folded Spill
                                        # imm = 0x89800
	cmpl	$57, %ecx
	je	.LBB0_71
.LBB0_58:                               # %for.cond375.preheader.i
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_60 Depth 2
                                        #       Child Loop BB0_62 Depth 3
                                        #         Child Loop BB0_63 Depth 4
                                        #           Child Loop BB0_64 Depth 5
                                        #             Child Loop BB0_65 Depth 6
	cmpl	$1799, %eax                     # imm = 0x707
	movl	$1799, %r12d                    # imm = 0x707
	movl	%eax, 48(%rsp)                  # 4-byte Spill
	cmovbl	%eax, %r12d
	movl	%ecx, 40(%rsp)                  # 4-byte Spill
	movl	%ecx, %eax
	shll	$5, %eax
	leal	31(%rax), %ecx
	cmpl	$1799, %ecx                     # imm = 0x707
	movl	$1799, %edx                     # imm = 0x707
	cmovael	%edx, %ecx
	cmpl	%ecx, %eax
	ja	.LBB0_70
# %bb.59:                               # %for.cond392.preheader.i.preheader
                                        #   in Loop: Header=BB0_58 Depth=1
	addl	$1, %r12d
	xorl	%eax, %eax
	movl	$31, %ecx
	movl	$1, %edx
	movq	%rdx, 24(%rsp)                  # 8-byte Spill
	movq	112(%rsp), %rdx                 # 8-byte Reload
	movq	%rdx, 56(%rsp)                  # 8-byte Spill
	movq	64(%rsp), %rdx                  # 8-byte Reload
	movq	%rdx, 16(%rsp)                  # 8-byte Spill
	jmp	.LBB0_60
	.p2align	4, 0x90
.LBB0_69:                               # %for.inc480.i
                                        #   in Loop: Header=BB0_60 Depth=2
	movl	8(%rsp), %eax                   # 4-byte Reload
	addl	$1, %eax
	movl	32(%rsp), %ecx                  # 4-byte Reload
	addl	$32, %ecx
	addq	$-32, 24(%rsp)                  # 8-byte Folded Spill
	addq	$256, 16(%rsp)                  # 8-byte Folded Spill
                                        # imm = 0x100
	addq	$256, 56(%rsp)                  # 8-byte Folded Spill
                                        # imm = 0x100
	cmpl	$69, %eax
	je	.LBB0_70
.LBB0_60:                               # %for.cond392.preheader.i
                                        #   Parent Loop BB0_58 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_62 Depth 3
                                        #         Child Loop BB0_63 Depth 4
                                        #           Child Loop BB0_64 Depth 5
                                        #             Child Loop BB0_65 Depth 6
	cmpl	$2199, %ecx                     # imm = 0x897
	movl	$2199, %r8d                     # imm = 0x897
	movl	%ecx, 32(%rsp)                  # 4-byte Spill
	cmovbl	%ecx, %r8d
	movl	%eax, 8(%rsp)                   # 4-byte Spill
                                        # kill: def $eax killed $eax def $rax
	shll	$5, %eax
	leal	31(%rax), %ecx
	cmpl	$2199, %ecx                     # imm = 0x897
	movl	$2199, %edx                     # imm = 0x897
	cmovael	%edx, %ecx
	cmpl	%ecx, %eax
	ja	.LBB0_69
# %bb.61:                               # %for.body408.i.preheader
                                        #   in Loop: Header=BB0_60 Depth=2
	addq	24(%rsp), %r8                   # 8-byte Folded Reload
	xorl	%ecx, %ecx
	movl	$32, %r10d
	movq	56(%rsp), %rdi                  # 8-byte Reload
	xorl	%r15d, %r15d
	.p2align	4, 0x90
.LBB0_62:                               # %for.body408.i
                                        #   Parent Loop BB0_58 Depth=1
                                        #     Parent Loop BB0_60 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_63 Depth 4
                                        #           Child Loop BB0_64 Depth 5
                                        #             Child Loop BB0_65 Depth 6
	movq	16(%rsp), %rsi                  # 8-byte Reload
	movq	(%rsp), %rax                    # 8-byte Reload
	.p2align	4, 0x90
.LBB0_63:                               # %for.body423.i
                                        #   Parent Loop BB0_58 Depth=1
                                        #     Parent Loop BB0_60 Depth=2
                                        #       Parent Loop BB0_62 Depth=3
                                        # =>      This Loop Header: Depth=4
                                        #           Child Loop BB0_64 Depth 5
                                        #             Child Loop BB0_65 Depth 6
	movq	%rdi, %rbp
	movq	%rcx, %r9
	.p2align	4, 0x90
.LBB0_64:                               # %for.body438.i
                                        #   Parent Loop BB0_58 Depth=1
                                        #     Parent Loop BB0_60 Depth=2
                                        #       Parent Loop BB0_62 Depth=3
                                        #         Parent Loop BB0_63 Depth=4
                                        # =>        This Loop Header: Depth=5
                                        #             Child Loop BB0_65 Depth 6
	imulq	$19200, %rax, %rdx              # imm = 0x4B00
	addq	%rbx, %rdx
	movsd	(%rdx,%r9,8), %xmm0             # xmm0 = mem[0],zero
	xorl	%r11d, %r11d
	.p2align	4, 0x90
.LBB0_65:                               # %for.body453.i
                                        #   Parent Loop BB0_58 Depth=1
                                        #     Parent Loop BB0_60 Depth=2
                                        #       Parent Loop BB0_62 Depth=3
                                        #         Parent Loop BB0_63 Depth=4
                                        #           Parent Loop BB0_64 Depth=5
                                        # =>          This Inner Loop Header: Depth=6
	movsd	(%rbp,%r11,8), %xmm1            # xmm1 = mem[0],zero
	mulsd	%xmm0, %xmm1
	addsd	(%rsi,%r11,8), %xmm1
	movsd	%xmm1, (%rsi,%r11,8)
	addq	$1, %r11
	cmpq	%r11, %r8
	jne	.LBB0_65
# %bb.66:                               # %for.inc471.i
                                        #   in Loop: Header=BB0_64 Depth=5
	addq	$1, %r9
	addq	$17600, %rbp                    # imm = 0x44C0
	cmpq	%r10, %r9
	jne	.LBB0_64
# %bb.67:                               # %for.inc474.i
                                        #   in Loop: Header=BB0_63 Depth=4
	addq	$1, %rax
	addq	$17600, %rsi                    # imm = 0x44C0
	cmpq	%r12, %rax
	jne	.LBB0_63
# %bb.68:                               # %for.inc477.i
                                        #   in Loop: Header=BB0_62 Depth=3
	addl	$1, %r15d
	addq	$32, %rcx
	addq	$32, %r10
	addq	$563200, %rdi                   # imm = 0x89800
	cmpl	$75, %r15d
	jne	.LBB0_62
	jmp	.LBB0_69
.LBB0_71:                               # %for.body522.preheader.i.preheader
	xorl	%eax, %eax
	movq	%r14, %r15
	jmp	.LBB0_72
	.p2align	4, 0x90
.LBB0_77:                               # %for.inc566.i
                                        #   in Loop: Header=BB0_72 Depth=1
	movq	(%rsp), %rax                    # 8-byte Reload
	addq	$1, %rax
	movq	8(%rsp), %r15                   # 8-byte Reload
	addq	$460800, %r15                   # imm = 0x70800
	cmpq	$50, %rax
	je	.LBB0_78
.LBB0_72:                               # %for.body522.preheader.i
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_73 Depth 2
                                        #       Child Loop BB0_75 Depth 3
	movq	%rax, (%rsp)                    # 8-byte Spill
	movq	%r15, 8(%rsp)                   # 8-byte Spill
	xorl	%eax, %eax
	jmp	.LBB0_73
	.p2align	4, 0x90
.LBB0_76:                               # %for.inc563.i
                                        #   in Loop: Header=BB0_73 Depth=2
	movq	16(%rsp), %rax                  # 8-byte Reload
	addq	$1, %rax
	addq	$256, %r15                      # imm = 0x100
	cmpq	$57, %rax
	je	.LBB0_77
.LBB0_73:                               # %for.body522.i
                                        #   Parent Loop BB0_72 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_75 Depth 3
	movq	%rax, 16(%rsp)                  # 8-byte Spill
                                        # kill: def $eax killed $eax killed $rax def $rax
	shll	$5, %eax
	leal	31(%rax), %edx
	cmpl	$1799, %edx                     # imm = 0x707
	movl	$1799, %ecx                     # imm = 0x707
	cmovael	%ecx, %edx
	cmpl	%eax, %edx
	movl	%eax, %ecx
	cmovgl	%edx, %ecx
	cmpl	%edx, %eax
	ja	.LBB0_76
# %bb.74:                               # %for.body537.i.preheader
                                        #   in Loop: Header=BB0_73 Depth=2
	subl	%eax, %ecx
	leaq	8(,%rcx,8), %rbp
	xorl	%r12d, %r12d
	.p2align	4, 0x90
.LBB0_75:                               # %for.body537.i
                                        #   Parent Loop BB0_72 Depth=1
                                        #     Parent Loop BB0_73 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	leaq	(%r15,%r12), %rdi
	xorl	%esi, %esi
	movq	%rbp, %rdx
	callq	memset
	addq	$14400, %r12                    # imm = 0x3840
	cmpq	$460800, %r12                   # imm = 0x70800
	jne	.LBB0_75
	jmp	.LBB0_76
.LBB0_78:                               # %for.cond589.preheader.i.preheader
	xorl	%eax, %eax
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	movl	$32, %r15d
	movq	72(%rsp), %rax                  # 8-byte Reload
	movq	%rax, 104(%rsp)                 # 8-byte Spill
	movq	%r14, %rax
	xorl	%ecx, %ecx
	jmp	.LBB0_79
	.p2align	4, 0x90
.LBB0_100:                              # %for.inc791.i
                                        #   in Loop: Header=BB0_79 Depth=1
	movl	80(%rsp), %ecx                  # 4-byte Reload
	addl	$1, %ecx
	addq	$32, 32(%rsp)                   # 8-byte Folded Spill
	addq	$32, %r15
	movq	128(%rsp), %rax                 # 8-byte Reload
	addq	$460800, %rax                   # imm = 0x70800
	addq	$563200, 104(%rsp)              # 8-byte Folded Spill
                                        # imm = 0x89800
	cmpl	$50, %ecx
	je	.LBB0_101
.LBB0_79:                               # %for.cond589.preheader.i
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_80 Depth 2
                                        #       Child Loop BB0_82 Depth 3
                                        #         Child Loop BB0_84 Depth 4
                                        #           Child Loop BB0_85 Depth 5
                                        #             Child Loop BB0_86 Depth 6
                                        #       Child Loop BB0_91 Depth 3
                                        #         Child Loop BB0_93 Depth 4
                                        #           Child Loop BB0_94 Depth 5
                                        #             Child Loop BB0_95 Depth 6
	movl	%ecx, 80(%rsp)                  # 4-byte Spill
	movl	$31, %ecx
	movl	$1, %edx
	movq	%rdx, 40(%rsp)                  # 8-byte Spill
	movq	96(%rsp), %rdx                  # 8-byte Reload
	movq	%rdx, 64(%rsp)                  # 8-byte Spill
	movq	120(%rsp), %rdx                 # 8-byte Reload
	movq	%rdx, 88(%rsp)                  # 8-byte Spill
	movq	%rax, 128(%rsp)                 # 8-byte Spill
	movq	%rax, 24(%rsp)                  # 8-byte Spill
	xorl	%eax, %eax
	movq	%rax, 16(%rsp)                  # 8-byte Spill
	xorl	%eax, %eax
	jmp	.LBB0_80
	.p2align	4, 0x90
.LBB0_99:                               # %for.inc788.i
                                        #   in Loop: Header=BB0_80 Depth=2
	movl	48(%rsp), %eax                  # 4-byte Reload
	addl	$1, %eax
	addq	$32, 16(%rsp)                   # 8-byte Folded Spill
	movl	56(%rsp), %ecx                  # 4-byte Reload
	addl	$32, %ecx
	addq	$-32, 40(%rsp)                  # 8-byte Folded Spill
	addq	$256, 24(%rsp)                  # 8-byte Folded Spill
                                        # imm = 0x100
	addq	$256, 88(%rsp)                  # 8-byte Folded Spill
                                        # imm = 0x100
	addq	$563200, 64(%rsp)               # 8-byte Folded Spill
                                        # imm = 0x89800
	cmpl	$57, %eax
	je	.LBB0_100
.LBB0_80:                               # %for.cond608.preheader.i
                                        #   Parent Loop BB0_79 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_82 Depth 3
                                        #         Child Loop BB0_84 Depth 4
                                        #           Child Loop BB0_85 Depth 5
                                        #             Child Loop BB0_86 Depth 6
                                        #       Child Loop BB0_91 Depth 3
                                        #         Child Loop BB0_93 Depth 4
                                        #           Child Loop BB0_94 Depth 5
                                        #             Child Loop BB0_95 Depth 6
	cmpl	$1799, %ecx                     # imm = 0x707
	movl	$1799, %r11d                    # imm = 0x707
	movl	%ecx, 56(%rsp)                  # 4-byte Spill
	cmovbl	%ecx, %r11d
	movl	%eax, 48(%rsp)                  # 4-byte Spill
                                        # kill: def $eax killed $eax def $rax
	shll	$5, %eax
	leal	31(%rax), %ecx
	cmpl	$1799, %ecx                     # imm = 0x707
	movl	$1799, %edx                     # imm = 0x707
	cmovael	%edx, %ecx
	cmpl	%ecx, %eax
	ja	.LBB0_99
# %bb.81:                               # %for.body639.lr.ph.i.preheader
                                        #   in Loop: Header=BB0_80 Depth=2
	movq	40(%rsp), %rax                  # 8-byte Reload
	leaq	(%rax,%r11), %rbp
	addl	$1, %r11d
	xorl	%r9d, %r9d
	movl	$31, %eax
	movq	88(%rsp), %r8                   # 8-byte Reload
	xorl	%r12d, %r12d
	jmp	.LBB0_82
	.p2align	4, 0x90
.LBB0_89:                               # %for.inc693.i
                                        #   in Loop: Header=BB0_82 Depth=3
	addl	$1, %r12d
	addq	$32, %r9
	movl	(%rsp), %eax                    # 4-byte Reload
	addl	$32, %eax
	addq	$460800, %r8                    # imm = 0x70800
	cmpl	$63, %r12d
	je	.LBB0_90
.LBB0_82:                               # %for.body639.lr.ph.i
                                        #   Parent Loop BB0_79 Depth=1
                                        #     Parent Loop BB0_80 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_84 Depth 4
                                        #           Child Loop BB0_85 Depth 5
                                        #             Child Loop BB0_86 Depth 6
	cmpl	$1999, %eax                     # imm = 0x7CF
	movl	$1999, %edi                     # imm = 0x7CF
	movl	%eax, (%rsp)                    # 4-byte Spill
	cmovbl	%eax, %edi
	movl	%r12d, %eax
	shll	$5, %eax
	leal	31(%rax), %ecx
	cmpl	$1999, %ecx                     # imm = 0x7CF
	movl	$1999, %edx                     # imm = 0x7CF
	cmovael	%edx, %ecx
	cmpl	%ecx, %eax
	ja	.LBB0_89
# %bb.83:                               # %for.body639.i.preheader
                                        #   in Loop: Header=BB0_82 Depth=3
	addl	$1, %edi
	movq	24(%rsp), %rax                  # 8-byte Reload
	movq	32(%rsp), %r10                  # 8-byte Reload
	.p2align	4, 0x90
.LBB0_84:                               # %for.body639.i
                                        #   Parent Loop BB0_79 Depth=1
                                        #     Parent Loop BB0_80 Depth=2
                                        #       Parent Loop BB0_82 Depth=3
                                        # =>      This Loop Header: Depth=4
                                        #           Child Loop BB0_85 Depth 5
                                        #             Child Loop BB0_86 Depth 6
	movq	%r8, %rdx
	movq	%r9, %rsi
	.p2align	4, 0x90
.LBB0_85:                               # %for.body654.i
                                        #   Parent Loop BB0_79 Depth=1
                                        #     Parent Loop BB0_80 Depth=2
                                        #       Parent Loop BB0_82 Depth=3
                                        #         Parent Loop BB0_84 Depth=4
                                        # =>        This Loop Header: Depth=5
                                        #             Child Loop BB0_86 Depth 6
	imulq	$16000, %r10, %rcx              # imm = 0x3E80
	addq	%r13, %rcx
	movsd	(%rcx,%rsi,8), %xmm0            # xmm0 = mem[0],zero
	xorl	%ecx, %ecx
	.p2align	4, 0x90
.LBB0_86:                               # %for.body669.i
                                        #   Parent Loop BB0_79 Depth=1
                                        #     Parent Loop BB0_80 Depth=2
                                        #       Parent Loop BB0_82 Depth=3
                                        #         Parent Loop BB0_84 Depth=4
                                        #           Parent Loop BB0_85 Depth=5
                                        # =>          This Inner Loop Header: Depth=6
	movsd	(%rdx,%rcx,8), %xmm1            # xmm1 = mem[0],zero
	mulsd	%xmm0, %xmm1
	addsd	(%rax,%rcx,8), %xmm1
	movsd	%xmm1, (%rax,%rcx,8)
	addq	$1, %rcx
	cmpq	%rcx, %rbp
	jne	.LBB0_86
# %bb.87:                               # %for.inc687.i
                                        #   in Loop: Header=BB0_85 Depth=5
	addq	$1, %rsi
	addq	$14400, %rdx                    # imm = 0x3840
	cmpq	%rdi, %rsi
	jne	.LBB0_85
# %bb.88:                               # %for.inc690.i
                                        #   in Loop: Header=BB0_84 Depth=4
	addq	$1, %r10
	addq	$14400, %rax                    # imm = 0x3840
	cmpq	%r15, %r10
	jne	.LBB0_84
	jmp	.LBB0_89
	.p2align	4, 0x90
.LBB0_90:                               # %for.body730.lr.ph.i.preheader
                                        #   in Loop: Header=BB0_80 Depth=2
	xorl	%r9d, %r9d
	movl	$31, %r8d
	movl	$1, %eax
	movq	%rax, (%rsp)                    # 8-byte Spill
	movq	64(%rsp), %r10                  # 8-byte Reload
	movq	104(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 8(%rsp)                   # 8-byte Spill
	jmp	.LBB0_91
	.p2align	4, 0x90
.LBB0_98:                               # %for.inc784.i
                                        #   in Loop: Header=BB0_91 Depth=3
	addl	$1, %r9d
	addl	$32, %r8d
	addq	$-32, (%rsp)                    # 8-byte Folded Spill
	addq	$256, 8(%rsp)                   # 8-byte Folded Spill
                                        # imm = 0x100
	addq	$256, %r10                      # imm = 0x100
	cmpl	$69, %r9d
	je	.LBB0_99
.LBB0_91:                               # %for.body730.lr.ph.i
                                        #   Parent Loop BB0_79 Depth=1
                                        #     Parent Loop BB0_80 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_93 Depth 4
                                        #           Child Loop BB0_94 Depth 5
                                        #             Child Loop BB0_95 Depth 6
	cmpl	$2199, %r8d                     # imm = 0x897
	movl	$2199, %eax                     # imm = 0x897
	cmovbl	%r8d, %eax
	movl	%r9d, %ecx
	shll	$5, %ecx
	leal	31(%rcx), %edx
	cmpl	$2199, %edx                     # imm = 0x897
	movl	$2199, %esi                     # imm = 0x897
	cmovael	%esi, %edx
	cmpl	%edx, %ecx
	ja	.LBB0_98
# %bb.92:                               # %for.body730.i.preheader
                                        #   in Loop: Header=BB0_91 Depth=3
	addq	(%rsp), %rax                    # 8-byte Folded Reload
	movq	8(%rsp), %rdx                   # 8-byte Reload
	movq	32(%rsp), %rdi                  # 8-byte Reload
	.p2align	4, 0x90
.LBB0_93:                               # %for.body730.i
                                        #   Parent Loop BB0_79 Depth=1
                                        #     Parent Loop BB0_80 Depth=2
                                        #       Parent Loop BB0_91 Depth=3
                                        # =>      This Loop Header: Depth=4
                                        #           Child Loop BB0_94 Depth 5
                                        #             Child Loop BB0_95 Depth 6
	movq	%r10, %rsi
	movq	16(%rsp), %rcx                  # 8-byte Reload
	.p2align	4, 0x90
.LBB0_94:                               # %for.body745.i
                                        #   Parent Loop BB0_79 Depth=1
                                        #     Parent Loop BB0_80 Depth=2
                                        #       Parent Loop BB0_91 Depth=3
                                        #         Parent Loop BB0_93 Depth=4
                                        # =>        This Loop Header: Depth=5
                                        #             Child Loop BB0_95 Depth 6
	imulq	$14400, %rdi, %rbp              # imm = 0x3840
	addq	%r14, %rbp
	movsd	(%rbp,%rcx,8), %xmm0            # xmm0 = mem[0],zero
	xorl	%r12d, %r12d
	.p2align	4, 0x90
.LBB0_95:                               # %for.body760.i
                                        #   Parent Loop BB0_79 Depth=1
                                        #     Parent Loop BB0_80 Depth=2
                                        #       Parent Loop BB0_91 Depth=3
                                        #         Parent Loop BB0_93 Depth=4
                                        #           Parent Loop BB0_94 Depth=5
                                        # =>          This Inner Loop Header: Depth=6
	movsd	(%rsi,%r12,8), %xmm1            # xmm1 = mem[0],zero
	mulsd	%xmm0, %xmm1
	addsd	(%rdx,%r12,8), %xmm1
	movsd	%xmm1, (%rdx,%r12,8)
	addq	$1, %r12
	cmpq	%r12, %rax
	jne	.LBB0_95
# %bb.96:                               # %for.inc778.i
                                        #   in Loop: Header=BB0_94 Depth=5
	addq	$1, %rcx
	addq	$17600, %rsi                    # imm = 0x44C0
	cmpq	%r11, %rcx
	jne	.LBB0_94
# %bb.97:                               # %for.inc781.i
                                        #   in Loop: Header=BB0_93 Depth=4
	addq	$1, %rdi
	addq	$17600, %rdx                    # imm = 0x44C0
	cmpq	%r15, %rdi
	jne	.LBB0_93
	jmp	.LBB0_98
.LBB0_101:                              # %kernel_3mm.exit
	xorl	%eax, %eax
	callq	polybench_timer_stop
	xorl	%eax, %eax
	callq	polybench_timer_print
	cmpl	$43, 140(%rsp)                  # 4-byte Folded Reload
	jl	.LBB0_110
# %bb.102:                              # %land.lhs.true
	movq	152(%rsp), %rax                 # 8-byte Reload
	movq	(%rax), %rax
	cmpb	$0, (%rax)
	je	.LBB0_103
.LBB0_110:                              # %if.end
	movq	%r14, %rdi
	callq	free
	movq	%r13, %rdi
	callq	free
	movq	120(%rsp), %rdi                 # 8-byte Reload
	callq	free
	movq	96(%rsp), %rdi                  # 8-byte Reload
	callq	free
	movq	%rbx, %rdi
	callq	free
	movq	112(%rsp), %rdi                 # 8-byte Reload
	callq	free
	movq	72(%rsp), %rdi                  # 8-byte Reload
	callq	free
	xorl	%eax, %eax
	addq	$184, %rsp
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
.LBB0_103:                              # %if.then
	.cfi_def_cfa_offset 240
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
	xorl	%edx, %edx
	movq	72(%rsp), %r15                  # 8-byte Reload
	xorl	%eax, %eax
.LBB0_104:                              # %for.cond2.preheader.i
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_105 Depth 2
	movq	%rax, 8(%rsp)                   # 8-byte Spill
	movq	%rbp, (%rsp)                    # 8-byte Spill
	movl	%ebp, %r12d
	xorl	%ebp, %ebp
	movq	%rdx, 16(%rsp)                  # 8-byte Spill
.LBB0_105:                              # %for.body4.i
                                        #   Parent Loop BB0_104 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movl	%r12d, %eax
	movl	$3435973837, %ecx               # imm = 0xCCCCCCCD
	imulq	%rcx, %rax
	shrq	$36, %rax
	leal	(%rax,%rax,4), %eax
	leal	(%rdx,%rax,4), %eax
	cmpl	%ebp, %eax
	jne	.LBB0_107
# %bb.106:                              # %if.then.i
                                        #   in Loop: Header=BB0_105 Depth=2
	movq	stderr(%rip), %rsi
	movl	$10, %edi
	callq	fputc
.LBB0_107:                              # %if.end.i
                                        #   in Loop: Header=BB0_105 Depth=2
	movq	stderr(%rip), %rdi
	movsd	(%r15,%rbp,8), %xmm0            # xmm0 = mem[0],zero
	movl	$.L.str.5, %esi
	movb	$1, %al
	callq	fprintf
	addq	$1, %rbp
	addl	$1, %r12d
	cmpq	$2200, %rbp                     # imm = 0x898
	movq	16(%rsp), %rdx                  # 8-byte Reload
	jne	.LBB0_105
# %bb.108:                              # %for.inc10.i
                                        #   in Loop: Header=BB0_104 Depth=1
	movq	8(%rsp), %rax                   # 8-byte Reload
	addq	$1, %rax
	addq	$17600, %r15                    # imm = 0x44C0
	addl	$-1600, %edx                    # imm = 0xF9C0
	movq	(%rsp), %rbp                    # 8-byte Reload
	addl	$1600, %ebp                     # imm = 0x640
	cmpq	$1600, %rax                     # imm = 0x640
	jne	.LBB0_104
# %bb.109:                              # %print_array.exit
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
	jmp	.LBB0_110
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
	.asciz	"G"
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
