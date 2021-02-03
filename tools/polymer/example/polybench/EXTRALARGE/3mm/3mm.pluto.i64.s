	.text
	.file	"3mm.pluto.i64.c"
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
	subq	$200, %rsp
	.cfi_def_cfa_offset 256
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rsi, 152(%rsp)                 # 8-byte Spill
	movl	%edi, 148(%rsp)                 # 4-byte Spill
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
	movq	%rax, 80(%rsp)                  # 8-byte Spill
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
	movl	$31, %esi
	movl	$1, %r9d
	movq	80(%rsp), %r11                  # 8-byte Reload
	movq	96(%rsp), %r10                  # 8-byte Reload
	xorl	%edi, %edi
	jmp	.LBB0_17
	.p2align	4, 0x90
.LBB0_65:                               # %for.inc210.i
                                        #   in Loop: Header=BB0_17 Depth=1
	movq	160(%rsp), %rdi                 # 8-byte Reload
	addq	$1, %rdi
	movq	192(%rsp), %rsi                 # 8-byte Reload
	addq	$32, %rsi
	movq	184(%rsp), %r9                  # 8-byte Reload
	addq	$-32, %r9
	movq	168(%rsp), %r10                 # 8-byte Reload
	addq	$563200, %r10                   # imm = 0x89800
	movq	176(%rsp), %r11                 # 8-byte Reload
	addq	$563200, %r11                   # imm = 0x89800
	movq	128(%rsp), %rbp                 # 8-byte Reload
	addq	$32, %rbp
	cmpq	$107, %rdi
	je	.LBB0_66
.LBB0_17:                               # %for.cond16.preheader.i
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_39 Depth 2
                                        #       Child Loop BB0_41 Depth 3
                                        #       Child Loop BB0_43 Depth 3
                                        #     Child Loop BB0_47 Depth 2
                                        #       Child Loop BB0_49 Depth 3
                                        #       Child Loop BB0_51 Depth 3
                                        #     Child Loop BB0_55 Depth 2
                                        #       Child Loop BB0_57 Depth 3
                                        #     Child Loop BB0_61 Depth 2
                                        #       Child Loop BB0_63 Depth 3
                                        #     Child Loop BB0_26 Depth 2
                                        #       Child Loop BB0_28 Depth 3
                                        #     Child Loop BB0_32 Depth 2
                                        #       Child Loop BB0_34 Depth 3
                                        #     Child Loop BB0_20 Depth 2
	cmpq	$1800, %rbp                     # imm = 0x708
	movl	$1800, %eax                     # imm = 0x708
	cmovaq	%rbp, %rax
	movq	%rax, 8(%rsp)                   # 8-byte Spill
	cmpq	$1600, %rbp                     # imm = 0x640
	movl	$1600, %eax                     # imm = 0x640
	cmovaq	%rbp, %rax
	imulq	$17600, %rax, %rcx              # imm = 0x44C0
	addq	96(%rsp), %rcx                  # 8-byte Folded Reload
	movq	%rcx, 24(%rsp)                  # 8-byte Spill
	addq	$-1, %rax
	movq	%rax, 72(%rsp)                  # 8-byte Spill
	cmpq	$1599, %rsi                     # imm = 0x63F
	movl	$1599, %r15d                    # imm = 0x63F
	cmovbq	%rsi, %r15
	movq	%rdi, %rdx
	shlq	$5, %rdx
	cmpq	$1800, %rdx                     # imm = 0x708
	movl	$1800, %eax                     # imm = 0x708
	cmovaq	%rdx, %rax
	movq	%rax, 40(%rsp)                  # 8-byte Spill
	cmpq	$1600, %rdx                     # imm = 0x640
	movl	$1600, %r12d                    # imm = 0x640
	cmovaq	%rdx, %r12
	leaq	31(%rdx), %rcx
	cmpq	$1599, %rcx                     # imm = 0x63F
	movl	$1599, %r8d                     # imm = 0x63F
	cmovbq	%rcx, %r8
	cmpq	$1799, %rcx                     # imm = 0x707
	movl	$1799, %eax                     # imm = 0x707
	cmovaeq	%rax, %rcx
	movq	%rcx, %rax
	movq	%r8, 16(%rsp)                   # 8-byte Spill
	cmpq	%r8, %rdx
	movq	%rcx, 64(%rsp)                  # 8-byte Spill
	movq	%rbp, 128(%rsp)                 # 8-byte Spill
	movq	%rsi, 192(%rsp)                 # 8-byte Spill
	movq	%r9, 184(%rsp)                  # 8-byte Spill
	movq	%r11, 176(%rsp)                 # 8-byte Spill
	movq	%r10, 168(%rsp)                 # 8-byte Spill
	movq	%rdi, 160(%rsp)                 # 8-byte Spill
	jbe	.LBB0_36
# %bb.18:                               # %for.body35.i.us.preheader
                                        #   in Loop: Header=BB0_17 Depth=1
	movq	8(%rsp), %rcx                   # 8-byte Reload
	cmpq	%rax, %r12
	jle	.LBB0_24
# %bb.19:                               # %for.body35.i.us.us.preheader
                                        #   in Loop: Header=BB0_17 Depth=1
	imulq	$17600, %rcx, %rax              # imm = 0x44C0
	addq	80(%rsp), %rax                  # 8-byte Folded Reload
	movq	%rax, 8(%rsp)                   # 8-byte Spill
	movl	$31, %ebp
	movl	$1, %r15d
	xorl	%r12d, %r12d
	jmp	.LBB0_20
	.p2align	4, 0x90
.LBB0_23:                               # %for.inc207.i.us.us
                                        #   in Loop: Header=BB0_20 Depth=2
	addq	$32, %r12
	addq	$32, %rbp
	addq	$-32, %r15
	cmpq	$2208, %r12                     # imm = 0x8A0
	je	.LBB0_65
.LBB0_20:                               # %for.body35.i.us.us
                                        #   Parent Loop BB0_17 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	cmpq	$2199, %rbp                     # imm = 0x897
	movl	$2199, %edx                     # imm = 0x897
	cmovbq	%rbp, %rdx
	cmpq	%r12, %rdx
	cmovbeq	%r12, %rdx
	leaq	31(%r12), %rax
	cmpq	$2199, %rax                     # imm = 0x897
	movl	$2199, %ecx                     # imm = 0x897
	cmovaeq	%rcx, %rax
	movq	16(%rsp), %rcx                  # 8-byte Reload
	cmpq	%rcx, 40(%rsp)                  # 8-byte Folded Reload
	jg	.LBB0_23
# %bb.21:                               # %for.body35.i.us.us
                                        #   in Loop: Header=BB0_20 Depth=2
	cmpq	%rax, %r12
	ja	.LBB0_23
# %bb.22:                               # %for.inc207.loopexit285.i.us.us
                                        #   in Loop: Header=BB0_20 Depth=2
	addq	%r15, %rdx
	shlq	$3, %rdx
	movq	8(%rsp), %rax                   # 8-byte Reload
	leaq	(%rax,%r12,8), %rdi
	xorl	%esi, %esi
	callq	memset
	jmp	.LBB0_23
	.p2align	4, 0x90
.LBB0_36:                               # %for.cond16.preheader.i.split
                                        #   in Loop: Header=BB0_17 Depth=1
	addq	%r9, %r15
	cmpq	%rax, %r12
	movq	%r15, 88(%rsp)                  # 8-byte Spill
	jle	.LBB0_37
# %bb.53:                               # %for.body35.i.us240.preheader
                                        #   in Loop: Header=BB0_17 Depth=1
	movq	16(%rsp), %rax                  # 8-byte Reload
	cmpq	%rax, 40(%rsp)                  # 8-byte Folded Reload
	jle	.LBB0_54
# %bb.60:                               # %for.body35.i.us240.us.preheader
                                        #   in Loop: Header=BB0_17 Depth=1
	xorl	%eax, %eax
	jmp	.LBB0_61
	.p2align	4, 0x90
.LBB0_64:                               # %for.inc207.i.us272.us
                                        #   in Loop: Header=BB0_61 Depth=2
	movq	48(%rsp), %rax                  # 8-byte Reload
	addq	$1, %rax
	addq	$256, %r10                      # imm = 0x100
	addq	$256, %r11                      # imm = 0x100
	cmpq	$69, %rax
	je	.LBB0_65
.LBB0_61:                               # %for.body35.i.us240.us
                                        #   Parent Loop BB0_17 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_63 Depth 3
	movq	%rax, 48(%rsp)                  # 8-byte Spill
	shlq	$5, %rax
	leaq	31(%rax), %rcx
	cmpq	$2199, %rcx                     # imm = 0x897
	movl	$2199, %edx                     # imm = 0x897
	cmovaeq	%rdx, %rcx
	subq	%rcx, %rax
	ja	.LBB0_64
# %bb.62:                               # %for.body71.i.us244.us.preheader
                                        #   in Loop: Header=BB0_61 Depth=2
	shlq	$3, %rax
	movl	$8, %ecx
	subq	%rax, %rcx
	movq	%rcx, 56(%rsp)                  # 8-byte Spill
	xorl	%r12d, %r12d
	movq	88(%rsp), %r15                  # 8-byte Reload
	movq	%r11, 8(%rsp)                   # 8-byte Spill
	movq	%r10, 32(%rsp)                  # 8-byte Spill
	.p2align	4, 0x90
.LBB0_63:                               # %for.body71.i.us244.us
                                        #   Parent Loop BB0_17 Depth=1
                                        #     Parent Loop BB0_61 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	leaq	(%r11,%r12), %rax
	movq	%rax, 16(%rsp)                  # 8-byte Spill
	leaq	(%r10,%r12), %rdi
	xorl	%esi, %esi
	movq	56(%rsp), %rbp                  # 8-byte Reload
	movq	%rbp, %rdx
	callq	memset
	movq	16(%rsp), %rdi                  # 8-byte Reload
	xorl	%esi, %esi
	movq	%rbp, %rdx
	callq	memset
	movq	32(%rsp), %r10                  # 8-byte Reload
	movq	8(%rsp), %r11                   # 8-byte Reload
	addq	$17600, %r12                    # imm = 0x44C0
	addq	$-1, %r15
	jne	.LBB0_63
	jmp	.LBB0_64
	.p2align	4, 0x90
.LBB0_24:                               # %for.body35.i.us.preheader.split
                                        #   in Loop: Header=BB0_17 Depth=1
	movq	16(%rsp), %rax                  # 8-byte Reload
	cmpq	%rax, 40(%rsp)                  # 8-byte Folded Reload
	jle	.LBB0_25
# %bb.31:                               # %for.body35.i.us.us26.preheader
                                        #   in Loop: Header=BB0_17 Depth=1
	xorl	%ecx, %ecx
	movq	24(%rsp), %rax                  # 8-byte Reload
	jmp	.LBB0_32
	.p2align	4, 0x90
.LBB0_35:                               # %for.inc207.i.us.us44
                                        #   in Loop: Header=BB0_32 Depth=2
	movq	8(%rsp), %rcx                   # 8-byte Reload
	addq	$1, %rcx
	movq	24(%rsp), %rax                  # 8-byte Reload
	addq	$256, %rax                      # imm = 0x100
	cmpq	$69, %rcx
	je	.LBB0_65
.LBB0_32:                               # %for.body35.i.us.us26
                                        #   Parent Loop BB0_17 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_34 Depth 3
	movq	%rax, 24(%rsp)                  # 8-byte Spill
	movq	%rcx, 8(%rsp)                   # 8-byte Spill
	movq	%rcx, %rax
	shlq	$5, %rax
	leaq	31(%rax), %rdx
	cmpq	$2199, %rdx                     # imm = 0x897
	movl	$2199, %ecx                     # imm = 0x897
	cmovaeq	%rcx, %rdx
	cmpq	%rax, %rdx
	movq	%rax, %rcx
	cmovaq	%rdx, %rcx
	cmpq	%rdx, %rax
	movq	64(%rsp), %r15                  # 8-byte Reload
	ja	.LBB0_35
# %bb.33:                               # %for.body124.i.us.us31.preheader
                                        #   in Loop: Header=BB0_32 Depth=2
	subq	%rax, %rcx
	leaq	8(,%rcx,8), %rax
	movq	%rax, 16(%rsp)                  # 8-byte Spill
	movq	72(%rsp), %r12                  # 8-byte Reload
	movq	24(%rsp), %rbp                  # 8-byte Reload
	.p2align	4, 0x90
.LBB0_34:                               # %for.body124.i.us.us31
                                        #   Parent Loop BB0_17 Depth=1
                                        #     Parent Loop BB0_32 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movq	%rbp, %rdi
	xorl	%esi, %esi
	movq	16(%rsp), %rdx                  # 8-byte Reload
	callq	memset
	addq	$17600, %rbp                    # imm = 0x44C0
	addq	$1, %r12
	cmpq	%r15, %r12
	jb	.LBB0_34
	jmp	.LBB0_35
	.p2align	4, 0x90
.LBB0_37:                               # %for.body35.i.preheader
                                        #   in Loop: Header=BB0_17 Depth=1
	movq	16(%rsp), %rax                  # 8-byte Reload
	cmpq	%rax, 40(%rsp)                  # 8-byte Folded Reload
	jle	.LBB0_38
# %bb.46:                               # %for.body35.i.us6.preheader
                                        #   in Loop: Header=BB0_17 Depth=1
	xorl	%ecx, %ecx
	movq	24(%rsp), %rax                  # 8-byte Reload
	jmp	.LBB0_47
	.p2align	4, 0x90
.LBB0_52:                               # %for.inc207.i.us23
                                        #   in Loop: Header=BB0_47 Depth=2
	movq	40(%rsp), %rcx                  # 8-byte Reload
	addq	$1, %rcx
	movq	32(%rsp), %r10                  # 8-byte Reload
	addq	$256, %r10                      # imm = 0x100
	movq	8(%rsp), %r11                   # 8-byte Reload
	addq	$256, %r11                      # imm = 0x100
	movq	24(%rsp), %rax                  # 8-byte Reload
	addq	$256, %rax                      # imm = 0x100
	cmpq	$69, %rcx
	je	.LBB0_65
.LBB0_47:                               # %for.body35.i.us6
                                        #   Parent Loop BB0_17 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_49 Depth 3
                                        #       Child Loop BB0_51 Depth 3
	movq	%r10, 32(%rsp)                  # 8-byte Spill
	movq	%r11, 8(%rsp)                   # 8-byte Spill
	movq	%rax, 24(%rsp)                  # 8-byte Spill
	movq	%rcx, 40(%rsp)                  # 8-byte Spill
	movq	%rcx, %rax
	shlq	$5, %rax
	leaq	31(%rax), %rdi
	cmpq	$2199, %rdi                     # imm = 0x897
	movl	$2199, %ecx                     # imm = 0x897
	cmovaeq	%rcx, %rdi
	cmpq	%rax, %rdi
	movq	%rax, %rdx
	cmovaq	%rdi, %rdx
	cmpq	%rdi, %rax
	ja	.LBB0_52
# %bb.48:                               # %for.body71.i.us.preheader
                                        #   in Loop: Header=BB0_47 Depth=2
	movq	%rdi, %rsi
	movl	$1, %ecx
	subq	%rax, %rcx
	addq	%rcx, %rsi
	shlq	$3, %rsi
	movq	%rsi, 48(%rsp)                  # 8-byte Spill
	addq	%rcx, %rdx
	shlq	$3, %rdx
	movq	%rdx, 56(%rsp)                  # 8-byte Spill
	xorl	%r15d, %r15d
	movq	88(%rsp), %rbp                  # 8-byte Reload
	.p2align	4, 0x90
.LBB0_49:                               # %for.body71.i.us
                                        #   Parent Loop BB0_17 Depth=1
                                        #     Parent Loop BB0_47 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movq	8(%rsp), %rax                   # 8-byte Reload
	addq	%r15, %rax
	movq	%rax, 16(%rsp)                  # 8-byte Spill
	movq	32(%rsp), %rax                  # 8-byte Reload
	leaq	(%rax,%r15), %rdi
	xorl	%esi, %esi
	movq	48(%rsp), %r12                  # 8-byte Reload
	movq	%r12, %rdx
	callq	memset
	movq	16(%rsp), %rdi                  # 8-byte Reload
	xorl	%esi, %esi
	movq	%r12, %rdx
	callq	memset
	addq	$17600, %r15                    # imm = 0x44C0
	addq	$-1, %rbp
	jne	.LBB0_49
# %bb.50:                               # %for.body124.i.us10.preheader
                                        #   in Loop: Header=BB0_47 Depth=2
	movq	72(%rsp), %r15                  # 8-byte Reload
	movq	24(%rsp), %rbp                  # 8-byte Reload
	movq	64(%rsp), %r12                  # 8-byte Reload
	.p2align	4, 0x90
.LBB0_51:                               # %for.body124.i.us10
                                        #   Parent Loop BB0_17 Depth=1
                                        #     Parent Loop BB0_47 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movq	%rbp, %rdi
	xorl	%esi, %esi
	movq	56(%rsp), %rdx                  # 8-byte Reload
	callq	memset
	addq	$17600, %rbp                    # imm = 0x44C0
	addq	$1, %r15
	cmpq	%r12, %r15
	jb	.LBB0_51
	jmp	.LBB0_52
.LBB0_54:                               # %for.body35.i.us240.preheader88
                                        #   in Loop: Header=BB0_17 Depth=1
	movq	%r11, 8(%rsp)                   # 8-byte Spill
	xorl	%ecx, %ecx
	jmp	.LBB0_55
	.p2align	4, 0x90
.LBB0_59:                               # %for.inc207.i.us272
                                        #   in Loop: Header=BB0_55 Depth=2
	addq	$1, %rcx
	addq	$256, %r10                      # imm = 0x100
	addq	$256, 8(%rsp)                   # 8-byte Folded Spill
                                        # imm = 0x100
	cmpq	$69, %rcx
	je	.LBB0_65
.LBB0_55:                               # %for.body35.i.us240
                                        #   Parent Loop BB0_17 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_57 Depth 3
	movq	%rcx, %rdx
	shlq	$5, %rdx
	leaq	31(%rdx), %rdi
	cmpq	$2199, %rdi                     # imm = 0x897
	movl	$2199, %eax                     # imm = 0x897
	cmovaeq	%rax, %rdi
	cmpq	%rdx, %rdi
	movq	%rdx, %rsi
	cmovaq	%rdi, %rsi
	cmpq	%rdi, %rdx
	ja	.LBB0_59
# %bb.56:                               # %for.body71.i.us244.preheader
                                        #   in Loop: Header=BB0_55 Depth=2
	movq	%rdi, %rbp
	movq	%rcx, 48(%rsp)                  # 8-byte Spill
	movl	$1, %eax
	movq	%rdx, %rcx
	movq	%rdx, 24(%rsp)                  # 8-byte Spill
	subq	%rdx, %rax
	addq	%rax, %rbp
	shlq	$3, %rbp
	movq	%rbp, 56(%rsp)                  # 8-byte Spill
	addq	%rax, %rsi
	shlq	$3, %rsi
	movq	%rsi, 64(%rsp)                  # 8-byte Spill
	xorl	%r15d, %r15d
	movq	88(%rsp), %r12                  # 8-byte Reload
	movq	%r10, 32(%rsp)                  # 8-byte Spill
	.p2align	4, 0x90
.LBB0_57:                               # %for.body71.i.us244
                                        #   Parent Loop BB0_17 Depth=1
                                        #     Parent Loop BB0_55 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movq	8(%rsp), %rax                   # 8-byte Reload
	addq	%r15, %rax
	movq	%rax, 16(%rsp)                  # 8-byte Spill
	movq	32(%rsp), %rax                  # 8-byte Reload
	leaq	(%rax,%r15), %rdi
	xorl	%esi, %esi
	movq	56(%rsp), %rbp                  # 8-byte Reload
	movq	%rbp, %rdx
	callq	memset
	movq	16(%rsp), %rdi                  # 8-byte Reload
	xorl	%esi, %esi
	movq	%rbp, %rdx
	callq	memset
	addq	$17600, %r15                    # imm = 0x44C0
	addq	$-1, %r12
	jne	.LBB0_57
# %bb.58:                               # %for.inc207.loopexit285.i.us269
                                        #   in Loop: Header=BB0_55 Depth=2
	imulq	$17600, 40(%rsp), %rax          # 8-byte Folded Reload
                                        # imm = 0x44C0
	addq	80(%rsp), %rax                  # 8-byte Folded Reload
	movq	24(%rsp), %rcx                  # 8-byte Reload
	leaq	(%rax,%rcx,8), %rdi
	xorl	%esi, %esi
	movq	64(%rsp), %rdx                  # 8-byte Reload
	callq	memset
	movq	32(%rsp), %r10                  # 8-byte Reload
	movq	48(%rsp), %rcx                  # 8-byte Reload
	jmp	.LBB0_59
.LBB0_25:                               # %for.body35.i.us.preheader85
                                        #   in Loop: Header=BB0_17 Depth=1
	xorl	%edx, %edx
	movq	24(%rsp), %rbp                  # 8-byte Reload
	jmp	.LBB0_26
	.p2align	4, 0x90
.LBB0_30:                               # %for.inc207.i.us
                                        #   in Loop: Header=BB0_26 Depth=2
	addq	$1, %rdx
	addq	$256, %rbp                      # imm = 0x100
	cmpq	$69, %rdx
	je	.LBB0_65
.LBB0_26:                               # %for.body35.i.us
                                        #   Parent Loop BB0_17 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_28 Depth 3
	movq	%rdx, %rsi
	shlq	$5, %rsi
	leaq	31(%rsi), %rcx
	cmpq	$2199, %rcx                     # imm = 0x897
	movl	$2199, %eax                     # imm = 0x897
	cmovaeq	%rax, %rcx
	cmpq	%rsi, %rcx
	movq	%rsi, %rax
	cmovaq	%rcx, %rax
	cmpq	%rcx, %rsi
	ja	.LBB0_30
# %bb.27:                               # %for.body124.i.us.preheader
                                        #   in Loop: Header=BB0_26 Depth=2
	movq	%rdx, 8(%rsp)                   # 8-byte Spill
	movq	%rsi, %rcx
	movq	%rsi, 32(%rsp)                  # 8-byte Spill
	subq	%rsi, %rax
	leaq	8(,%rax,8), %rax
	movq	%rax, 16(%rsp)                  # 8-byte Spill
	movq	72(%rsp), %r12                  # 8-byte Reload
	movq	%rbp, 24(%rsp)                  # 8-byte Spill
	movq	64(%rsp), %r15                  # 8-byte Reload
	.p2align	4, 0x90
.LBB0_28:                               # %for.body124.i.us
                                        #   Parent Loop BB0_17 Depth=1
                                        #     Parent Loop BB0_26 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movq	%rbp, %rdi
	xorl	%esi, %esi
	movq	16(%rsp), %rdx                  # 8-byte Reload
	callq	memset
	addq	$17600, %rbp                    # imm = 0x44C0
	addq	$1, %r12
	cmpq	%r15, %r12
	jb	.LBB0_28
# %bb.29:                               # %for.inc207.loopexit285.i.us
                                        #   in Loop: Header=BB0_26 Depth=2
	imulq	$17600, 40(%rsp), %rax          # 8-byte Folded Reload
                                        # imm = 0x44C0
	addq	80(%rsp), %rax                  # 8-byte Folded Reload
	movq	32(%rsp), %rcx                  # 8-byte Reload
	leaq	(%rax,%rcx,8), %rdi
	xorl	%esi, %esi
	movq	16(%rsp), %rdx                  # 8-byte Reload
	callq	memset
	movq	24(%rsp), %rbp                  # 8-byte Reload
	movq	8(%rsp), %rdx                   # 8-byte Reload
	jmp	.LBB0_30
.LBB0_38:                               # %for.body35.i.preheader91
                                        #   in Loop: Header=BB0_17 Depth=1
	xorl	%edx, %edx
	movq	24(%rsp), %rcx                  # 8-byte Reload
	jmp	.LBB0_39
	.p2align	4, 0x90
.LBB0_45:                               # %for.inc207.i
                                        #   in Loop: Header=BB0_39 Depth=2
	addq	$1, %rdx
	addq	$256, %r10                      # imm = 0x100
	addq	$256, %r11                      # imm = 0x100
	addq	$256, %rcx                      # imm = 0x100
	cmpq	$69, %rdx
	je	.LBB0_65
.LBB0_39:                               # %for.body35.i
                                        #   Parent Loop BB0_17 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_41 Depth 3
                                        #       Child Loop BB0_43 Depth 3
	movq	%rdx, %rsi
	shlq	$5, %rsi
	leaq	31(%rsi), %rbp
	cmpq	$2199, %rbp                     # imm = 0x897
	movl	$2199, %eax                     # imm = 0x897
	cmovaeq	%rax, %rbp
	cmpq	%rsi, %rbp
	movq	%rsi, %rdi
	cmovaq	%rbp, %rdi
	cmpq	%rbp, %rsi
	ja	.LBB0_45
# %bb.40:                               # %for.body71.i.preheader
                                        #   in Loop: Header=BB0_39 Depth=2
	movq	%rbp, %r8
	movq	%rdx, 104(%rsp)                 # 8-byte Spill
	movq	%rcx, 24(%rsp)                  # 8-byte Spill
	movl	$1, %eax
	movq	%rsi, %rcx
	movq	%rsi, 136(%rsp)                 # 8-byte Spill
	subq	%rsi, %rax
	addq	%rax, %r8
	shlq	$3, %r8
	movq	%r8, 48(%rsp)                   # 8-byte Spill
	addq	%rax, %rdi
	shlq	$3, %rdi
	movq	%rdi, 56(%rsp)                  # 8-byte Spill
	xorl	%r15d, %r15d
	movq	88(%rsp), %r12                  # 8-byte Reload
	movq	%r11, 8(%rsp)                   # 8-byte Spill
	movq	%r10, 32(%rsp)                  # 8-byte Spill
	.p2align	4, 0x90
.LBB0_41:                               # %for.body71.i
                                        #   Parent Loop BB0_17 Depth=1
                                        #     Parent Loop BB0_39 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	leaq	(%r11,%r15), %rax
	movq	%rax, 16(%rsp)                  # 8-byte Spill
	leaq	(%r10,%r15), %rdi
	xorl	%esi, %esi
	movq	48(%rsp), %rbp                  # 8-byte Reload
	movq	%rbp, %rdx
	callq	memset
	movq	16(%rsp), %rdi                  # 8-byte Reload
	xorl	%esi, %esi
	movq	%rbp, %rdx
	callq	memset
	movq	32(%rsp), %r10                  # 8-byte Reload
	movq	8(%rsp), %r11                   # 8-byte Reload
	addq	$17600, %r15                    # imm = 0x44C0
	addq	$-1, %r12
	jne	.LBB0_41
# %bb.42:                               # %for.body124.i.preheader
                                        #   in Loop: Header=BB0_39 Depth=2
	movq	72(%rsp), %r15                  # 8-byte Reload
	movq	24(%rsp), %rbp                  # 8-byte Reload
	movq	64(%rsp), %r12                  # 8-byte Reload
	.p2align	4, 0x90
.LBB0_43:                               # %for.body124.i
                                        #   Parent Loop BB0_17 Depth=1
                                        #     Parent Loop BB0_39 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movq	%rbp, %rdi
	xorl	%esi, %esi
	movq	56(%rsp), %rdx                  # 8-byte Reload
	callq	memset
	addq	$17600, %rbp                    # imm = 0x44C0
	addq	$1, %r15
	cmpq	%r12, %r15
	jb	.LBB0_43
# %bb.44:                               # %for.inc207.loopexit285.i
                                        #   in Loop: Header=BB0_39 Depth=2
	imulq	$17600, 40(%rsp), %rax          # 8-byte Folded Reload
                                        # imm = 0x44C0
	addq	80(%rsp), %rax                  # 8-byte Folded Reload
	movq	136(%rsp), %rcx                 # 8-byte Reload
	leaq	(%rax,%rcx,8), %rdi
	xorl	%esi, %esi
	movq	56(%rsp), %rdx                  # 8-byte Reload
	callq	memset
	movq	24(%rsp), %rcx                  # 8-byte Reload
	movq	8(%rsp), %r11                   # 8-byte Reload
	movq	32(%rsp), %r10                  # 8-byte Reload
	movq	104(%rsp), %rdx                 # 8-byte Reload
	jmp	.LBB0_45
.LBB0_66:                               # %for.cond437.preheader.i.preheader
	movl	$31, %eax
	xorl	%ecx, %ecx
	movq	96(%rsp), %rdx                  # 8-byte Reload
	movq	%rdx, 72(%rsp)                  # 8-byte Spill
	jmp	.LBB0_67
	.p2align	4, 0x90
.LBB0_79:                               # %for.inc557.i
                                        #   in Loop: Header=BB0_67 Depth=1
	movq	64(%rsp), %rcx                  # 8-byte Reload
	addq	$1, %rcx
	movq	40(%rsp), %rax                  # 8-byte Reload
	addq	$32, %rax
	addq	$563200, 72(%rsp)               # 8-byte Folded Spill
                                        # imm = 0x89800
	cmpq	$57, %rcx
	je	.LBB0_80
.LBB0_67:                               # %for.cond437.preheader.i
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_69 Depth 2
                                        #       Child Loop BB0_71 Depth 3
                                        #         Child Loop BB0_72 Depth 4
                                        #           Child Loop BB0_73 Depth 5
                                        #             Child Loop BB0_74 Depth 6
	cmpq	$1799, %rax                     # imm = 0x707
	movl	$1799, %ebp                     # imm = 0x707
	movq	%rax, 40(%rsp)                  # 8-byte Spill
	cmovbq	%rax, %rbp
	movq	%rcx, 64(%rsp)                  # 8-byte Spill
	movq	%rcx, %rdx
	shlq	$5, %rdx
	leaq	31(%rdx), %rax
	cmpq	$1799, %rax                     # imm = 0x707
	movl	$1799, %ecx                     # imm = 0x707
	cmovaeq	%rcx, %rax
	movq	%rdx, 8(%rsp)                   # 8-byte Spill
	cmpq	%rax, %rdx
	ja	.LBB0_79
# %bb.68:                               # %for.cond457.preheader.i.preheader
                                        #   in Loop: Header=BB0_67 Depth=1
	movl	$31, %eax
	movl	$1, %ecx
	movq	%rcx, 48(%rsp)                  # 8-byte Spill
	movq	112(%rsp), %rcx                 # 8-byte Reload
	movq	%rcx, 24(%rsp)                  # 8-byte Spill
	movq	72(%rsp), %rcx                  # 8-byte Reload
	movq	%rcx, 16(%rsp)                  # 8-byte Spill
	xorl	%ecx, %ecx
	jmp	.LBB0_69
	.p2align	4, 0x90
.LBB0_78:                               # %for.inc554.i
                                        #   in Loop: Header=BB0_69 Depth=2
	movq	56(%rsp), %rcx                  # 8-byte Reload
	addq	$1, %rcx
	movq	32(%rsp), %rax                  # 8-byte Reload
	addq	$32, %rax
	addq	$-32, 48(%rsp)                  # 8-byte Folded Spill
	addq	$256, 16(%rsp)                  # 8-byte Folded Spill
                                        # imm = 0x100
	addq	$256, 24(%rsp)                  # 8-byte Folded Spill
                                        # imm = 0x100
	cmpq	$69, %rcx
	je	.LBB0_79
.LBB0_69:                               # %for.cond457.preheader.i
                                        #   Parent Loop BB0_67 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_71 Depth 3
                                        #         Child Loop BB0_72 Depth 4
                                        #           Child Loop BB0_73 Depth 5
                                        #             Child Loop BB0_74 Depth 6
	cmpq	$2199, %rax                     # imm = 0x897
	movl	$2199, %r8d                     # imm = 0x897
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	cmovbq	%rax, %r8
	movq	%rcx, 56(%rsp)                  # 8-byte Spill
	movq	%rcx, %rax
	shlq	$5, %rax
	leaq	31(%rax), %rcx
	cmpq	$2199, %rcx                     # imm = 0x897
	movl	$2199, %edx                     # imm = 0x897
	cmovaeq	%rdx, %rcx
	cmpq	%rcx, %rax
	ja	.LBB0_78
# %bb.70:                               # %for.body476.i.preheader
                                        #   in Loop: Header=BB0_69 Depth=2
	addq	48(%rsp), %r8                   # 8-byte Folded Reload
	movl	$32, %r9d
	movq	24(%rsp), %rdi                  # 8-byte Reload
	xorl	%r12d, %r12d
	.p2align	4, 0x90
.LBB0_71:                               # %for.body476.i
                                        #   Parent Loop BB0_67 Depth=1
                                        #     Parent Loop BB0_69 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_72 Depth 4
                                        #           Child Loop BB0_73 Depth 5
                                        #             Child Loop BB0_74 Depth 6
	movq	%r12, %r15
	shlq	$5, %r15
	movq	16(%rsp), %rsi                  # 8-byte Reload
	movq	8(%rsp), %rcx                   # 8-byte Reload
	.p2align	4, 0x90
.LBB0_72:                               # %for.body495.i
                                        #   Parent Loop BB0_67 Depth=1
                                        #     Parent Loop BB0_69 Depth=2
                                        #       Parent Loop BB0_71 Depth=3
                                        # =>      This Loop Header: Depth=4
                                        #           Child Loop BB0_73 Depth 5
                                        #             Child Loop BB0_74 Depth 6
	movq	%rdi, %rdx
	movq	%r15, %r11
	.p2align	4, 0x90
.LBB0_73:                               # %for.body514.i
                                        #   Parent Loop BB0_67 Depth=1
                                        #     Parent Loop BB0_69 Depth=2
                                        #       Parent Loop BB0_71 Depth=3
                                        #         Parent Loop BB0_72 Depth=4
                                        # =>        This Loop Header: Depth=5
                                        #             Child Loop BB0_74 Depth 6
	imulq	$19200, %rcx, %rax              # imm = 0x4B00
	addq	%rbx, %rax
	movsd	(%rax,%r11,8), %xmm0            # xmm0 = mem[0],zero
	xorl	%r10d, %r10d
	.p2align	4, 0x90
.LBB0_74:                               # %for.body533.i
                                        #   Parent Loop BB0_67 Depth=1
                                        #     Parent Loop BB0_69 Depth=2
                                        #       Parent Loop BB0_71 Depth=3
                                        #         Parent Loop BB0_72 Depth=4
                                        #           Parent Loop BB0_73 Depth=5
                                        # =>          This Inner Loop Header: Depth=6
	movsd	(%rdx,%r10,8), %xmm1            # xmm1 = mem[0],zero
	mulsd	%xmm0, %xmm1
	addsd	(%rsi,%r10,8), %xmm1
	movsd	%xmm1, (%rsi,%r10,8)
	addq	$1, %r10
	cmpq	%r10, %r8
	jne	.LBB0_74
# %bb.75:                               # %for.inc545.i
                                        #   in Loop: Header=BB0_73 Depth=5
	addq	$1, %r11
	addq	$17600, %rdx                    # imm = 0x44C0
	cmpq	%r9, %r11
	jne	.LBB0_73
# %bb.76:                               # %for.inc548.i
                                        #   in Loop: Header=BB0_72 Depth=4
	leaq	1(%rcx), %rax
	addq	$17600, %rsi                    # imm = 0x44C0
	cmpq	%rbp, %rcx
	movq	%rax, %rcx
	jne	.LBB0_72
# %bb.77:                               # %for.inc551.i
                                        #   in Loop: Header=BB0_71 Depth=3
	addq	$1, %r12
	addq	$32, %r9
	addq	$563200, %rdi                   # imm = 0x89800
	cmpq	$75, %r12
	jne	.LBB0_71
	jmp	.LBB0_78
.LBB0_80:                               # %for.body603.preheader.i.preheader
	xorl	%eax, %eax
	movq	%r14, %r15
	jmp	.LBB0_81
	.p2align	4, 0x90
.LBB0_86:                               # %for.inc653.i
                                        #   in Loop: Header=BB0_81 Depth=1
	movq	8(%rsp), %rax                   # 8-byte Reload
	addq	$1, %rax
	movq	32(%rsp), %r15                  # 8-byte Reload
	addq	$460800, %r15                   # imm = 0x70800
	cmpq	$50, %rax
	je	.LBB0_87
.LBB0_81:                               # %for.body603.preheader.i
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_82 Depth 2
                                        #       Child Loop BB0_84 Depth 3
	movq	%rax, 8(%rsp)                   # 8-byte Spill
	movq	%r15, 32(%rsp)                  # 8-byte Spill
	xorl	%eax, %eax
	jmp	.LBB0_82
	.p2align	4, 0x90
.LBB0_85:                               # %for.inc650.i
                                        #   in Loop: Header=BB0_82 Depth=2
	movq	16(%rsp), %rax                  # 8-byte Reload
	addq	$1, %rax
	addq	$256, %r15                      # imm = 0x100
	cmpq	$57, %rax
	je	.LBB0_86
.LBB0_82:                               # %for.body603.i
                                        #   Parent Loop BB0_81 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_84 Depth 3
	movq	%rax, 16(%rsp)                  # 8-byte Spill
	shlq	$5, %rax
	leaq	31(%rax), %rdx
	cmpq	$1799, %rdx                     # imm = 0x707
	movl	$1799, %ecx                     # imm = 0x707
	cmovaeq	%rcx, %rdx
	cmpq	%rax, %rdx
	movq	%rax, %rcx
	cmovaq	%rdx, %rcx
	cmpq	%rdx, %rax
	ja	.LBB0_85
# %bb.83:                               # %for.body622.i.preheader
                                        #   in Loop: Header=BB0_82 Depth=2
	subq	%rax, %rcx
	leaq	8(,%rcx,8), %rbp
	xorl	%r12d, %r12d
	.p2align	4, 0x90
.LBB0_84:                               # %for.body622.i
                                        #   Parent Loop BB0_81 Depth=1
                                        #     Parent Loop BB0_82 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	leaq	(%r15,%r12), %rdi
	xorl	%esi, %esi
	movq	%rbp, %rdx
	callq	memset
	addq	$14400, %r12                    # imm = 0x3840
	cmpq	$460800, %r12                   # imm = 0x70800
	jne	.LBB0_84
	jmp	.LBB0_85
.LBB0_87:                               # %for.cond680.preheader.i.preheader
	movl	$32, %r15d
	xorl	%eax, %eax
	movq	80(%rsp), %rcx                  # 8-byte Reload
	movq	%rcx, 104(%rsp)                 # 8-byte Spill
	movq	%r14, %rcx
	jmp	.LBB0_88
	.p2align	4, 0x90
.LBB0_109:                              # %for.inc905.i
                                        #   in Loop: Header=BB0_88 Depth=1
	movq	136(%rsp), %rax                 # 8-byte Reload
	addq	$1, %rax
	addq	$32, %r15
	movq	128(%rsp), %rcx                 # 8-byte Reload
	addq	$460800, %rcx                   # imm = 0x70800
	addq	$563200, 104(%rsp)              # 8-byte Folded Spill
                                        # imm = 0x89800
	cmpq	$50, %rax
	je	.LBB0_110
.LBB0_88:                               # %for.cond680.preheader.i
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_89 Depth 2
                                        #       Child Loop BB0_91 Depth 3
                                        #         Child Loop BB0_93 Depth 4
                                        #           Child Loop BB0_94 Depth 5
                                        #             Child Loop BB0_95 Depth 6
                                        #       Child Loop BB0_100 Depth 3
                                        #         Child Loop BB0_102 Depth 4
                                        #           Child Loop BB0_103 Depth 5
                                        #             Child Loop BB0_104 Depth 6
	movq	%rax, 136(%rsp)                 # 8-byte Spill
	shlq	$5, %rax
	movq	%rax, 56(%rsp)                  # 8-byte Spill
	movl	$31, %eax
	movl	$1, %edx
	movq	%rdx, 64(%rsp)                  # 8-byte Spill
	movq	96(%rsp), %rdx                  # 8-byte Reload
	movq	%rdx, 72(%rsp)                  # 8-byte Spill
	movq	120(%rsp), %rdx                 # 8-byte Reload
	movq	%rdx, 88(%rsp)                  # 8-byte Spill
	movq	%rcx, 128(%rsp)                 # 8-byte Spill
	movq	%rcx, 48(%rsp)                  # 8-byte Spill
	xorl	%ecx, %ecx
	jmp	.LBB0_89
	.p2align	4, 0x90
.LBB0_108:                              # %for.inc902.i
                                        #   in Loop: Header=BB0_89 Depth=2
	movq	40(%rsp), %rcx                  # 8-byte Reload
	addq	$1, %rcx
	movq	24(%rsp), %rax                  # 8-byte Reload
	addq	$32, %rax
	addq	$-32, 64(%rsp)                  # 8-byte Folded Spill
	addq	$256, 48(%rsp)                  # 8-byte Folded Spill
                                        # imm = 0x100
	addq	$256, 88(%rsp)                  # 8-byte Folded Spill
                                        # imm = 0x100
	addq	$563200, 72(%rsp)               # 8-byte Folded Spill
                                        # imm = 0x89800
	cmpq	$57, %rcx
	je	.LBB0_109
.LBB0_89:                               # %for.cond703.preheader.i
                                        #   Parent Loop BB0_88 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_91 Depth 3
                                        #         Child Loop BB0_93 Depth 4
                                        #           Child Loop BB0_94 Depth 5
                                        #             Child Loop BB0_95 Depth 6
                                        #       Child Loop BB0_100 Depth 3
                                        #         Child Loop BB0_102 Depth 4
                                        #           Child Loop BB0_103 Depth 5
                                        #             Child Loop BB0_104 Depth 6
	cmpq	$1799, %rax                     # imm = 0x707
	movl	$1799, %r10d                    # imm = 0x707
	movq	%rax, 24(%rsp)                  # 8-byte Spill
	cmovbq	%rax, %r10
	movq	%rcx, 40(%rsp)                  # 8-byte Spill
	movq	%rcx, %rdx
	shlq	$5, %rdx
	leaq	31(%rdx), %rax
	cmpq	$1799, %rax                     # imm = 0x707
	movl	$1799, %ecx                     # imm = 0x707
	cmovaeq	%rcx, %rax
	movq	%rdx, 16(%rsp)                  # 8-byte Spill
	cmpq	%rax, %rdx
	ja	.LBB0_108
# %bb.90:                               # %for.body741.lr.ph.i.preheader
                                        #   in Loop: Header=BB0_89 Depth=2
	movq	64(%rsp), %rax                  # 8-byte Reload
	leaq	(%r10,%rax), %rsi
	movl	$31, %eax
	movq	88(%rsp), %r8                   # 8-byte Reload
	xorl	%r9d, %r9d
	jmp	.LBB0_91
	.p2align	4, 0x90
.LBB0_98:                               # %for.inc797.i
                                        #   in Loop: Header=BB0_91 Depth=3
	addq	$1, %r9
	movq	8(%rsp), %rax                   # 8-byte Reload
	addq	$32, %rax
	addq	$460800, %r8                    # imm = 0x70800
	cmpq	$63, %r9
	je	.LBB0_99
.LBB0_91:                               # %for.body741.lr.ph.i
                                        #   Parent Loop BB0_88 Depth=1
                                        #     Parent Loop BB0_89 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_93 Depth 4
                                        #           Child Loop BB0_94 Depth 5
                                        #             Child Loop BB0_95 Depth 6
	cmpq	$1999, %rax                     # imm = 0x7CF
	movl	$1999, %ecx                     # imm = 0x7CF
	movq	%rax, 8(%rsp)                   # 8-byte Spill
	cmovbq	%rax, %rcx
	movq	%r9, %r12
	shlq	$5, %r12
	leaq	31(%r12), %rax
	cmpq	$1999, %rax                     # imm = 0x7CF
	movl	$1999, %edx                     # imm = 0x7CF
	cmovaeq	%rdx, %rax
	cmpq	%rax, %r12
	ja	.LBB0_98
# %bb.92:                               # %for.body741.i.preheader
                                        #   in Loop: Header=BB0_91 Depth=3
	movq	48(%rsp), %rdi                  # 8-byte Reload
	movq	56(%rsp), %r11                  # 8-byte Reload
	.p2align	4, 0x90
.LBB0_93:                               # %for.body741.i
                                        #   Parent Loop BB0_88 Depth=1
                                        #     Parent Loop BB0_89 Depth=2
                                        #       Parent Loop BB0_91 Depth=3
                                        # =>      This Loop Header: Depth=4
                                        #           Child Loop BB0_94 Depth 5
                                        #             Child Loop BB0_95 Depth 6
	movq	%r8, %rbp
	movq	%r12, %rdx
	.p2align	4, 0x90
.LBB0_94:                               # %for.body760.i
                                        #   Parent Loop BB0_88 Depth=1
                                        #     Parent Loop BB0_89 Depth=2
                                        #       Parent Loop BB0_91 Depth=3
                                        #         Parent Loop BB0_93 Depth=4
                                        # =>        This Loop Header: Depth=5
                                        #             Child Loop BB0_95 Depth 6
	imulq	$16000, %r11, %rax              # imm = 0x3E80
	addq	%r13, %rax
	movsd	(%rax,%rdx,8), %xmm0            # xmm0 = mem[0],zero
	xorl	%eax, %eax
	.p2align	4, 0x90
.LBB0_95:                               # %for.body779.i
                                        #   Parent Loop BB0_88 Depth=1
                                        #     Parent Loop BB0_89 Depth=2
                                        #       Parent Loop BB0_91 Depth=3
                                        #         Parent Loop BB0_93 Depth=4
                                        #           Parent Loop BB0_94 Depth=5
                                        # =>          This Inner Loop Header: Depth=6
	movsd	(%rbp,%rax,8), %xmm1            # xmm1 = mem[0],zero
	mulsd	%xmm0, %xmm1
	addsd	(%rdi,%rax,8), %xmm1
	movsd	%xmm1, (%rdi,%rax,8)
	addq	$1, %rax
	cmpq	%rax, %rsi
	jne	.LBB0_95
# %bb.96:                               # %for.inc791.i
                                        #   in Loop: Header=BB0_94 Depth=5
	leaq	1(%rdx), %rax
	addq	$14400, %rbp                    # imm = 0x3840
	cmpq	%rcx, %rdx
	movq	%rax, %rdx
	jne	.LBB0_94
# %bb.97:                               # %for.inc794.i
                                        #   in Loop: Header=BB0_93 Depth=4
	addq	$1, %r11
	addq	$14400, %rdi                    # imm = 0x3840
	cmpq	%r15, %r11
	jne	.LBB0_93
	jmp	.LBB0_98
	.p2align	4, 0x90
.LBB0_99:                               # %for.body842.lr.ph.i.preheader
                                        #   in Loop: Header=BB0_89 Depth=2
	movl	$31, %r11d
	movl	$1, %eax
	movq	%rax, 8(%rsp)                   # 8-byte Spill
	movq	72(%rsp), %rdi                  # 8-byte Reload
	movq	104(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	xorl	%r8d, %r8d
	jmp	.LBB0_100
	.p2align	4, 0x90
.LBB0_107:                              # %for.inc898.i
                                        #   in Loop: Header=BB0_100 Depth=3
	addq	$1, %r8
	addq	$32, %r11
	addq	$-32, 8(%rsp)                   # 8-byte Folded Spill
	addq	$256, 32(%rsp)                  # 8-byte Folded Spill
                                        # imm = 0x100
	addq	$256, %rdi                      # imm = 0x100
	cmpq	$69, %r8
	je	.LBB0_108
.LBB0_100:                              # %for.body842.lr.ph.i
                                        #   Parent Loop BB0_88 Depth=1
                                        #     Parent Loop BB0_89 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_102 Depth 4
                                        #           Child Loop BB0_103 Depth 5
                                        #             Child Loop BB0_104 Depth 6
	cmpq	$2199, %r11                     # imm = 0x897
	movl	$2199, %edx                     # imm = 0x897
	cmovbq	%r11, %rdx
	movq	%r8, %rcx
	shlq	$5, %rcx
	leaq	31(%rcx), %rsi
	cmpq	$2199, %rsi                     # imm = 0x897
	movl	$2199, %eax                     # imm = 0x897
	cmovaeq	%rax, %rsi
	cmpq	%rsi, %rcx
	ja	.LBB0_107
# %bb.101:                              # %for.body842.i.preheader
                                        #   in Loop: Header=BB0_100 Depth=3
	addq	8(%rsp), %rdx                   # 8-byte Folded Reload
	movq	32(%rsp), %rcx                  # 8-byte Reload
	movq	56(%rsp), %rsi                  # 8-byte Reload
	.p2align	4, 0x90
.LBB0_102:                              # %for.body842.i
                                        #   Parent Loop BB0_88 Depth=1
                                        #     Parent Loop BB0_89 Depth=2
                                        #       Parent Loop BB0_100 Depth=3
                                        # =>      This Loop Header: Depth=4
                                        #           Child Loop BB0_103 Depth 5
                                        #             Child Loop BB0_104 Depth 6
	movq	%rdi, %rbp
	movq	16(%rsp), %r9                   # 8-byte Reload
	.p2align	4, 0x90
.LBB0_103:                              # %for.body861.i
                                        #   Parent Loop BB0_88 Depth=1
                                        #     Parent Loop BB0_89 Depth=2
                                        #       Parent Loop BB0_100 Depth=3
                                        #         Parent Loop BB0_102 Depth=4
                                        # =>        This Loop Header: Depth=5
                                        #             Child Loop BB0_104 Depth 6
	imulq	$14400, %rsi, %rax              # imm = 0x3840
	addq	%r14, %rax
	movsd	(%rax,%r9,8), %xmm0             # xmm0 = mem[0],zero
	xorl	%r12d, %r12d
	.p2align	4, 0x90
.LBB0_104:                              # %for.body880.i
                                        #   Parent Loop BB0_88 Depth=1
                                        #     Parent Loop BB0_89 Depth=2
                                        #       Parent Loop BB0_100 Depth=3
                                        #         Parent Loop BB0_102 Depth=4
                                        #           Parent Loop BB0_103 Depth=5
                                        # =>          This Inner Loop Header: Depth=6
	movsd	(%rbp,%r12,8), %xmm1            # xmm1 = mem[0],zero
	mulsd	%xmm0, %xmm1
	addsd	(%rcx,%r12,8), %xmm1
	movsd	%xmm1, (%rcx,%r12,8)
	addq	$1, %r12
	cmpq	%r12, %rdx
	jne	.LBB0_104
# %bb.105:                              # %for.inc892.i
                                        #   in Loop: Header=BB0_103 Depth=5
	leaq	1(%r9), %rax
	addq	$17600, %rbp                    # imm = 0x44C0
	cmpq	%r10, %r9
	movq	%rax, %r9
	jne	.LBB0_103
# %bb.106:                              # %for.inc895.i
                                        #   in Loop: Header=BB0_102 Depth=4
	addq	$1, %rsi
	addq	$17600, %rcx                    # imm = 0x44C0
	cmpq	%r15, %rsi
	jne	.LBB0_102
	jmp	.LBB0_107
.LBB0_110:                              # %kernel_3mm.exit
	xorl	%eax, %eax
	callq	polybench_timer_stop
	xorl	%eax, %eax
	callq	polybench_timer_print
	cmpl	$43, 148(%rsp)                  # 4-byte Folded Reload
	jl	.LBB0_119
# %bb.111:                              # %if.end138
	movq	152(%rsp), %rax                 # 8-byte Reload
	movq	(%rax), %rax
	cmpb	$0, (%rax)
	je	.LBB0_112
.LBB0_119:                              # %if.end153
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
	movq	80(%rsp), %rdi                  # 8-byte Reload
	callq	free
	xorl	%eax, %eax
	addq	$200, %rsp
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
.LBB0_112:                              # %if.then151
	.cfi_def_cfa_offset 256
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
	movq	80(%rsp), %r15                  # 8-byte Reload
	xorl	%eax, %eax
.LBB0_113:                              # %for.cond2.preheader.i
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_114 Depth 2
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	movq	%rbp, 8(%rsp)                   # 8-byte Spill
	movl	%ebp, %r12d
	xorl	%ebp, %ebp
	movq	%rdx, 16(%rsp)                  # 8-byte Spill
.LBB0_114:                              # %for.body4.i
                                        #   Parent Loop BB0_113 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movl	%r12d, %eax
	movl	$3435973837, %ecx               # imm = 0xCCCCCCCD
	imulq	%rcx, %rax
	shrq	$36, %rax
	leal	(%rax,%rax,4), %eax
	leal	(%rdx,%rax,4), %eax
	cmpl	%ebp, %eax
	jne	.LBB0_116
# %bb.115:                              # %if.then.i
                                        #   in Loop: Header=BB0_114 Depth=2
	movq	stderr(%rip), %rsi
	movl	$10, %edi
	callq	fputc
.LBB0_116:                              # %if.end.i
                                        #   in Loop: Header=BB0_114 Depth=2
	movq	stderr(%rip), %rdi
	movsd	(%r15,%rbp,8), %xmm0            # xmm0 = mem[0],zero
	movl	$.L.str.5, %esi
	movb	$1, %al
	callq	fprintf
	addq	$1, %rbp
	addl	$1, %r12d
	cmpq	$2200, %rbp                     # imm = 0x898
	movq	16(%rsp), %rdx                  # 8-byte Reload
	jne	.LBB0_114
# %bb.117:                              # %for.inc10.i
                                        #   in Loop: Header=BB0_113 Depth=1
	movq	32(%rsp), %rax                  # 8-byte Reload
	addq	$1, %rax
	addq	$17600, %r15                    # imm = 0x44C0
	addl	$-1600, %edx                    # imm = 0xF9C0
	movq	8(%rsp), %rbp                   # 8-byte Reload
	addl	$1600, %ebp                     # imm = 0x640
	cmpq	$1600, %rax                     # imm = 0x640
	jne	.LBB0_113
# %bb.118:                              # %print_array.exit
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
	jmp	.LBB0_119
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

	.ident	"clang version 12.0.0 (git@github.com:wsmoses/MLIR-GPU 1112d5451cea635029a160c950f14a85f31b2258)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
