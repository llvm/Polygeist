	.text
	.file	"LLVMDialectModule"
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3                               # -- Begin function main
.LCPI0_0:
	.quad	0x4000000000000000              # double 2
.LCPI0_1:
	.quad	0x40af400000000000              # double 4000
	.text
	.globl	main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
.Lfunc_begin0:
	.file	1 "/mnt/ccnas2/bdp/rz3515/projects/polymer/example/polybench/seidel-2d-debug/seidel-2d" "<stdin>"
	.loc	1 14 0                          # <stdin>:14:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	.cfi_offset %rbx, -32
	.cfi_offset %r14, -24
	.cfi_offset %rbp, -16
	movq	%rsi, %r14
	movl	%edi, %ebp
.Ltmp0:
	.loc	1 28 11 prologue_end            # <stdin>:28:11
	movl	$128000000, %edi                # imm = 0x7A12000
	callq	malloc
	movq	%rax, %rbx
.Ltmp1:
	.loc	1 100 5                         # <stdin>:100:5
	addq	$8, %rax
	xorl	%ecx, %ecx
	movsd	.LCPI0_0(%rip), %xmm0           # xmm0 = mem[0],zero
	movsd	.LCPI0_1(%rip), %xmm1           # xmm1 = mem[0],zero
	.p2align	4, 0x90
.LBB0_1:                                # %.preheader.us.i
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_2 Depth 2
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	xorps	%xmm2, %xmm2
	cvtsi2sd	%ecx, %xmm2
	xorl	%edx, %edx
	.p2align	4, 0x90
.LBB0_2:                                #   Parent Loop BB0_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	.loc	1 110 11 is_stmt 1              # <stdin>:110:11
	leaq	2(%rdx), %rsi
	xorps	%xmm3, %xmm3
	cvtsi2sd	%esi, %xmm3
	.loc	1 111 11                        # <stdin>:111:11
	mulsd	%xmm2, %xmm3
	.loc	1 113 11                        # <stdin>:113:11
	addsd	%xmm0, %xmm3
	.loc	1 115 11                        # <stdin>:115:11
	divsd	%xmm1, %xmm3
	.loc	1 121 5                         # <stdin>:121:5
	movsd	%xmm3, -8(%rax,%rdx,8)
	.loc	1 110 11                        # <stdin>:110:11
	leal	3(%rdx), %edi
	xorps	%xmm3, %xmm3
	cvtsi2sd	%edi, %xmm3
	.loc	1 111 11                        # <stdin>:111:11
	mulsd	%xmm2, %xmm3
	.loc	1 113 11                        # <stdin>:113:11
	addsd	%xmm0, %xmm3
	.loc	1 115 11                        # <stdin>:115:11
	divsd	%xmm1, %xmm3
	.loc	1 121 5                         # <stdin>:121:5
	movsd	%xmm3, (%rax,%rdx,8)
	movq	%rsi, %rdx
	.loc	1 104 11                        # <stdin>:104:11
	cmpq	$4000, %rsi                     # imm = 0xFA0
	.loc	1 106 5                         # <stdin>:106:5
	jne	.LBB0_2
# %bb.3:                                # %._crit_edge.us.i
                                        #   in Loop: Header=BB0_1 Depth=1
	.loc	1 125 11                        # <stdin>:125:11
	addq	$1, %rcx
	.loc	1 100 5                         # <stdin>:100:5
	addq	$32000, %rax                    # imm = 0x7D00
	.loc	1 98 11                         # <stdin>:98:11
	cmpq	$4000, %rcx                     # imm = 0xFA0
	.loc	1 100 5                         # <stdin>:100:5
	jne	.LBB0_1
.Ltmp2:
# %bb.4:                                # %init_array.exit
	.loc	1 47 5                          # <stdin>:47:5
	callq	polybench_timer_start
	.loc	1 55 5                          # <stdin>:55:5
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	movl	$4000, %r9d                     # imm = 0xFA0
	movl	$1000, %edi                     # imm = 0x3E8
	movl	$4000, %esi                     # imm = 0xFA0
	movq	%rbx, %rdx
	movq	%rbx, %rcx
	xorl	%r8d, %r8d
	pushq	$1
	.cfi_adjust_cfa_offset 8
	pushq	$4000                           # imm = 0xFA0
	.cfi_adjust_cfa_offset 8
	pushq	$4000                           # imm = 0xFA0
	.cfi_adjust_cfa_offset 8
	callq	kernel_seidel_2d_new
	addq	$32, %rsp
	.cfi_adjust_cfa_offset -32
	.loc	1 56 5                          # <stdin>:56:5
	callq	polybench_timer_stop
	.loc	1 57 5                          # <stdin>:57:5
	callq	polybench_timer_print
	.loc	1 58 11                         # <stdin>:58:11
	cmpl	$43, %ebp
	.loc	1 59 5                          # <stdin>:59:5
	jl	.LBB0_7
# %bb.5:
	.loc	1 61 11                         # <stdin>:61:11
	movq	(%r14), %rax
	.loc	1 70 5                          # <stdin>:70:5
	testb	$1, (%rax)
	jne	.LBB0_7
# %bb.6:
	.loc	1 79 5                          # <stdin>:79:5
	movl	$4000, %r8d                     # imm = 0xFA0
	movl	$4000, %r9d                     # imm = 0xFA0
	movl	$4000, %edi                     # imm = 0xFA0
	movq	%rbx, %rsi
	movq	%rbx, %rdx
	xorl	%ecx, %ecx
	pushq	$1
	.cfi_adjust_cfa_offset 8
	pushq	$4000                           # imm = 0xFA0
	.cfi_adjust_cfa_offset 8
	callq	print_array
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB0_7:                                # %.critedge
	.loc	1 82 5                          # <stdin>:82:5
	xorl	%eax, %eax
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.Ltmp3:
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3                               # -- Begin function init_array
.LCPI1_0:
	.quad	0x4000000000000000              # double 2
	.text
	.globl	init_array
	.p2align	4, 0x90
	.type	init_array,@function
init_array:                             # @init_array
.Lfunc_begin1:
	.loc	1 84 0                          # <stdin>:84:0
	.cfi_startproc
# %bb.0:
	.loc	1 98 11 prologue_end            # <stdin>:98:11
	testl	%edi, %edi
	.loc	1 100 5                         # <stdin>:100:5
	jle	.LBB1_9
# %bb.1:                                # %.preheader.lr.ph
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	cvtsi2sd	%edi, %xmm0
	.loc	1 98 11 is_stmt 1               # <stdin>:98:11
	movl	%edi, %r8d
	movl	%r8d, %r10d
	andl	$-2, %r10d
	.loc	1 100 5                         # <stdin>:100:5
	leaq	8(%rdx), %r11
	xorl	%r9d, %r9d
	movsd	.LCPI1_0(%rip), %xmm1           # xmm1 = mem[0],zero
	jmp	.LBB1_2
	.p2align	4, 0x90
.LBB1_8:                                # %._crit_edge.us
                                        #   in Loop: Header=BB1_2 Depth=1
	.loc	1 125 11                        # <stdin>:125:11
	addq	$1, %r9
	.loc	1 100 5                         # <stdin>:100:5
	addq	$32000, %r11                    # imm = 0x7D00
	.loc	1 98 11                         # <stdin>:98:11
	cmpq	%r8, %r9
	.loc	1 100 5                         # <stdin>:100:5
	je	.LBB1_9
.LBB1_2:                                # %.preheader.us
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB1_5 Depth 2
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	xorps	%xmm2, %xmm2
	cvtsi2sd	%r9d, %xmm2
	cmpl	$1, %edi
	.loc	1 106 5 is_stmt 1               # <stdin>:106:5
	jne	.LBB1_4
# %bb.3:                                #   in Loop: Header=BB1_2 Depth=1
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	xorl	%ecx, %ecx
	jmp	.LBB1_6
	.p2align	4, 0x90
.LBB1_4:                                # %.preheader.us.new.preheader
                                        #   in Loop: Header=BB1_2 Depth=1
	xorl	%eax, %eax
	.p2align	4, 0x90
.LBB1_5:                                # %.preheader.us.new
                                        #   Parent Loop BB1_2 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	.loc	1 110 11 is_stmt 1              # <stdin>:110:11
	leaq	2(%rax), %rcx
	xorps	%xmm3, %xmm3
	cvtsi2sd	%ecx, %xmm3
	.loc	1 111 11                        # <stdin>:111:11
	mulsd	%xmm2, %xmm3
	.loc	1 113 11                        # <stdin>:113:11
	addsd	%xmm1, %xmm3
	.loc	1 115 11                        # <stdin>:115:11
	divsd	%xmm0, %xmm3
	.loc	1 121 5                         # <stdin>:121:5
	movsd	%xmm3, -8(%r11,%rax,8)
	.loc	1 110 11                        # <stdin>:110:11
	leal	3(%rax), %esi
	xorps	%xmm3, %xmm3
	cvtsi2sd	%esi, %xmm3
	.loc	1 111 11                        # <stdin>:111:11
	mulsd	%xmm2, %xmm3
	.loc	1 113 11                        # <stdin>:113:11
	addsd	%xmm1, %xmm3
	.loc	1 115 11                        # <stdin>:115:11
	divsd	%xmm0, %xmm3
	.loc	1 121 5                         # <stdin>:121:5
	movsd	%xmm3, (%r11,%rax,8)
	movq	%rcx, %rax
	.loc	1 106 5                         # <stdin>:106:5
	cmpq	%rcx, %r10
	jne	.LBB1_5
.LBB1_6:                                # %._crit_edge.us.unr-lcssa
                                        #   in Loop: Header=BB1_2 Depth=1
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	testb	$1, %r8b
	.loc	1 106 5                         # <stdin>:106:5
	je	.LBB1_8
# %bb.7:                                # %.epil.preheader
                                        #   in Loop: Header=BB1_2 Depth=1
	.loc	1 0 5                           # <stdin>:0:5
	imulq	$4000, %r9, %rax                # imm = 0xFA0
	.loc	1 110 11 is_stmt 1              # <stdin>:110:11
	leal	2(%rcx), %esi
	xorps	%xmm3, %xmm3
	cvtsi2sd	%esi, %xmm3
	.loc	1 111 11                        # <stdin>:111:11
	mulsd	%xmm3, %xmm2
	.loc	1 113 11                        # <stdin>:113:11
	addsd	%xmm1, %xmm2
	.loc	1 115 11                        # <stdin>:115:11
	divsd	%xmm0, %xmm2
	.loc	1 119 11                        # <stdin>:119:11
	addq	%rcx, %rax
	.loc	1 121 5                         # <stdin>:121:5
	movsd	%xmm2, (%rdx,%rax,8)
	jmp	.LBB1_8
.LBB1_9:                                # %._crit_edge2
	.loc	1 102 5                         # <stdin>:102:5
	retq
.Ltmp4:
.Lfunc_end1:
	.size	init_array, .Lfunc_end1-init_array
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3                               # -- Begin function kernel_seidel_2d
.LCPI2_0:
	.quad	0x4022000000000000              # double 9
	.text
	.globl	kernel_seidel_2d
	.p2align	4, 0x90
	.type	kernel_seidel_2d,@function
kernel_seidel_2d:                       # @kernel_seidel_2d
.Lfunc_begin2:
	.loc	1 129 0                         # <stdin>:129:0
	.cfi_startproc
# %bb.0:
	pushq	%r14
	.cfi_def_cfa_offset 16
	pushq	%rbx
	.cfi_def_cfa_offset 24
	.cfi_offset %rbx, -24
	.cfi_offset %r14, -16
	.loc	1 145 11 prologue_end           # <stdin>:145:11
	testl	%edi, %edi
	.loc	1 146 5                         # <stdin>:146:5
	jle	.LBB2_8
# %bb.1:                                # %.lr.ph4
	.loc	1 0 0 is_stmt 0                 # <stdin>:0:0
	movslq	%edi, %r9
	.loc	1 142 11 is_stmt 1              # <stdin>:142:11
	movslq	%esi, %rdx
	leaq	-1(%rdx), %r11
	.loc	1 146 5                         # <stdin>:146:5
	leaq	64016(%rcx), %r8
	addq	$-2, %rdx
	xorl	%r10d, %r10d
	movsd	.LCPI2_0(%rip), %xmm0           # xmm0 = mem[0],zero
	jmp	.LBB2_2
	.p2align	4, 0x90
.LBB2_7:                                # %._crit_edge2
                                        #   in Loop: Header=BB2_2 Depth=1
	.loc	1 171 11                        # <stdin>:171:11
	addq	$1, %r10
	.loc	1 145 11                        # <stdin>:145:11
	cmpq	%r9, %r10
	.loc	1 146 5                         # <stdin>:146:5
	je	.LBB2_8
.LBB2_2:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB2_4 Depth 2
                                        #       Child Loop BB2_5 Depth 3
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	cmpl	$3, %esi
	.loc	1 152 5 is_stmt 1               # <stdin>:152:5
	jl	.LBB2_7
# %bb.3:                                # %.preheader.us.preheader
                                        #   in Loop: Header=BB2_2 Depth=1
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movl	$1, %r14d
	movq	%r8, %rdi
	xorl	%ebx, %ebx
	.p2align	4, 0x90
.LBB2_4:                                # %.preheader.us
                                        #   Parent Loop BB2_2 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB2_5 Depth 3
	imulq	$32000, %rbx, %rax              # imm = 0x7D00
	movsd	32000(%rax,%rcx), %xmm1         # xmm1 = mem[0],zero
	xorl	%eax, %eax
	.p2align	4, 0x90
.LBB2_5:                                #   Parent Loop BB2_2 Depth=1
                                        #     Parent Loop BB2_4 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
.Ltmp5:
	.loc	1 276 11 is_stmt 1              # <stdin>:276:11
	movsd	-64016(%rdi,%rax,8), %xmm2      # xmm2 = mem[0],zero
	.loc	1 283 11                        # <stdin>:283:11
	addsd	-64008(%rdi,%rax,8), %xmm2
	.loc	1 291 11                        # <stdin>:291:11
	addsd	-64000(%rdi,%rax,8), %xmm2
	.loc	1 298 11                        # <stdin>:298:11
	addsd	%xmm1, %xmm2
	.loc	1 305 11                        # <stdin>:305:11
	addsd	-32008(%rdi,%rax,8), %xmm2
	.loc	1 312 11                        # <stdin>:312:11
	addsd	-32000(%rdi,%rax,8), %xmm2
	.loc	1 320 11                        # <stdin>:320:11
	addsd	-16(%rdi,%rax,8), %xmm2
	.loc	1 327 11                        # <stdin>:327:11
	addsd	-8(%rdi,%rax,8), %xmm2
	.loc	1 334 11                        # <stdin>:334:11
	addsd	(%rdi,%rax,8), %xmm2
	.loc	1 335 11                        # <stdin>:335:11
	divsd	%xmm0, %xmm2
	.loc	1 341 5                         # <stdin>:341:5
	movsd	%xmm2, -32008(%rdi,%rax,8)
.Ltmp6:
	.loc	1 154 11                        # <stdin>:154:11
	addq	$1, %rax
	movapd	%xmm2, %xmm1
	cmpq	%rax, %rdx
	.loc	1 155 5                         # <stdin>:155:5
	jne	.LBB2_5
# %bb.6:                                # %._crit_edge.us
                                        #   in Loop: Header=BB2_4 Depth=2
	.loc	1 168 11                        # <stdin>:168:11
	addq	$1, %r14
	.loc	1 152 5                         # <stdin>:152:5
	addq	$1, %rbx
	addq	$32000, %rdi                    # imm = 0x7D00
	.loc	1 151 11                        # <stdin>:151:11
	cmpq	%r11, %r14
	.loc	1 152 5                         # <stdin>:152:5
	jne	.LBB2_4
	jmp	.LBB2_7
.LBB2_8:                                # %._crit_edge5
	.loc	1 174 5                         # <stdin>:174:5
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	retq
.Ltmp7:
.Lfunc_end2:
	.size	kernel_seidel_2d, .Lfunc_end2-kernel_seidel_2d
	.cfi_endproc
                                        # -- End function
	.globl	print_array                     # -- Begin function print_array
	.p2align	4, 0x90
	.type	print_array,@function
print_array:                            # @print_array
.Lfunc_begin3:
	.loc	1 178 0                         # <stdin>:178:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r12
	.cfi_def_cfa_offset 40
	pushq	%rbx
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -48
	.cfi_offset %r12, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rdx, %r15
	movl	%edi, %ebp
.Ltmp8:
	.loc	1 191 11 prologue_end           # <stdin>:191:11
	movq	stderr(%rip), %rcx
	.loc	1 195 11                        # <stdin>:195:11
	movl	$str1, %edi
	movl	$22, %esi
	movl	$1, %edx
	callq	fwrite
	.loc	1 197 11                        # <stdin>:197:11
	movq	stderr(%rip), %rdi
	.loc	1 202 11                        # <stdin>:202:11
	movl	$str2, %esi
	movl	$str3, %edx
	xorl	%eax, %eax
	callq	fprintf
	.loc	1 205 11                        # <stdin>:205:11
	testl	%ebp, %ebp
	.loc	1 207 5                         # <stdin>:207:5
	jle	.LBB3_7
# %bb.1:                                # %.preheader.us.preheader
	.loc	1 205 11                        # <stdin>:205:11
	movl	%ebp, %r14d
	xorl	%ebp, %ebp
	xorl	%r12d, %r12d
.LBB3_2:                                # %.preheader.us
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB3_3 Depth 2
	.loc	1 0 11 is_stmt 0                # <stdin>:0:11
	xorl	%ebx, %ebx
.LBB3_3:                                #   Parent Loop BB3_2 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	.loc	1 229 11 is_stmt 1              # <stdin>:229:11
	leal	(%rbx,%rbp), %eax
	.loc	1 230 11                        # <stdin>:230:11
	imull	$-858993459, %eax, %eax         # imm = 0xCCCCCCCD
	addl	$429496728, %eax                # imm = 0x19999998
	rorl	$2, %eax
	cmpl	$214748364, %eax                # imm = 0xCCCCCCC
	.loc	1 231 5                         # <stdin>:231:5
	ja	.LBB3_5
# %bb.4:                                #   in Loop: Header=BB3_3 Depth=2
	.loc	1 234 11                        # <stdin>:234:11
	movq	stderr(%rip), %rsi
	.loc	1 237 11                        # <stdin>:237:11
	movl	$10, %edi
	callq	fputc
.LBB3_5:                                #   in Loop: Header=BB3_3 Depth=2
	.loc	1 241 11                        # <stdin>:241:11
	movq	stderr(%rip), %rdi
	.loc	1 249 11                        # <stdin>:249:11
	movsd	(%r15,%rbx,8), %xmm0            # xmm0 = mem[0],zero
	.loc	1 250 11                        # <stdin>:250:11
	movl	$str5, %esi
	movb	$1, %al
	callq	fprintf
	.loc	1 251 11                        # <stdin>:251:11
	addq	$1, %rbx
	.loc	1 223 11                        # <stdin>:223:11
	cmpq	%rbx, %r14
	.loc	1 225 5                         # <stdin>:225:5
	jne	.LBB3_3
# %bb.6:                                # %._crit_edge.us
                                        #   in Loop: Header=BB3_2 Depth=1
	.loc	1 254 11                        # <stdin>:254:11
	addq	$1, %r12
	.loc	1 207 5                         # <stdin>:207:5
	addq	$32000, %r15                    # imm = 0x7D00
	addq	%r14, %rbp
	.loc	1 205 11                        # <stdin>:205:11
	cmpq	%r14, %r12
	.loc	1 207 5                         # <stdin>:207:5
	jne	.LBB3_2
.LBB3_7:                                # %._crit_edge2
	.loc	1 210 11                        # <stdin>:210:11
	movq	stderr(%rip), %rdi
	.loc	1 215 11                        # <stdin>:215:11
	movl	$str6, %esi
	movl	$str3, %edx
	xorl	%eax, %eax
	callq	fprintf
	.loc	1 217 11                        # <stdin>:217:11
	movq	stderr(%rip), %rcx
	.loc	1 220 11                        # <stdin>:220:11
	movl	$str7, %edi
	movl	$22, %esi
	movl	$1, %edx
	popq	%rbx
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	jmp	fwrite                          # TAILCALL
.Ltmp9:
.Lfunc_end3:
	.size	print_array, .Lfunc_end3-print_array
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3                               # -- Begin function S0
.LCPI4_0:
	.quad	0x4022000000000000              # double 9
	.text
	.globl	S0
	.p2align	4, 0x90
	.type	S0,@function
S0:                                     # @S0
.Lfunc_begin4:
	.loc	1 257 0                         # <stdin>:257:0
	.cfi_startproc
# %bb.0:
	movq	24(%rsp), %rax
.Ltmp10:
	.loc	1 273 11 prologue_end           # <stdin>:273:11
	imulq	$4000, 16(%rsp), %rcx           # imm = 0xFA0
	.loc	1 274 11                        # <stdin>:274:11
	leaq	(%rcx,%rax), %rdx
	addq	$-4001, %rdx                    # imm = 0xF05F
	.loc	1 276 11                        # <stdin>:276:11
	movsd	(%rsi,%rdx,8), %xmm0            # xmm0 = mem[0],zero
	.loc	1 280 11                        # <stdin>:280:11
	leaq	-4000(%rcx,%rax), %rdx
	.loc	1 283 11                        # <stdin>:283:11
	addsd	(%rsi,%rdx,8), %xmm0
	.loc	1 288 11                        # <stdin>:288:11
	leaq	-3999(%rcx,%rax), %rdx
	.loc	1 291 11                        # <stdin>:291:11
	addsd	(%rsi,%rdx,8), %xmm0
	.loc	1 295 11                        # <stdin>:295:11
	leaq	-1(%rax,%rcx), %rdx
	.loc	1 298 11                        # <stdin>:298:11
	addsd	(%rsi,%rdx,8), %xmm0
	.loc	1 302 11                        # <stdin>:302:11
	leaq	(%rcx,%rax), %rdx
	.loc	1 305 11                        # <stdin>:305:11
	addsd	(%rsi,%rdx,8), %xmm0
	.loc	1 309 11                        # <stdin>:309:11
	leaq	1(%rax,%rcx), %rdi
	.loc	1 312 11                        # <stdin>:312:11
	addsd	(%rsi,%rdi,8), %xmm0
	.loc	1 317 11                        # <stdin>:317:11
	leaq	3999(%rcx,%rax), %rdi
	.loc	1 320 11                        # <stdin>:320:11
	addsd	(%rsi,%rdi,8), %xmm0
	.loc	1 324 11                        # <stdin>:324:11
	leaq	4000(%rcx,%rax), %rdi
	.loc	1 327 11                        # <stdin>:327:11
	addsd	(%rsi,%rdi,8), %xmm0
	.loc	1 331 11                        # <stdin>:331:11
	leaq	4001(%rcx,%rax), %rax
	.loc	1 334 11                        # <stdin>:334:11
	addsd	(%rsi,%rax,8), %xmm0
	.loc	1 335 11                        # <stdin>:335:11
	divsd	.LCPI4_0(%rip), %xmm0
	.loc	1 341 5                         # <stdin>:341:5
	movsd	%xmm0, (%rsi,%rdx,8)
	.loc	1 342 5                         # <stdin>:342:5
	retq
.Ltmp11:
.Lfunc_end4:
	.size	S0, .Lfunc_end4-S0
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3                               # -- Begin function kernel_seidel_2d_new
.LCPI5_0:
	.quad	0x4022000000000000              # double 9
	.text
	.globl	kernel_seidel_2d_new
	.p2align	4, 0x90
	.type	kernel_seidel_2d_new,@function
kernel_seidel_2d_new:                   # @kernel_seidel_2d_new
.Lfunc_begin5:
	.loc	1 344 0                         # <stdin>:344:0
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
	subq	$312, %rsp                      # imm = 0x138
	.cfi_def_cfa_offset 368
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rcx, (%rsp)                    # 8-byte Spill
.Ltmp12:
	.loc	1 373 11 prologue_end           # <stdin>:373:11
	testl	%edi, %edi
	.loc	1 377 5                         # <stdin>:377:5
	jle	.LBB5_19
# %bb.1:
	cmpl	$3, %esi
	jl	.LBB5_19
# %bb.2:                                # %.lr.ph33
	.loc	1 0 0 is_stmt 0                 # <stdin>:0:0
	movslq	%edi, %rdx
	movslq	%esi, %rbp
	.loc	1 372 11 is_stmt 1              # <stdin>:372:11
	leaq	-1(%rdx), %rcx
	.loc	1 382 11                        # <stdin>:382:11
	leaq	30(%rdx), %rax
	testq	%rcx, %rcx
	cmovnsq	%rcx, %rax
	sarq	$5, %rax
	leaq	(%rdx,%rbp), %rbx
	movq	%rdx, 24(%rsp)                  # 8-byte Spill
	leaq	(%rdx,%rbp), %r8
	addq	$-3, %r8
	movl	$2, %r9d
	movl	$2, %edx
	subq	%rbx, %rdx
	xorl	%esi, %esi
	movq	%rbx, 72(%rsp)                  # 8-byte Spill
	cmpq	$3, %rbx
	setl	%sil
	cmovgeq	%r8, %rdx
	leaq	31(%rdx), %rbx
	leaq	15(%rdx), %rcx
	testq	%rdx, %rdx
	cmovnsq	%rdx, %rbx
	cmovnsq	%rdx, %rcx
	sarq	$5, %rbx
	negq	%rsi
	xorq	%rsi, %rbx
	movq	%rbx, 40(%rsp)                  # 8-byte Spill
	sarq	$4, %rcx
	xorq	%rsi, %rcx
	movq	%rcx, 16(%rsp)                  # 8-byte Spill
	.loc	1 389 5                         # <stdin>:389:5
	addq	$1, %rax
	cmpl	$32, %edi
	movl	$1, %ecx
	cmovgq	%rax, %rcx
	movq	%rcx, 32(%rsp)                  # 8-byte Spill
	leaq	-1(%rbp), %r10
	.loc	1 481 5                         # <stdin>:481:5
	subq	%rbp, %r9
	movq	$-29, %rax
	movq	%rbp, -128(%rsp)                # 8-byte Spill
	.loc	1 389 5                         # <stdin>:389:5
	subq	%rbp, %rax
	xorl	%ecx, %ecx
	movq	%rcx, -96(%rsp)                 # 8-byte Spill
	movsd	.LCPI5_0(%rip), %xmm0           # xmm0 = mem[0],zero
	movq	%rax, 8(%rsp)                   # 8-byte Spill
	movq	%rax, -104(%rsp)                # 8-byte Spill
	movq	%r9, -88(%rsp)                  # 8-byte Spill
	movq	%r9, -112(%rsp)                 # 8-byte Spill
	xorl	%eax, %eax
	movq	%rax, -120(%rsp)                # 8-byte Spill
	xorl	%ebx, %ebx
	jmp	.LBB5_3
	.p2align	4, 0x90
.LBB5_18:                               # %._crit_edge30
                                        #   in Loop: Header=BB5_3 Depth=1
	.loc	1 576 12                        # <stdin>:576:12
	addq	$1, %rbx
	.loc	1 389 5                         # <stdin>:389:5
	addq	$32, -120(%rsp)                 # 8-byte Folded Spill
	addq	$32, -112(%rsp)                 # 8-byte Folded Spill
	addq	$2, -96(%rsp)                   # 8-byte Folded Spill
	addq	$-32, -104(%rsp)                # 8-byte Folded Spill
	.loc	1 388 11                        # <stdin>:388:11
	cmpq	32(%rsp), %rbx                  # 8-byte Folded Reload
	.loc	1 389 5                         # <stdin>:389:5
	je	.LBB5_19
.LBB5_3:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB5_5 Depth 2
                                        #       Child Loop BB5_7 Depth 3
                                        #         Child Loop BB5_10 Depth 4
                                        #           Child Loop BB5_12 Depth 5
                                        #             Child Loop BB5_14 Depth 6
	.loc	1 400 11                        # <stdin>:400:11
	movq	%rbx, %rcx
	shlq	$5, %rcx
	movq	-128(%rsp), %rax                # 8-byte Reload
	.loc	1 401 11                        # <stdin>:401:11
	leaq	(%rcx,%rax), %rdx
	movq	%rcx, -80(%rsp)                 # 8-byte Spill
	.loc	1 402 11                        # <stdin>:402:11
	addq	%rcx, %rax
	addq	$29, %rax
	.loc	1 404 11                        # <stdin>:404:11
	movq	$-30, %rcx
	subq	%rdx, %rcx
	movq	%rdx, 104(%rsp)                 # 8-byte Spill
	.loc	1 403 11                        # <stdin>:403:11
	cmpq	$-29, %rdx
	.loc	1 405 11                        # <stdin>:405:11
	cmovgeq	%rax, %rcx
	.loc	1 406 11                        # <stdin>:406:11
	leaq	31(%rcx), %rsi
	testq	%rcx, %rcx
	cmovnsq	%rcx, %rsi
	sarq	$5, %rsi
	.loc	1 408 11                        # <stdin>:408:11
	sarq	$63, %rax
	xorq	%rax, %rsi
	movq	40(%rsp), %rdx                  # 8-byte Reload
	.loc	1 410 11                        # <stdin>:410:11
	cmpq	%rsi, %rdx
	.loc	1 411 11                        # <stdin>:411:11
	cmovlq	%rdx, %rsi
	movq	%rsi, 96(%rsp)                  # 8-byte Spill
	.loc	1 414 11                        # <stdin>:414:11
	cmpq	%rsi, %rbx
	.loc	1 415 5                         # <stdin>:415:5
	jg	.LBB5_18
# %bb.4:                                # %.lr.ph29
                                        #   in Loop: Header=BB5_3 Depth=1
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	-88(%rsp), %rdx                 # 8-byte Reload
	movq	-80(%rsp), %rsi                 # 8-byte Reload
	addq	%rsi, %rdx
	movq	%rdx, 64(%rsp)                  # 8-byte Spill
	movq	8(%rsp), %rdx                   # 8-byte Reload
	subq	%rsi, %rdx
	movq	%rdx, 56(%rsp)                  # 8-byte Spill
	leaq	15(%rcx), %rdx
	testq	%rcx, %rcx
	cmovnsq	%rcx, %rdx
	sarq	$4, %rdx
	xorq	%rax, %rdx
	movq	16(%rsp), %rax                  # 8-byte Reload
	cmpq	%rdx, %rax
	cmovlq	%rax, %rdx
	movq	%rdx, 80(%rsp)                  # 8-byte Spill
	leaq	32(%rsi), %rcx
	movq	24(%rsp), %rax                  # 8-byte Reload
	cmpq	%rax, %rcx
	cmovgq	%rax, %rcx
	movq	%rcx, 48(%rsp)                  # 8-byte Spill
	movq	-120(%rsp), %rax                # 8-byte Reload
	movq	%rax, -32(%rsp)                 # 8-byte Spill
	movq	-104(%rsp), %rax                # 8-byte Reload
	movq	%rax, -56(%rsp)                 # 8-byte Spill
	movq	-96(%rsp), %rax                 # 8-byte Reload
	movq	%rax, -64(%rsp)                 # 8-byte Spill
	movq	-112(%rsp), %rax                # 8-byte Reload
	movq	%rax, -48(%rsp)                 # 8-byte Spill
	xorl	%eax, %eax
	movq	%rax, -72(%rsp)                 # 8-byte Spill
	movq	%rbx, %rbp
	movq	%rbx, 88(%rsp)                  # 8-byte Spill
	jmp	.LBB5_5
	.p2align	4, 0x90
.LBB5_17:                               # %._crit_edge26
                                        #   in Loop: Header=BB5_5 Depth=2
	movq	112(%rsp), %rcx                 # 8-byte Reload
	.loc	1 573 12 is_stmt 1              # <stdin>:573:12
	leaq	1(%rcx), %rbp
	.loc	1 415 5                         # <stdin>:415:5
	addq	$1, -72(%rsp)                   # 8-byte Folded Spill
	addq	$32, -48(%rsp)                  # 8-byte Folded Spill
	addq	$1, -64(%rsp)                   # 8-byte Folded Spill
	addq	$-32, -56(%rsp)                 # 8-byte Folded Spill
	addq	$32, -32(%rsp)                  # 8-byte Folded Spill
	.loc	1 414 11                        # <stdin>:414:11
	cmpq	96(%rsp), %rcx                  # 8-byte Folded Reload
	movq	88(%rsp), %rbx                  # 8-byte Reload
	.loc	1 415 5                         # <stdin>:415:5
	je	.LBB5_18
.LBB5_5:                                #   Parent Loop BB5_3 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB5_7 Depth 3
                                        #         Child Loop BB5_10 Depth 4
                                        #           Child Loop BB5_12 Depth 5
                                        #             Child Loop BB5_14 Depth 6
	.loc	1 417 11                        # <stdin>:417:11
	movq	%rbp, %rax
	shlq	$6, %rax
	.loc	1 419 11                        # <stdin>:419:11
	movq	%rax, %rcx
	movq	-128(%rsp), %r9                 # 8-byte Reload
	subq	%r9, %rcx
	.loc	1 422 11                        # <stdin>:422:11
	movl	$28, %edx
	subq	%rcx, %rdx
	.loc	1 423 11                        # <stdin>:423:11
	leaq	-29(%rcx), %rsi
	.loc	1 421 11                        # <stdin>:421:11
	cmpq	$29, %rcx
	.loc	1 424 11                        # <stdin>:424:11
	cmovlq	%rdx, %rsi
	.loc	1 425 11                        # <stdin>:425:11
	leaq	31(%rsi), %rdx
	testq	%rsi, %rsi
	cmovnsq	%rsi, %rdx
	sarq	$5, %rdx
	.loc	1 426 11                        # <stdin>:426:11
	movq	%rdx, %rsi
	negq	%rsi
	.loc	1 427 11                        # <stdin>:427:11
	addq	$1, %rdx
	.loc	1 421 11                        # <stdin>:421:11
	cmpq	$29, %rcx
	.loc	1 428 11                        # <stdin>:428:11
	cmovlq	%rsi, %rdx
	.loc	1 429 11                        # <stdin>:429:11
	leaq	(%rbx,%rbp), %r11
	.loc	1 430 11                        # <stdin>:430:11
	cmpq	%r11, %rdx
	movq	%rdx, %r8
	.loc	1 431 11                        # <stdin>:431:11
	cmovgq	%rdx, %r11
	.loc	1 441 11                        # <stdin>:441:11
	leaq	(%rax,%r9), %rbx
	addq	$59, %rbx
	.loc	1 440 11                        # <stdin>:440:11
	addq	%r9, %rax
	.loc	1 443 11                        # <stdin>:443:11
	movq	$-60, %rcx
	subq	%rax, %rcx
	.loc	1 442 11                        # <stdin>:442:11
	cmpq	$-59, %rax
	.loc	1 444 11                        # <stdin>:444:11
	cmovgeq	%rbx, %rcx
	.loc	1 445 11                        # <stdin>:445:11
	leaq	31(%rcx), %rax
	testq	%rcx, %rcx
	cmovnsq	%rcx, %rax
	sarq	$5, %rax
	.loc	1 447 11                        # <stdin>:447:11
	sarq	$63, %rbx
	xorq	%rax, %rbx
	movq	%rbp, 112(%rsp)                 # 8-byte Spill
	.loc	1 449 11                        # <stdin>:449:11
	shlq	$5, %rbp
	movq	104(%rsp), %rax                 # 8-byte Reload
	.loc	1 451 11                        # <stdin>:451:11
	leaq	(%rax,%rbp), %rcx
	.loc	1 452 11                        # <stdin>:452:11
	addq	%rbp, %rax
	addq	$60, %rax
	.loc	1 454 12                        # <stdin>:454:12
	movq	$-61, %rdx
	subq	%rcx, %rdx
	.loc	1 453 12                        # <stdin>:453:12
	cmpq	$-60, %rcx
	.loc	1 455 12                        # <stdin>:455:12
	cmovgeq	%rax, %rdx
	.loc	1 456 12                        # <stdin>:456:12
	leaq	31(%rdx), %rcx
	testq	%rdx, %rdx
	cmovnsq	%rdx, %rcx
	sarq	$5, %rcx
	.loc	1 458 12                        # <stdin>:458:12
	sarq	$63, %rax
	xorq	%rcx, %rax
	movq	72(%rsp), %rdx                  # 8-byte Reload
	.loc	1 461 12                        # <stdin>:461:12
	leaq	(%rdx,%rbp), %rcx
	movq	%rbp, -24(%rsp)                 # 8-byte Spill
	.loc	1 462 12                        # <stdin>:462:12
	addq	%rbp, %rdx
	addq	$28, %rdx
	.loc	1 464 12                        # <stdin>:464:12
	movq	$-29, %rdi
	subq	%rcx, %rdi
	.loc	1 463 12                        # <stdin>:463:12
	testq	%rdx, %rdx
	.loc	1 465 12                        # <stdin>:465:12
	cmovnsq	%rdx, %rdi
	.loc	1 466 12                        # <stdin>:466:12
	leaq	31(%rdi), %rcx
	testq	%rdi, %rdi
	cmovnsq	%rdi, %rcx
	sarq	$5, %rcx
	.loc	1 468 12                        # <stdin>:468:12
	sarq	$63, %rdx
	xorq	%rcx, %rdx
	movq	80(%rsp), %rcx                  # 8-byte Reload
	.loc	1 472 12                        # <stdin>:472:12
	cmpq	%rbx, %rcx
	.loc	1 473 12                        # <stdin>:473:12
	cmovlq	%rcx, %rbx
	.loc	1 474 12                        # <stdin>:474:12
	cmpq	%rax, %rbx
	.loc	1 475 12                        # <stdin>:475:12
	cmovgeq	%rax, %rbx
	.loc	1 476 12                        # <stdin>:476:12
	cmpq	%rdx, %rbx
	.loc	1 477 12                        # <stdin>:477:12
	cmovgeq	%rdx, %rbx
	movq	%rbx, 184(%rsp)                 # 8-byte Spill
	.loc	1 480 12                        # <stdin>:480:12
	cmpq	%rbx, %r11
	.loc	1 481 5                         # <stdin>:481:5
	jg	.LBB5_17
# %bb.6:                                # %.lr.ph25
                                        #   in Loop: Header=BB5_5 Depth=2
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	-72(%rsp), %rax                 # 8-byte Reload
	.loc	1 417 11 is_stmt 1              # <stdin>:417:11
	shlq	$5, %rax
	movq	-80(%rsp), %rcx                 # 8-byte Reload
	leaq	(%rcx,%rax), %rdx
	movq	%rdx, 248(%rsp)                 # 8-byte Spill
	movq	64(%rsp), %rdx                  # 8-byte Reload
	leaq	(%rdx,%rax), %r9
	movq	56(%rsp), %rbp                  # 8-byte Reload
	subq	%rax, %rbp
	movq	-128(%rsp), %rax                # 8-byte Reload
	negq	%rax
	movq	-24(%rsp), %rdx                 # 8-byte Reload
	movq	%rax, 240(%rsp)                 # 8-byte Spill
	addq	%rdx, %rax
	addq	$2, %rax
	movq	%rdx, %rbx
	negq	%rbx
	movq	%rbx, 152(%rsp)                 # 8-byte Spill
	cmpq	%rax, %rcx
	cmovgq	%rcx, %rax
	movq	%rax, 160(%rsp)                 # 8-byte Spill
	movq	%rdx, %rbx
	orq	$31, %rbx
	movq	48(%rsp), %rax                  # 8-byte Reload
	cmpq	%rbx, %rax
	cmovlq	%rax, %rbx
	movq	%rbx, 144(%rsp)                 # 8-byte Spill
	leaq	32(%rdx), %rax
	movq	%rax, 232(%rsp)                 # 8-byte Spill
	.loc	1 481 5                         # <stdin>:481:5
	movq	%r11, %rdx
	shlq	$5, %rdx
	movq	-88(%rsp), %rax                 # 8-byte Reload
	leaq	(%rax,%rdx), %rbx
	movq	%rbx, 120(%rsp)                 # 8-byte Spill
	movq	%r11, %rdi
	shlq	$4, %rdi
	addq	%rax, %rdi
	movq	%rdi, 136(%rsp)                 # 8-byte Spill
	movq	%rdx, 128(%rsp)                 # 8-byte Spill
	addq	%rdx, %rbp
	movq	%rbp, 168(%rsp)                 # 8-byte Spill
	cmpq	%r9, %rcx
	cmovgq	%rcx, %r9
	movq	%r9, 176(%rsp)                  # 8-byte Spill
	movq	-64(%rsp), %rcx                 # 8-byte Reload
	movq	%r8, %rbx
	cmpq	%rcx, %r8
	cmovleq	%rcx, %rbx
	movq	%rbx, %rdx
	shlq	$4, %rdx
	addq	%rax, %rdx
	shlq	$5, %rbx
	movq	-56(%rsp), %rcx                 # 8-byte Reload
	leaq	(%rcx,%rbx), %rdi
	addq	%rax, %rbx
	movq	%rbx, -40(%rsp)                 # 8-byte Spill
	xorl	%ebp, %ebp
	jmp	.LBB5_7
	.p2align	4, 0x90
.LBB5_16:                               # %._crit_edge22
                                        #   in Loop: Header=BB5_7 Depth=3
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	216(%rsp), %rcx                 # 8-byte Reload
	.loc	1 570 12 is_stmt 1              # <stdin>:570:12
	leaq	1(%rcx), %r11
	movq	192(%rsp), %rbp                 # 8-byte Reload
	.loc	1 481 5                         # <stdin>:481:5
	addq	$1, %rbp
	movq	208(%rsp), %rdx                 # 8-byte Reload
	addq	$16, %rdx
	movq	200(%rsp), %rdi                 # 8-byte Reload
	addq	$32, %rdi
	addq	$32, -40(%rsp)                  # 8-byte Folded Spill
	.loc	1 480 12                        # <stdin>:480:12
	cmpq	184(%rsp), %rcx                 # 8-byte Folded Reload
	.loc	1 481 5                         # <stdin>:481:5
	jge	.LBB5_17
.LBB5_7:                                #   Parent Loop BB5_3 Depth=1
                                        #     Parent Loop BB5_5 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB5_10 Depth 4
                                        #           Child Loop BB5_12 Depth 5
                                        #             Child Loop BB5_14 Depth 6
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	-120(%rsp), %rax                # 8-byte Reload
	movq	-48(%rsp), %rcx                 # 8-byte Reload
	.loc	1 485 12 is_stmt 1              # <stdin>:485:12
	cmpq	%rcx, %rax
	cmovgq	%rax, %rcx
	cmpq	%rdx, %rcx
	movq	%rdx, 208(%rsp)                 # 8-byte Spill
	cmovleq	%rdx, %rcx
	cmpq	%rdi, %rcx
	movq	%rdi, 200(%rsp)                 # 8-byte Spill
	cmovleq	%rdi, %rcx
	movq	%rbp, %rdx
	shlq	$5, %rdx
	movq	%rbp, 192(%rsp)                 # 8-byte Spill
	shlq	$4, %rbp
	addq	136(%rsp), %rbp                 # 8-byte Folded Reload
	movq	176(%rsp), %rax                 # 8-byte Reload
	cmpq	%rbp, %rax
	cmovgq	%rax, %rbp
	movq	168(%rsp), %rax                 # 8-byte Reload
	movq	%rdx, %rdi
	addq	%rdx, %rax
	cmpq	%rax, %rbp
	cmovleq	%rax, %rbp
	movq	%r11, %rsi
	shlq	$4, %rsi
	.loc	1 486 12                        # <stdin>:486:12
	movq	%rsi, %r8
	movq	-128(%rsp), %rdx                # 8-byte Reload
	subq	%rdx, %r8
	.loc	1 487 12                        # <stdin>:487:12
	addq	$2, %r8
	movq	%r11, 216(%rsp)                 # 8-byte Spill
	movq	%r11, %r9
	.loc	1 489 12                        # <stdin>:489:12
	shlq	$5, %r9
	movq	152(%rsp), %rax                 # 8-byte Reload
	.loc	1 490 12                        # <stdin>:490:12
	leaq	(%r9,%rax), %rbx
	.loc	1 491 12                        # <stdin>:491:12
	movq	%rbx, %rax
	subq	%rdx, %rax
	.loc	1 492 12                        # <stdin>:492:12
	addq	$-29, %rax
	movq	160(%rsp), %rdx                 # 8-byte Reload
	.loc	1 495 12                        # <stdin>:495:12
	cmpq	%r8, %rdx
	.loc	1 496 12                        # <stdin>:496:12
	cmovgq	%rdx, %r8
	.loc	1 497 12                        # <stdin>:497:12
	cmpq	%rax, %r8
	.loc	1 498 12                        # <stdin>:498:12
	cmovleq	%rax, %r8
	.loc	1 501 12                        # <stdin>:501:12
	orq	$15, %rsi
	.loc	1 502 12                        # <stdin>:502:12
	orq	$31, %rbx
	movq	144(%rsp), %rdx                 # 8-byte Reload
	.loc	1 507 12                        # <stdin>:507:12
	cmpq	%rsi, %rdx
	.loc	1 508 12                        # <stdin>:508:12
	cmovlq	%rdx, %rsi
	.loc	1 509 12                        # <stdin>:509:12
	cmpq	%rbx, %rsi
	.loc	1 510 12                        # <stdin>:510:12
	cmovgeq	%rbx, %rsi
	movq	%rsi, 272(%rsp)                 # 8-byte Spill
	.loc	1 513 12                        # <stdin>:513:12
	cmpq	%rsi, %r8
	.loc	1 514 5                         # <stdin>:514:5
	jge	.LBB5_16
# %bb.8:                                # %.lr.ph21
                                        #   in Loop: Header=BB5_7 Depth=3
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	imulq	$-32008, %rcx, %r14             # imm = 0x82F8
	addq	(%rsp), %r14                    # 8-byte Folded Reload
	movq	-40(%rsp), %rsi                 # 8-byte Reload
	subq	%rcx, %rsi
	addq	$1, %rcx
	movq	128(%rsp), %rax                 # 8-byte Reload
	addq	%rdi, %rax
	movq	%rax, 304(%rsp)                 # 8-byte Spill
	addq	120(%rsp), %rdi                 # 8-byte Folded Reload
	subq	%rbp, %rdi
	movq	%rdi, 264(%rsp)                 # 8-byte Spill
	imulq	$-4001, %rbp, %rax              # imm = 0xF05F
	addq	$1, %rbp
	movq	%rbp, 256(%rsp)                 # 8-byte Spill
	addq	$-1, %rax
	movq	%rax, 224(%rsp)                 # 8-byte Spill
	leaq	32(%r9), %rbx
	xorl	%edx, %edx
	jmp	.LBB5_10
	.p2align	4, 0x90
.LBB5_9:                                # %.loopexit
                                        #   in Loop: Header=BB5_10 Depth=4
	movq	-8(%rsp), %rdx                  # 8-byte Reload
	.loc	1 514 5                         # <stdin>:514:5
	addq	$1, %rdx
	movq	-16(%rsp), %rax                 # 8-byte Reload
	addq	$-32008, %rax                   # imm = 0x82F8
	movq	%rax, %r14
	movq	280(%rsp), %rsi                 # 8-byte Reload
	addq	$-1, %rsi
	movq	296(%rsp), %rcx                 # 8-byte Reload
	addq	$1, %rcx
	movq	288(%rsp), %r8                  # 8-byte Reload
	.loc	1 513 12 is_stmt 1              # <stdin>:513:12
	cmpq	272(%rsp), %r8                  # 8-byte Folded Reload
	.loc	1 514 5                         # <stdin>:514:5
	jge	.LBB5_16
.LBB5_10:                               #   Parent Loop BB5_3 Depth=1
                                        #     Parent Loop BB5_5 Depth=2
                                        #       Parent Loop BB5_7 Depth=3
                                        # =>      This Loop Header: Depth=4
                                        #           Child Loop BB5_12 Depth 5
                                        #             Child Loop BB5_14 Depth 6
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	%r14, -16(%rsp)                 # 8-byte Spill
	movq	-32(%rsp), %rax                 # 8-byte Reload
	.loc	1 516 12 is_stmt 1              # <stdin>:516:12
	cmpq	%rsi, %rax
	movq	%rsi, 280(%rsp)                 # 8-byte Spill
	cmovgq	%rax, %rsi
	cmpq	%rcx, %rsi
	movq	%rcx, 296(%rsp)                 # 8-byte Spill
	cmovleq	%rcx, %rsi
	movq	264(%rsp), %rcx                 # 8-byte Reload
	subq	%rdx, %rcx
	movq	248(%rsp), %rax                 # 8-byte Reload
	cmpq	%rcx, %rax
	cmovgq	%rax, %rcx
	movq	256(%rsp), %rax                 # 8-byte Reload
	movq	%rdx, -8(%rsp)                  # 8-byte Spill
	leaq	(%rax,%rdx), %r11
	cmpq	%r11, %rcx
	cmovleq	%r11, %rcx
	leaq	1(%r8), %rbp
	.loc	1 518 12                        # <stdin>:518:12
	movq	%r9, %r13
	subq	%r8, %r13
	movq	240(%rsp), %rax                 # 8-byte Reload
	.loc	1 520 12                        # <stdin>:520:12
	leaq	(%rax,%r13), %rdi
	addq	$2, %rdi
	movq	-24(%rsp), %rdx                 # 8-byte Reload
	.loc	1 521 12                        # <stdin>:521:12
	cmpq	%rbp, %rdx
	movq	%rbp, 288(%rsp)                 # 8-byte Spill
	.loc	1 522 12                        # <stdin>:522:12
	movq	%rbp, %rax
	cmovgq	%rdx, %rax
	.loc	1 523 12                        # <stdin>:523:12
	cmpq	%rdi, %rax
	.loc	1 524 12                        # <stdin>:524:12
	cmovleq	%rdi, %rax
	.loc	1 526 12                        # <stdin>:526:12
	addq	$31, %r13
	.loc	1 528 12                        # <stdin>:528:12
	leaq	(%r10,%r8), %rdi
	movq	232(%rsp), %rdx                 # 8-byte Reload
	.loc	1 529 12                        # <stdin>:529:12
	cmpq	%r13, %rdx
	.loc	1 530 12                        # <stdin>:530:12
	cmovlq	%rdx, %r13
	.loc	1 531 12                        # <stdin>:531:12
	cmpq	%rdi, %r13
	.loc	1 532 12                        # <stdin>:532:12
	cmovgeq	%rdi, %r13
	.loc	1 535 12                        # <stdin>:535:12
	cmpq	%r13, %rax
	.loc	1 536 5                         # <stdin>:536:5
	jge	.LBB5_9
# %bb.11:                               # %.lr.ph19.preheader
                                        #   in Loop: Header=BB5_10 Depth=4
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	imulq	$31992, %rsi, %r15              # imm = 0x7CF8
	addq	-16(%rsp), %r15                 # 8-byte Folded Reload
	addq	%rcx, %r11
	imulq	$-4001, -8(%rsp), %rsi          # 8-byte Folded Reload
                                        # imm = 0xF05F
	addq	224(%rsp), %rsi                 # 8-byte Folded Reload
	imulq	$3999, %rcx, %r12               # imm = 0xF9F
	addq	%rsi, %r12
	xorl	%ecx, %ecx
	jmp	.LBB5_12
	.p2align	4, 0x90
.LBB5_15:                               # %._crit_edge
                                        #   in Loop: Header=BB5_12 Depth=5
	.loc	1 567 12 is_stmt 1              # <stdin>:567:12
	addq	$1, %rax
	.loc	1 536 5                         # <stdin>:536:5
	addq	$1, %rcx
	addq	$31992, %r15                    # imm = 0x7CF8
	.loc	1 535 12                        # <stdin>:535:12
	cmpq	%r13, %rax
	.loc	1 536 5                         # <stdin>:536:5
	jge	.LBB5_9
.LBB5_12:                               # %.lr.ph19
                                        #   Parent Loop BB5_3 Depth=1
                                        #     Parent Loop BB5_5 Depth=2
                                        #       Parent Loop BB5_7 Depth=3
                                        #         Parent Loop BB5_10 Depth=4
                                        # =>        This Loop Header: Depth=5
                                        #             Child Loop BB5_14 Depth 6
	.loc	1 538 12                        # <stdin>:538:12
	leaq	(%rax,%r8), %rdi
	.loc	1 539 12                        # <stdin>:539:12
	leaq	(%rax,%r8), %rsi
	addq	$1, %rsi
	.loc	1 540 12                        # <stdin>:540:12
	cmpq	%rsi, %r9
	.loc	1 541 12                        # <stdin>:541:12
	cmovgq	%r9, %rsi
	.loc	1 544 12                        # <stdin>:544:12
	addq	%r10, %rdi
	.loc	1 545 12                        # <stdin>:545:12
	cmpq	%rdi, %rbx
	.loc	1 546 12                        # <stdin>:546:12
	cmovlq	%rbx, %rdi
	.loc	1 549 12                        # <stdin>:549:12
	cmpq	%rdi, %rsi
	.loc	1 550 5                         # <stdin>:550:5
	jge	.LBB5_15
# %bb.13:                               # %.lr.ph
                                        #   in Loop: Header=BB5_12 Depth=5
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	%r10, %rdx
	.loc	1 538 12 is_stmt 1              # <stdin>:538:12
	leaq	(%r11,%rcx), %r10
	movq	304(%rsp), %rbp                 # 8-byte Reload
	cmpq	%r10, %rbp
	cmovgq	%rbp, %r10
	imulq	$3999, %rcx, %r14               # imm = 0xF9F
	addq	%r12, %r14
	addq	%r10, %r14
	movq	%rdx, %r10
	movq	(%rsp), %rdx                    # 8-byte Reload
	movsd	(%rdx,%r14,8), %xmm1            # xmm1 = mem[0],zero
	.p2align	4, 0x90
.LBB5_14:                               #   Parent Loop BB5_3 Depth=1
                                        #     Parent Loop BB5_5 Depth=2
                                        #       Parent Loop BB5_7 Depth=3
                                        #         Parent Loop BB5_10 Depth=4
                                        #           Parent Loop BB5_12 Depth=5
                                        # =>          This Inner Loop Header: Depth=6
.Ltmp13:
	.loc	1 276 11                        # <stdin>:276:11
	movsd	-32008(%r15,%rsi,8), %xmm2      # xmm2 = mem[0],zero
	.loc	1 283 11                        # <stdin>:283:11
	addsd	-32000(%r15,%rsi,8), %xmm2
	.loc	1 291 11                        # <stdin>:291:11
	addsd	-31992(%r15,%rsi,8), %xmm2
	.loc	1 298 11                        # <stdin>:298:11
	addsd	%xmm1, %xmm2
	.loc	1 305 11                        # <stdin>:305:11
	addsd	(%r15,%rsi,8), %xmm2
	.loc	1 312 11                        # <stdin>:312:11
	addsd	8(%r15,%rsi,8), %xmm2
	.loc	1 320 11                        # <stdin>:320:11
	addsd	31992(%r15,%rsi,8), %xmm2
	.loc	1 327 11                        # <stdin>:327:11
	addsd	32000(%r15,%rsi,8), %xmm2
	.loc	1 334 11                        # <stdin>:334:11
	addsd	32008(%r15,%rsi,8), %xmm2
	.loc	1 335 11                        # <stdin>:335:11
	divsd	%xmm0, %xmm2
	.loc	1 341 5                         # <stdin>:341:5
	movsd	%xmm2, (%r15,%rsi,8)
.Ltmp14:
	.loc	1 564 12                        # <stdin>:564:12
	addq	$1, %rsi
	movapd	%xmm2, %xmm1
	.loc	1 549 12                        # <stdin>:549:12
	cmpq	%rdi, %rsi
	.loc	1 550 5                         # <stdin>:550:5
	jl	.LBB5_14
	jmp	.LBB5_15
.LBB5_19:                               # %.loopexit17
	.loc	1 579 5                         # <stdin>:579:5
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
.Ltmp15:
.Lfunc_end5:
	.size	kernel_seidel_2d_new, .Lfunc_end5-kernel_seidel_2d_new
	.cfi_endproc
                                        # -- End function
	.type	str7,@object                    # @str7
	.section	.rodata,"a",@progbits
	.p2align	4
str7:
	.asciz	"==END   DUMP_ARRAYS==\n"
	.size	str7, 23

	.type	str6,@object                    # @str6
	.p2align	4
str6:
	.asciz	"\nend   dump: %s\n"
	.size	str6, 17

	.type	str5,@object                    # @str5
str5:
	.asciz	"%0.2lf "
	.size	str5, 8

	.type	str3,@object                    # @str3
str3:
	.asciz	"A"
	.size	str3, 2

	.type	str2,@object                    # @str2
str2:
	.asciz	"begin dump: %s"
	.size	str2, 15

	.type	str1,@object                    # @str1
	.p2align	4
str1:
	.asciz	"==BEGIN DUMP_ARRAYS==\n"
	.size	str1, 23

	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	1                               # DW_CHILDREN_yes
	.byte	37                              # DW_AT_producer
	.byte	14                              # DW_FORM_strp
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	27                              # DW_AT_comp_dir
	.byte	14                              # DW_FORM_strp
	.ascii	"\264B"                         # DW_AT_GNU_pubnames
	.byte	25                              # DW_FORM_flag_present
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	0                               # DW_CHILDREN_no
	.byte	110                             # DW_AT_linkage_name
	.byte	14                              # DW_FORM_strp
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	32                              # DW_AT_inline
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	110                             # DW_AT_linkage_name
	.byte	14                              # DW_FORM_strp
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	4                               # Abbreviation Code
	.byte	29                              # DW_TAG_inlined_subroutine
	.byte	0                               # DW_CHILDREN_no
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	88                              # DW_AT_call_file
	.byte	11                              # DW_FORM_data1
	.byte	89                              # DW_AT_call_line
	.byte	11                              # DW_FORM_data1
	.byte	87                              # DW_AT_call_column
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	5                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	0                               # DW_CHILDREN_no
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	6                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	0                               # DW_CHILDREN_no
	.byte	110                             # DW_AT_linkage_name
	.byte	14                              # DW_FORM_strp
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	5                               # DW_FORM_data2
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	32                              # DW_AT_inline
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	7                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	0                               # DW_CHILDREN_no
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	110                             # DW_AT_linkage_name
	.byte	14                              # DW_FORM_strp
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	8                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	110                             # DW_AT_linkage_name
	.byte	14                              # DW_FORM_strp
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	5                               # DW_FORM_data2
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	9                               # Abbreviation Code
	.byte	29                              # DW_TAG_inlined_subroutine
	.byte	0                               # DW_CHILDREN_no
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	88                              # DW_AT_call_file
	.byte	11                              # DW_FORM_data1
	.byte	89                              # DW_AT_call_line
	.byte	5                               # DW_FORM_data2
	.byte	87                              # DW_AT_call_column
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	1                               # Abbrev [1] 0xb:0x104 DW_TAG_compile_unit
	.long	.Linfo_string0                  # DW_AT_producer
	.short	2                               # DW_AT_language
	.long	.Linfo_string1                  # DW_AT_name
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Linfo_string2                  # DW_AT_comp_dir
                                        # DW_AT_GNU_pubnames
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end5-.Lfunc_begin0       # DW_AT_high_pc
	.byte	2                               # Abbrev [2] 0x2a:0xc DW_TAG_subprogram
	.long	.Linfo_string3                  # DW_AT_linkage_name
	.long	.Linfo_string3                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	84                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	1                               # DW_AT_inline
	.byte	3                               # Abbrev [3] 0x36:0x2e DW_TAG_subprogram
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
	.long	.Linfo_string5                  # DW_AT_linkage_name
	.long	.Linfo_string5                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	4                               # Abbrev [4] 0x4f:0x14 DW_TAG_inlined_subroutine
	.long	42                              # DW_AT_abstract_origin
	.quad	.Ltmp1                          # DW_AT_low_pc
	.long	.Ltmp2-.Ltmp1                   # DW_AT_high_pc
	.byte	1                               # DW_AT_call_file
	.byte	46                              # DW_AT_call_line
	.byte	5                               # DW_AT_call_column
	.byte	0                               # End Of Children Mark
	.byte	5                               # Abbrev [5] 0x64:0x13 DW_TAG_subprogram
	.quad	.Lfunc_begin1                   # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
	.long	42                              # DW_AT_abstract_origin
	.byte	6                               # Abbrev [6] 0x77:0xd DW_TAG_subprogram
	.long	.Linfo_string4                  # DW_AT_linkage_name
	.long	.Linfo_string4                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.short	257                             # DW_AT_decl_line
                                        # DW_AT_external
	.byte	1                               # DW_AT_inline
	.byte	3                               # Abbrev [3] 0x84:0x2e DW_TAG_subprogram
	.quad	.Lfunc_begin2                   # DW_AT_low_pc
	.long	.Lfunc_end2-.Lfunc_begin2       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
	.long	.Linfo_string6                  # DW_AT_linkage_name
	.long	.Linfo_string6                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	129                             # DW_AT_decl_line
                                        # DW_AT_external
	.byte	4                               # Abbrev [4] 0x9d:0x14 DW_TAG_inlined_subroutine
	.long	119                             # DW_AT_abstract_origin
	.quad	.Ltmp5                          # DW_AT_low_pc
	.long	.Ltmp6-.Ltmp5                   # DW_AT_high_pc
	.byte	1                               # DW_AT_call_file
	.byte	164                             # DW_AT_call_line
	.byte	5                               # DW_AT_call_column
	.byte	0                               # End Of Children Mark
	.byte	7                               # Abbrev [7] 0xb2:0x19 DW_TAG_subprogram
	.quad	.Lfunc_begin3                   # DW_AT_low_pc
	.long	.Lfunc_end3-.Lfunc_begin3       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
	.long	.Linfo_string7                  # DW_AT_linkage_name
	.long	.Linfo_string7                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	178                             # DW_AT_decl_line
                                        # DW_AT_external
	.byte	5                               # Abbrev [5] 0xcb:0x13 DW_TAG_subprogram
	.quad	.Lfunc_begin4                   # DW_AT_low_pc
	.long	.Lfunc_end4-.Lfunc_begin4       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
	.long	119                             # DW_AT_abstract_origin
	.byte	8                               # Abbrev [8] 0xde:0x30 DW_TAG_subprogram
	.quad	.Lfunc_begin5                   # DW_AT_low_pc
	.long	.Lfunc_end5-.Lfunc_begin5       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
	.long	.Linfo_string8                  # DW_AT_linkage_name
	.long	.Linfo_string8                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.short	344                             # DW_AT_decl_line
                                        # DW_AT_external
	.byte	9                               # Abbrev [9] 0xf8:0x15 DW_TAG_inlined_subroutine
	.long	119                             # DW_AT_abstract_origin
	.quad	.Ltmp13                         # DW_AT_low_pc
	.long	.Ltmp14-.Ltmp13                 # DW_AT_high_pc
	.byte	1                               # DW_AT_call_file
	.short	563                             # DW_AT_call_line
	.byte	5                               # DW_AT_call_column
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"mlir"                          # string offset=0
.Linfo_string1:
	.asciz	"LLVMDialectModule"             # string offset=5
.Linfo_string2:
	.asciz	"/"                             # string offset=23
.Linfo_string3:
	.asciz	"init_array"                    # string offset=25
.Linfo_string4:
	.asciz	"S0"                            # string offset=36
.Linfo_string5:
	.asciz	"main"                          # string offset=39
.Linfo_string6:
	.asciz	"kernel_seidel_2d"              # string offset=44
.Linfo_string7:
	.asciz	"print_array"                   # string offset=61
.Linfo_string8:
	.asciz	"kernel_seidel_2d_new"          # string offset=73
	.section	.debug_pubnames,"",@progbits
	.long	.LpubNames_end0-.LpubNames_begin0 # Length of Public Names Info
.LpubNames_begin0:
	.short	2                               # DWARF Version
	.long	.Lcu_begin0                     # Offset of Compilation Unit Info
	.long	271                             # Compilation Unit Length
	.long	42                              # DIE offset
	.asciz	"init_array"                    # External Name
	.long	119                             # DIE offset
	.asciz	"S0"                            # External Name
	.long	222                             # DIE offset
	.asciz	"kernel_seidel_2d_new"          # External Name
	.long	54                              # DIE offset
	.asciz	"main"                          # External Name
	.long	132                             # DIE offset
	.asciz	"kernel_seidel_2d"              # External Name
	.long	178                             # DIE offset
	.asciz	"print_array"                   # External Name
	.long	0                               # End Mark
.LpubNames_end0:
	.section	.debug_pubtypes,"",@progbits
	.long	.LpubTypes_end0-.LpubTypes_begin0 # Length of Public Types Info
.LpubTypes_begin0:
	.short	2                               # DWARF Version
	.long	.Lcu_begin0                     # Offset of Compilation Unit Info
	.long	271                             # Compilation Unit Length
	.long	0                               # End Mark
.LpubTypes_end0:
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym str7
	.addrsig_sym str6
	.addrsig_sym str5
	.addrsig_sym str3
	.addrsig_sym str2
	.addrsig_sym str1
	.section	.debug_line,"",@progbits
.Lline_table_start0:
