	.text
	.file	"LLVMDialectModule"
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
.Lfunc_begin0:
	.file	1 "/home/ubuntu/polymer/example/polybench/EXTRALARGE/2mm" "<stdin>"
	.loc	1 14 0                          # <stdin>:14:0
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
	subq	$216, %rsp
	.cfi_def_cfa_offset 272
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rsi, 160(%rsp)                 # 8-byte Spill
	movl	%edi, 124(%rsp)                 # 4-byte Spill
.Ltmp0:
	.loc	1 79 11 prologue_end            # <stdin>:79:11
	movl	$23040000, %edi                 # imm = 0x15F9000
	callq	malloc
	movq	%rax, %r12
	.loc	1 97 11                         # <stdin>:97:11
	movl	$28160000, %edi                 # imm = 0x1ADB000
	callq	malloc
	movq	%rax, %r15
	.loc	1 115 12                        # <stdin>:115:12
	movl	$31680000, %edi                 # imm = 0x1E36600
	callq	malloc
	movq	%rax, 144(%rsp)                 # 8-byte Spill
	.loc	1 133 12                        # <stdin>:133:12
	movl	$34560000, %edi                 # imm = 0x20F5800
	callq	malloc
	movq	%rax, 136(%rsp)                 # 8-byte Spill
	.loc	1 151 12                        # <stdin>:151:12
	movl	$30720000, %edi                 # imm = 0x1D4C000
	callq	malloc
	movq	%rax, %r14
	xorl	%eax, %eax
	movsd	.LCPI0_0(%rip), %xmm0           # xmm0 = mem[0],zero
	movq	%r15, %rcx
	.p2align	4, 0x90
.LBB0_1:                                # %.preheader36
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_2 Depth 2
	.loc	1 0 12 is_stmt 0                # <stdin>:0:12
	movl	$1, %ebp
	xorl	%esi, %esi
	.p2align	4, 0x90
.LBB0_2:                                #   Parent Loop BB0_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	.loc	1 178 12 is_stmt 1              # <stdin>:178:12
	movl	%ebp, %edi
	imulq	$1374389535, %rdi, %rdi         # imm = 0x51EB851F
	shrq	$41, %rdi
	imull	$1600, %edi, %edi               # imm = 0x640
	movl	%ebp, %edx
	subl	%edi, %edx
	.loc	1 181 12                        # <stdin>:181:12
	xorps	%xmm1, %xmm1
	cvtsi2sd	%edx, %xmm1
	.loc	1 183 12                        # <stdin>:183:12
	divsd	%xmm0, %xmm1
	.loc	1 189 5                         # <stdin>:189:5
	movsd	%xmm1, (%rcx,%rsi,8)
	.loc	1 190 12                        # <stdin>:190:12
	addq	$1, %rsi
	.loc	1 174 12                        # <stdin>:174:12
	addl	%eax, %ebp
	cmpq	$2200, %rsi                     # imm = 0x898
	.loc	1 176 5                         # <stdin>:176:5
	jne	.LBB0_2
# %bb.3:                                #   in Loop: Header=BB0_1 Depth=1
	.loc	1 193 12                        # <stdin>:193:12
	addq	$1, %rax
	.loc	1 172 5                         # <stdin>:172:5
	addq	$17600, %rcx                    # imm = 0x44C0
	.loc	1 170 12                        # <stdin>:170:12
	cmpq	$1600, %rax                     # imm = 0x640
	.loc	1 172 5                         # <stdin>:172:5
	jne	.LBB0_1
# %bb.4:                                # %.preheader34.preheader
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	xorl	%eax, %eax
	movl	$2443359173, %ecx               # imm = 0x91A2B3C5
	movsd	.LCPI0_1(%rip), %xmm0           # xmm0 = mem[0],zero
	movq	144(%rsp), %rdx                 # 8-byte Reload
	.p2align	4, 0x90
.LBB0_5:                                # %.preheader34
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_6 Depth 2
	movl	%eax, %ebx
	xorl	%edi, %edi
	.p2align	4, 0x90
.LBB0_6:                                #   Parent Loop BB0_5 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	.loc	1 205 12 is_stmt 1              # <stdin>:205:12
	movl	%ebx, %ebp
	imulq	%rcx, %rbp
	shrq	$42, %rbp
	imull	$1800, %ebp, %ebp               # imm = 0x708
	movl	%ebx, %esi
	subl	%ebp, %esi
	.loc	1 207 12                        # <stdin>:207:12
	xorps	%xmm1, %xmm1
	cvtsi2sd	%esi, %xmm1
	.loc	1 209 12                        # <stdin>:209:12
	divsd	%xmm0, %xmm1
	.loc	1 215 5                         # <stdin>:215:5
	movsd	%xmm1, (%rdx,%rdi,8)
	.loc	1 204 12                        # <stdin>:204:12
	addq	$1, %rdi
	.loc	1 200 12                        # <stdin>:200:12
	addl	%eax, %ebx
	cmpq	$1800, %rdi                     # imm = 0x708
	.loc	1 202 5                         # <stdin>:202:5
	jne	.LBB0_6
# %bb.7:                                #   in Loop: Header=BB0_5 Depth=1
	.loc	1 218 12                        # <stdin>:218:12
	addq	$1, %rax
	.loc	1 198 5                         # <stdin>:198:5
	addq	$14400, %rdx                    # imm = 0x3840
	.loc	1 196 12                        # <stdin>:196:12
	cmpq	$2200, %rax                     # imm = 0x898
	.loc	1 198 5                         # <stdin>:198:5
	jne	.LBB0_5
# %bb.8:                                # %.preheader32.preheader
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movl	$1, %ebx
	xorl	%ecx, %ecx
	movsd	.LCPI0_2(%rip), %xmm0           # xmm0 = mem[0],zero
	movq	136(%rsp), %rdx                 # 8-byte Reload
	.p2align	4, 0x90
.LBB0_9:                                # %.preheader32
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_10 Depth 2
	movl	%ebx, %eax
	xorl	%edi, %edi
	.p2align	4, 0x90
.LBB0_10:                               #   Parent Loop BB0_9 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	.loc	1 229 12 is_stmt 1              # <stdin>:229:12
	movl	%eax, %ebp
	imulq	$458129845, %rbp, %rbp          # imm = 0x1B4E81B5
	shrq	$40, %rbp
	imull	$2400, %ebp, %ebp               # imm = 0x960
	movl	%eax, %esi
	subl	%ebp, %esi
	.loc	1 233 12                        # <stdin>:233:12
	xorps	%xmm1, %xmm1
	cvtsi2sd	%esi, %xmm1
	.loc	1 235 12                        # <stdin>:235:12
	divsd	%xmm0, %xmm1
	.loc	1 241 5                         # <stdin>:241:5
	movsd	%xmm1, (%rdx,%rdi,8)
	.loc	1 242 12                        # <stdin>:242:12
	addq	$1, %rdi
	.loc	1 225 12                        # <stdin>:225:12
	addl	%ecx, %eax
	cmpq	$2400, %rdi                     # imm = 0x960
	.loc	1 227 5                         # <stdin>:227:5
	jne	.LBB0_10
# %bb.11:                               #   in Loop: Header=BB0_9 Depth=1
	.loc	1 245 12                        # <stdin>:245:12
	addq	$1, %rcx
	.loc	1 223 5                         # <stdin>:223:5
	addl	$3, %ebx
	addq	$19200, %rdx                    # imm = 0x4B00
	.loc	1 221 12                        # <stdin>:221:12
	cmpq	$1800, %rcx                     # imm = 0x708
	.loc	1 223 5                         # <stdin>:223:5
	jne	.LBB0_9
# %bb.12:                               # %.preheader30.preheader
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	xorl	%eax, %eax
	movsd	.LCPI0_3(%rip), %xmm0           # xmm0 = mem[0],zero
	movq	%r14, %rcx
	xorl	%edx, %edx
	.p2align	4, 0x90
.LBB0_13:                               # %.preheader30
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_14 Depth 2
	movl	%eax, %ebx
	xorl	%edi, %edi
	.p2align	4, 0x90
.LBB0_14:                               #   Parent Loop BB0_13 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	.loc	1 265 12 is_stmt 1              # <stdin>:265:12
	movl	%ebx, %ebp
	shrl	$3, %ebp
	imulq	$499778013, %rbp, %rbp          # imm = 0x1DCA01DD
	shrq	$37, %rbp
	imull	$2200, %ebp, %ebp               # imm = 0x898
	movl	%ebx, %esi
	subl	%ebp, %esi
	.loc	1 268 12                        # <stdin>:268:12
	xorps	%xmm1, %xmm1
	cvtsi2sd	%esi, %xmm1
	.loc	1 270 12                        # <stdin>:270:12
	divsd	%xmm0, %xmm1
	.loc	1 276 5                         # <stdin>:276:5
	movsd	%xmm1, (%rcx,%rdi,8)
	.loc	1 277 12                        # <stdin>:277:12
	addq	$1, %rdi
	.loc	1 261 12                        # <stdin>:261:12
	addl	%edx, %ebx
	cmpq	$2400, %rdi                     # imm = 0x960
	.loc	1 263 5                         # <stdin>:263:5
	jne	.LBB0_14
# %bb.15:                               #   in Loop: Header=BB0_13 Depth=1
	.loc	1 280 12                        # <stdin>:280:12
	addq	$1, %rdx
	.loc	1 250 5                         # <stdin>:250:5
	addl	$2, %eax
	addq	$19200, %rcx                    # imm = 0x4B00
	.loc	1 248 12                        # <stdin>:248:12
	cmpq	$1600, %rdx                     # imm = 0x640
	.loc	1 250 5                         # <stdin>:250:5
	jne	.LBB0_13
# %bb.16:
	.loc	1 252 5                         # <stdin>:252:5
	callq	polybench_timer_start
	movl	$32, %esi
	xorl	%eax, %eax
	movsd	.LCPI0_4(%rip), %xmm1           # xmm1 = mem[0],zero
	movq	%r14, 128(%rsp)                 # 8-byte Spill
	jmp	.LBB0_17
	.p2align	4, 0x90
.LBB0_46:                               #   in Loop: Header=BB0_17 Depth=1
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	168(%rsp), %rax                 # 8-byte Reload
	.loc	1 455 12 is_stmt 1              # <stdin>:455:12
	addq	$1, %rax
	.loc	1 284 5                         # <stdin>:284:5
	addq	$32, %rsi
	movq	96(%rsp), %r14                  # 8-byte Reload
	addq	$614400, %r14                   # imm = 0x96000
	.loc	1 283 12                        # <stdin>:283:12
	cmpq	$50, %rax
	.loc	1 284 5                         # <stdin>:284:5
	je	.LBB0_47
.LBB0_17:                               # %.preheader29
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_18 Depth 2
                                        #       Child Loop BB0_33 Depth 3
                                        #         Child Loop BB0_34 Depth 4
                                        #       Child Loop BB0_22 Depth 3
                                        #         Child Loop BB0_23 Depth 4
                                        #       Child Loop BB0_26 Depth 3
                                        #         Child Loop BB0_27 Depth 4
                                        #         Child Loop BB0_29 Depth 4
                                        #       Child Loop BB0_42 Depth 3
                                        #         Child Loop BB0_43 Depth 4
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	imulq	$460800, %rax, %rcx             # imm = 0x70800
	movq	%rcx, 176(%rsp)                 # 8-byte Spill
	movq	%rax, 168(%rsp)                 # 8-byte Spill
	shlq	$5, %rax
	movq	%rax, 152(%rsp)                 # 8-byte Spill
	movq	%r14, 96(%rsp)                  # 8-byte Spill
	xorl	%edi, %edi
	movl	$32, %eax
	xorl	%ebp, %ebp
	movq	%rsi, 64(%rsp)                  # 8-byte Spill
	jmp	.LBB0_18
	.p2align	4, 0x90
.LBB0_45:                               # %.loopexit
                                        #   in Loop: Header=BB0_18 Depth=2
	.loc	1 452 12 is_stmt 1              # <stdin>:452:12
	addq	$1, %rbp
	movq	112(%rsp), %rax                 # 8-byte Reload
	.loc	1 287 5                         # <stdin>:287:5
	addq	$32, %rax
	addq	$-32, %rdi
	addq	$256, %r14                      # imm = 0x100
	.loc	1 286 12                        # <stdin>:286:12
	cmpq	$132, %rbp
	.loc	1 287 5                         # <stdin>:287:5
	je	.LBB0_46
.LBB0_18:                               #   Parent Loop BB0_17 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_33 Depth 3
                                        #         Child Loop BB0_34 Depth 4
                                        #       Child Loop BB0_22 Depth 3
                                        #         Child Loop BB0_23 Depth 4
                                        #       Child Loop BB0_26 Depth 3
                                        #         Child Loop BB0_27 Depth 4
                                        #         Child Loop BB0_29 Depth 4
                                        #       Child Loop BB0_42 Depth 3
                                        #         Child Loop BB0_43 Depth 4
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	%rdi, 80(%rsp)                  # 8-byte Spill
	movq	%r14, 32(%rsp)                  # 8-byte Spill
	.loc	1 290 12 is_stmt 1              # <stdin>:290:12
	cmpq	$2400, %rax                     # imm = 0x960
	movl	$2400, %ecx                     # imm = 0x960
	cmovbq	%rax, %rcx
	cmpq	$1801, %rcx                     # imm = 0x709
	movl	$1801, %r13d                    # imm = 0x709
	movq	%rcx, 208(%rsp)                 # 8-byte Spill
	cmovaq	%rcx, %r13
	cmpq	$1800, %rax                     # imm = 0x708
	movl	$1800, %ebx                     # imm = 0x708
	movq	%rax, 112(%rsp)                 # 8-byte Spill
	cmovbq	%rax, %rbx
	movq	%rbp, %rax
	shlq	$8, %rax
	addq	176(%rsp), %rax                 # 8-byte Folded Reload
	movq	%rax, 40(%rsp)                  # 8-byte Spill
	movq	%rbp, %rax
	shlq	$5, %rax
	leaq	32(%rax), %rcx
	cmpq	$1800, %rcx                     # imm = 0x708
	movl	$1800, %edx                     # imm = 0x708
	movq	%rcx, 184(%rsp)                 # 8-byte Spill
	cmovbq	%rcx, %rdx
	movq	%rdx, 48(%rsp)                  # 8-byte Spill
	movq	%rdx, %rcx
	movq	%rax, 56(%rsp)                  # 8-byte Spill
	subq	%rax, %rcx
	shlq	$3, %rcx
	movq	%rcx, 72(%rsp)                  # 8-byte Spill
	movl	$56, %eax
	subq	%rbp, %rax
	.loc	1 292 12                        # <stdin>:292:12
	movl	$74, %ecx
	movq	%rbp, 104(%rsp)                 # 8-byte Spill
	subq	%rbp, %rcx
	movq	%rax, 192(%rsp)                 # 8-byte Spill
	movq	%rcx, 200(%rsp)                 # 8-byte Spill
	.loc	1 294 12                        # <stdin>:294:12
	orq	%rcx, %rax
	.loc	1 295 5                         # <stdin>:295:5
	js	.LBB0_36
# %bb.19:                               # %.lr.ph60
                                        #   in Loop: Header=BB0_18 Depth=2
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	56(%rsp), %rdx                  # 8-byte Reload
	cmpq	$2367, %rdx                     # imm = 0x93F
	seta	%cl
	cmpq	$1768, %rdx                     # imm = 0x6E8
	setg	%al
	orb	%cl, %al
	cmpq	48(%rsp), %rdx                  # 8-byte Folded Reload
	.loc	1 304 5 is_stmt 1               # <stdin>:304:5
	jae	.LBB0_31
# %bb.20:                               # %.lr.ph60.split.us.preheader
                                        #   in Loop: Header=BB0_18 Depth=2
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	addq	80(%rsp), %rbx                  # 8-byte Folded Reload
	.loc	1 313 5 is_stmt 1               # <stdin>:313:5
	testb	%al, %al
	je	.LBB0_21
# %bb.25:                               # %.lr.ph60.split.us.us.preheader
                                        #   in Loop: Header=BB0_18 Depth=2
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	96(%rsp), %rbp                  # 8-byte Reload
	movq	32(%rsp), %r14                  # 8-byte Reload
	xorl	%eax, %eax
	.p2align	4, 0x90
.LBB0_26:                               # %.lr.ph60.split.us.us
                                        #   Parent Loop BB0_17 Depth=1
                                        #     Parent Loop BB0_18 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_27 Depth 4
                                        #         Child Loop BB0_29 Depth 4
	movq	%rax, 88(%rsp)                  # 8-byte Spill
	imulq	$14400, %rax, %rdi              # imm = 0x3840
	addq	40(%rsp), %rdi                  # 8-byte Folded Reload
	addq	%r12, %rdi
	.loc	1 333 5 is_stmt 1               # <stdin>:333:5
	xorl	%esi, %esi
	movq	72(%rsp), %rdx                  # 8-byte Reload
	callq	memset
	movsd	.LCPI0_4(%rip), %xmm1           # xmm1 = mem[0],zero
	xorl	%eax, %eax
	.p2align	4, 0x90
.LBB0_27:                               #   Parent Loop BB0_17 Depth=1
                                        #     Parent Loop BB0_18 Depth=2
                                        #       Parent Loop BB0_26 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movsd	(%r14,%rax,8), %xmm0            # xmm0 = mem[0],zero
	.loc	1 321 12 is_stmt 1              # <stdin>:321:12
	mulsd	%xmm1, %xmm0
	.loc	1 327 5                         # <stdin>:327:5
	movsd	%xmm0, (%r14,%rax,8)
	.loc	1 312 12                        # <stdin>:312:12
	addq	$1, %rax
	cmpq	%rax, %rbx
	.loc	1 313 5                         # <stdin>:313:5
	jne	.LBB0_27
# %bb.28:                               # %.lr.ph57.us.us.preheader
                                        #   in Loop: Header=BB0_26 Depth=3
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movl	$1800, %eax                     # imm = 0x708
	.p2align	4, 0x90
.LBB0_29:                               # %.lr.ph57.us.us
                                        #   Parent Loop BB0_17 Depth=1
                                        #     Parent Loop BB0_18 Depth=2
                                        #       Parent Loop BB0_26 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	movsd	(%rbp,%rax,8), %xmm0            # xmm0 = mem[0],zero
	.loc	1 362 12 is_stmt 1              # <stdin>:362:12
	mulsd	%xmm1, %xmm0
	.loc	1 368 5                         # <stdin>:368:5
	movsd	%xmm0, (%rbp,%rax,8)
	.loc	1 369 12                        # <stdin>:369:12
	addq	$1, %rax
	.loc	1 353 12                        # <stdin>:353:12
	cmpq	%rax, %r13
	.loc	1 354 5                         # <stdin>:354:5
	jne	.LBB0_29
# %bb.30:                               # %._crit_edge58.us.loopexit.us
                                        #   in Loop: Header=BB0_26 Depth=3
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	88(%rsp), %rax                  # 8-byte Reload
	.loc	1 304 5 is_stmt 1               # <stdin>:304:5
	addq	$1, %rax
	addq	$19200, %r14                    # imm = 0x4B00
	addq	$19200, %rbp                    # imm = 0x4B00
	.loc	1 303 12                        # <stdin>:303:12
	cmpq	$32, %rax
	.loc	1 304 5                         # <stdin>:304:5
	jne	.LBB0_26
	jmp	.LBB0_36
	.p2align	4, 0x90
.LBB0_31:                               # %.lr.ph60.split
                                        #   in Loop: Header=BB0_18 Depth=2
	testb	%al, %al
	je	.LBB0_36
# %bb.32:                               # %.lr.ph60.split.split.us.preheader
                                        #   in Loop: Header=BB0_18 Depth=2
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	96(%rsp), %rax                  # 8-byte Reload
	movq	152(%rsp), %rcx                 # 8-byte Reload
	.p2align	4, 0x90
.LBB0_33:                               # %.lr.ph60.split.split.us
                                        #   Parent Loop BB0_17 Depth=1
                                        #     Parent Loop BB0_18 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_34 Depth 4
	movl	$1800, %edx                     # imm = 0x708
	.p2align	4, 0x90
.LBB0_34:                               #   Parent Loop BB0_17 Depth=1
                                        #     Parent Loop BB0_18 Depth=2
                                        #       Parent Loop BB0_33 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	movsd	(%rax,%rdx,8), %xmm0            # xmm0 = mem[0],zero
	.loc	1 362 12 is_stmt 1              # <stdin>:362:12
	mulsd	%xmm1, %xmm0
	.loc	1 368 5                         # <stdin>:368:5
	movsd	%xmm0, (%rax,%rdx,8)
	.loc	1 369 12                        # <stdin>:369:12
	addq	$1, %rdx
	.loc	1 353 12                        # <stdin>:353:12
	cmpq	%rdx, %r13
	.loc	1 354 5                         # <stdin>:354:5
	jne	.LBB0_34
# %bb.35:                               # %._crit_edge58.us85
                                        #   in Loop: Header=BB0_33 Depth=3
	.loc	1 372 12                        # <stdin>:372:12
	addq	$1, %rcx
	.loc	1 304 5                         # <stdin>:304:5
	addq	$19200, %rax                    # imm = 0x4B00
	.loc	1 303 12                        # <stdin>:303:12
	cmpq	64(%rsp), %rcx                  # 8-byte Folded Reload
	.loc	1 304 5                         # <stdin>:304:5
	jne	.LBB0_33
	jmp	.LBB0_36
.LBB0_21:                               # %.lr.ph60.split.us.preheader82
                                        #   in Loop: Header=BB0_18 Depth=2
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	32(%rsp), %r14                  # 8-byte Reload
	xorl	%ebp, %ebp
	.p2align	4, 0x90
.LBB0_22:                               # %.lr.ph60.split.us
                                        #   Parent Loop BB0_17 Depth=1
                                        #     Parent Loop BB0_18 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_23 Depth 4
	imulq	$14400, %rbp, %rdi              # imm = 0x3840
	addq	40(%rsp), %rdi                  # 8-byte Folded Reload
	addq	%r12, %rdi
	.loc	1 333 5 is_stmt 1               # <stdin>:333:5
	xorl	%esi, %esi
	movq	72(%rsp), %rdx                  # 8-byte Reload
	callq	memset
	movsd	.LCPI0_4(%rip), %xmm1           # xmm1 = mem[0],zero
	xorl	%eax, %eax
	.p2align	4, 0x90
.LBB0_23:                               #   Parent Loop BB0_17 Depth=1
                                        #     Parent Loop BB0_18 Depth=2
                                        #       Parent Loop BB0_22 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movsd	(%r14,%rax,8), %xmm0            # xmm0 = mem[0],zero
	.loc	1 321 12 is_stmt 1              # <stdin>:321:12
	mulsd	%xmm1, %xmm0
	.loc	1 327 5                         # <stdin>:327:5
	movsd	%xmm0, (%r14,%rax,8)
	.loc	1 312 12                        # <stdin>:312:12
	addq	$1, %rax
	cmpq	%rax, %rbx
	.loc	1 313 5                         # <stdin>:313:5
	jne	.LBB0_23
# %bb.24:                               # %..preheader26_crit_edge.us
                                        #   in Loop: Header=BB0_22 Depth=3
	.loc	1 304 5                         # <stdin>:304:5
	addq	$1, %rbp
	addq	$19200, %r14                    # imm = 0x4B00
	.loc	1 303 12                        # <stdin>:303:12
	cmpq	$32, %rbp
	.loc	1 304 5                         # <stdin>:304:5
	jne	.LBB0_22
	.p2align	4, 0x90
.LBB0_36:                               # %.loopexit28
                                        #   in Loop: Header=BB0_18 Depth=2
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	104(%rsp), %rbp                 # 8-byte Reload
	.loc	1 375 12 is_stmt 1              # <stdin>:375:12
	leaq	-75(%rbp), %rax
	.loc	1 377 12                        # <stdin>:377:12
	orq	%rax, 192(%rsp)                 # 8-byte Folded Spill
	.loc	1 378 5                         # <stdin>:378:5
	js	.LBB0_39
# %bb.37:                               # %.loopexit28
                                        #   in Loop: Header=BB0_18 Depth=2
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	48(%rsp), %rax                  # 8-byte Reload
	.loc	1 378 5                         # <stdin>:378:5
	cmpq	%rax, 56(%rsp)                  # 8-byte Folded Reload
	jae	.LBB0_39
# %bb.38:                               # %.lr.ph65.split.us.preheader
                                        #   in Loop: Header=BB0_18 Depth=2
	.loc	1 0 5                           # <stdin>:0:5
	movq	40(%rsp), %rbx                  # 8-byte Reload
	leaq	(%r12,%rbx), %rdi
	.loc	1 403 5 is_stmt 1               # <stdin>:403:5
	xorl	%esi, %esi
	movq	72(%rsp), %r14                  # 8-byte Reload
	movq	%r14, %rdx
	callq	memset
	leaq	(%r12,%rbx), %rdi
	addq	$14400, %rdi                    # imm = 0x3840
	xorl	%esi, %esi
	movq	%r14, %rdx
	callq	memset
	leaq	(%r12,%rbx), %rdi
	addq	$28800, %rdi                    # imm = 0x7080
	xorl	%esi, %esi
	movq	%r14, %rdx
	callq	memset
	leaq	(%r12,%rbx), %rdi
	addq	$43200, %rdi                    # imm = 0xA8C0
	xorl	%esi, %esi
	movq	%r14, %rdx
	callq	memset
	leaq	(%r12,%rbx), %rdi
	addq	$57600, %rdi                    # imm = 0xE100
	xorl	%esi, %esi
	movq	%r14, %rdx
	callq	memset
	leaq	(%r12,%rbx), %rdi
	addq	$72000, %rdi                    # imm = 0x11940
	xorl	%esi, %esi
	movq	%r14, %rdx
	callq	memset
	leaq	(%r12,%rbx), %rdi
	addq	$86400, %rdi                    # imm = 0x15180
	xorl	%esi, %esi
	movq	%r14, %rdx
	callq	memset
	leaq	(%r12,%rbx), %rdi
	addq	$100800, %rdi                   # imm = 0x189C0
	xorl	%esi, %esi
	movq	%r14, %rdx
	callq	memset
	leaq	(%r12,%rbx), %rdi
	addq	$115200, %rdi                   # imm = 0x1C200
	xorl	%esi, %esi
	movq	%r14, %rdx
	callq	memset
	leaq	(%r12,%rbx), %rdi
	addq	$129600, %rdi                   # imm = 0x1FA40
	xorl	%esi, %esi
	movq	%r14, %rdx
	callq	memset
	leaq	(%r12,%rbx), %rdi
	addq	$144000, %rdi                   # imm = 0x23280
	xorl	%esi, %esi
	movq	%r14, %rdx
	callq	memset
	leaq	(%r12,%rbx), %rdi
	addq	$158400, %rdi                   # imm = 0x26AC0
	xorl	%esi, %esi
	movq	%r14, %rdx
	callq	memset
	leaq	(%r12,%rbx), %rdi
	addq	$172800, %rdi                   # imm = 0x2A300
	xorl	%esi, %esi
	movq	%r14, %rdx
	callq	memset
	leaq	(%r12,%rbx), %rdi
	addq	$187200, %rdi                   # imm = 0x2DB40
	xorl	%esi, %esi
	movq	%r14, %rdx
	callq	memset
	leaq	(%r12,%rbx), %rdi
	addq	$201600, %rdi                   # imm = 0x31380
	xorl	%esi, %esi
	movq	%r14, %rdx
	callq	memset
	leaq	(%r12,%rbx), %rdi
	addq	$216000, %rdi                   # imm = 0x34BC0
	xorl	%esi, %esi
	movq	%r14, %rdx
	callq	memset
	leaq	(%r12,%rbx), %rdi
	addq	$230400, %rdi                   # imm = 0x38400
	xorl	%esi, %esi
	movq	%r14, %rdx
	callq	memset
	leaq	(%r12,%rbx), %rdi
	addq	$244800, %rdi                   # imm = 0x3BC40
	xorl	%esi, %esi
	movq	%r14, %rdx
	callq	memset
	leaq	(%r12,%rbx), %rdi
	addq	$259200, %rdi                   # imm = 0x3F480
	xorl	%esi, %esi
	movq	%r14, %rdx
	callq	memset
	leaq	(%r12,%rbx), %rdi
	addq	$273600, %rdi                   # imm = 0x42CC0
	xorl	%esi, %esi
	movq	%r14, %rdx
	callq	memset
	leaq	(%r12,%rbx), %rdi
	addq	$288000, %rdi                   # imm = 0x46500
	xorl	%esi, %esi
	movq	%r14, %rdx
	callq	memset
	leaq	(%r12,%rbx), %rdi
	addq	$302400, %rdi                   # imm = 0x49D40
	xorl	%esi, %esi
	movq	%r14, %rdx
	callq	memset
	leaq	(%r12,%rbx), %rdi
	addq	$316800, %rdi                   # imm = 0x4D580
	xorl	%esi, %esi
	movq	%r14, %rdx
	callq	memset
	leaq	(%r12,%rbx), %rdi
	addq	$331200, %rdi                   # imm = 0x50DC0
	xorl	%esi, %esi
	movq	%r14, %rdx
	callq	memset
	leaq	(%r12,%rbx), %rdi
	addq	$345600, %rdi                   # imm = 0x54600
	xorl	%esi, %esi
	movq	%r14, %rdx
	callq	memset
	leaq	(%r12,%rbx), %rdi
	addq	$360000, %rdi                   # imm = 0x57E40
	xorl	%esi, %esi
	movq	%r14, %rdx
	callq	memset
	leaq	(%r12,%rbx), %rdi
	addq	$374400, %rdi                   # imm = 0x5B680
	xorl	%esi, %esi
	movq	%r14, %rdx
	callq	memset
	leaq	(%r12,%rbx), %rdi
	addq	$388800, %rdi                   # imm = 0x5EEC0
	xorl	%esi, %esi
	movq	%r14, %rdx
	callq	memset
	leaq	(%r12,%rbx), %rdi
	addq	$403200, %rdi                   # imm = 0x62700
	xorl	%esi, %esi
	movq	%r14, %rdx
	callq	memset
	leaq	(%r12,%rbx), %rdi
	addq	$417600, %rdi                   # imm = 0x65F40
	xorl	%esi, %esi
	movq	%r14, %rdx
	callq	memset
	leaq	(%r12,%rbx), %rdi
	addq	$432000, %rdi                   # imm = 0x69780
	xorl	%esi, %esi
	movq	%r14, %rdx
	callq	memset
	leaq	(%r12,%rbx), %rdi
	addq	$446400, %rdi                   # imm = 0x6CFC0
	xorl	%esi, %esi
	movq	%r14, %rdx
	callq	memset
	movsd	.LCPI0_4(%rip), %xmm1           # xmm1 = mem[0],zero
.LBB0_39:                               # %.loopexit27
                                        #   in Loop: Header=BB0_18 Depth=2
	.loc	1 410 12                        # <stdin>:410:12
	leaq	-57(%rbp), %rax
	.loc	1 412 12                        # <stdin>:412:12
	orq	%rax, 200(%rsp)                 # 8-byte Folded Spill
	movq	64(%rsp), %rsi                  # 8-byte Reload
	movq	32(%rsp), %r14                  # 8-byte Reload
	movq	80(%rsp), %rdi                  # 8-byte Reload
	movq	208(%rsp), %rbx                 # 8-byte Reload
	.loc	1 413 5                         # <stdin>:413:5
	js	.LBB0_45
# %bb.40:                               # %.lr.ph70
                                        #   in Loop: Header=BB0_18 Depth=2
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	184(%rsp), %rcx                 # 8-byte Reload
	cmpq	$2400, %rcx                     # imm = 0x960
	movl	$2400, %eax                     # imm = 0x960
	cmovaeq	%rax, %rcx
	cmpq	%rcx, 56(%rsp)                  # 8-byte Folded Reload
	.loc	1 422 5 is_stmt 1               # <stdin>:422:5
	jae	.LBB0_45
# %bb.41:                               # %.lr.ph70.split.us.preheader
                                        #   in Loop: Header=BB0_18 Depth=2
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	addq	%rdi, %rbx
	movq	%r14, %rax
	movq	152(%rsp), %rcx                 # 8-byte Reload
	.p2align	4, 0x90
.LBB0_42:                               # %.lr.ph70.split.us
                                        #   Parent Loop BB0_17 Depth=1
                                        #     Parent Loop BB0_18 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_43 Depth 4
	xorl	%edx, %edx
	.p2align	4, 0x90
.LBB0_43:                               #   Parent Loop BB0_17 Depth=1
                                        #     Parent Loop BB0_18 Depth=2
                                        #       Parent Loop BB0_42 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	movsd	(%rax,%rdx,8), %xmm0            # xmm0 = mem[0],zero
	.loc	1 439 12 is_stmt 1              # <stdin>:439:12
	mulsd	%xmm1, %xmm0
	.loc	1 445 5                         # <stdin>:445:5
	movsd	%xmm0, (%rax,%rdx,8)
	.loc	1 430 12                        # <stdin>:430:12
	addq	$1, %rdx
	cmpq	%rdx, %rbx
	.loc	1 431 5                         # <stdin>:431:5
	jne	.LBB0_43
# %bb.44:                               # %._crit_edge68.us
                                        #   in Loop: Header=BB0_42 Depth=3
	.loc	1 449 12                        # <stdin>:449:12
	addq	$1, %rcx
	.loc	1 422 5                         # <stdin>:422:5
	addq	$19200, %rax                    # imm = 0x4B00
	.loc	1 421 12                        # <stdin>:421:12
	cmpq	%rsi, %rcx
	.loc	1 422 5                         # <stdin>:422:5
	jne	.LBB0_42
	jmp	.LBB0_45
.LBB0_47:                               # %.preheader24.preheader
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movl	$32, %r11d
	xorl	%eax, %eax
	movsd	.LCPI0_5(%rip), %xmm0           # xmm0 = mem[0],zero
	movq	128(%rsp), %rcx                 # 8-byte Reload
	movq	%rcx, 32(%rsp)                  # 8-byte Spill
	movq	%r12, %rcx
	jmp	.LBB0_48
	.p2align	4, 0x90
.LBB0_63:                               #   in Loop: Header=BB0_48 Depth=1
	movq	112(%rsp), %rax                 # 8-byte Reload
	.loc	1 604 12 is_stmt 1              # <stdin>:604:12
	addq	$1, %rax
	.loc	1 459 5                         # <stdin>:459:5
	addq	$32, %r11
	movq	104(%rsp), %rcx                 # 8-byte Reload
	addq	$460800, %rcx                   # imm = 0x70800
	addq	$614400, 32(%rsp)               # 8-byte Folded Spill
                                        # imm = 0x96000
	.loc	1 458 12                        # <stdin>:458:12
	cmpq	$50, %rax
	.loc	1 459 5                         # <stdin>:459:5
	je	.LBB0_64
.LBB0_48:                               # %.preheader24
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_49 Depth 2
                                        #       Child Loop BB0_51 Depth 3
                                        #         Child Loop BB0_69 Depth 4
                                        #           Child Loop BB0_70 Depth 5
                                        #             Child Loop BB0_71 Depth 6
                                        #       Child Loop BB0_55 Depth 3
                                        #         Child Loop BB0_56 Depth 4
                                        #           Child Loop BB0_57 Depth 5
                                        #             Child Loop BB0_58 Depth 6
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	%rax, 112(%rsp)                 # 8-byte Spill
	shlq	$5, %rax
	movq	%rax, 88(%rsp)                  # 8-byte Spill
	movq	136(%rsp), %r14                 # 8-byte Reload
	movq	%rcx, 104(%rsp)                 # 8-byte Spill
	movq	%rcx, 64(%rsp)                  # 8-byte Spill
	movq	144(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 48(%rsp)                  # 8-byte Spill
	xorl	%esi, %esi
	xorl	%eax, %eax
	jmp	.LBB0_49
	.p2align	4, 0x90
.LBB0_62:                               # %.us-lcssa.us
                                        #   in Loop: Header=BB0_49 Depth=2
	movq	56(%rsp), %rax                  # 8-byte Reload
	.loc	1 601 12 is_stmt 1              # <stdin>:601:12
	addq	$1, %rax
	.loc	1 462 5                         # <stdin>:462:5
	addq	$32, %rsi
	addq	$256, 48(%rsp)                  # 8-byte Folded Spill
                                        # imm = 0x100
	addq	$256, 64(%rsp)                  # 8-byte Folded Spill
                                        # imm = 0x100
	addq	$614400, %r14                   # imm = 0x96000
	.loc	1 461 12                        # <stdin>:461:12
	cmpq	$57, %rax
	.loc	1 462 5                         # <stdin>:462:5
	je	.LBB0_63
.LBB0_49:                               # %.preheader23
                                        #   Parent Loop BB0_48 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_51 Depth 3
                                        #         Child Loop BB0_69 Depth 4
                                        #           Child Loop BB0_70 Depth 5
                                        #             Child Loop BB0_71 Depth 6
                                        #       Child Loop BB0_55 Depth 3
                                        #         Child Loop BB0_56 Depth 4
                                        #           Child Loop BB0_57 Depth 5
                                        #             Child Loop BB0_58 Depth 6
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	%rax, 56(%rsp)                  # 8-byte Spill
	shlq	$5, %rax
	cmpq	$1768, %rax                     # imm = 0x6E8
	movl	$1768, %ecx                     # imm = 0x6E8
	cmovbq	%rax, %rcx
	addq	$32, %rcx
	movq	%rax, 40(%rsp)                  # 8-byte Spill
	cmpq	%rcx, %rax
	.loc	1 465 5 is_stmt 1               # <stdin>:465:5
	jae	.LBB0_62
# %bb.50:                               # %.lr.ph41.us.preheader
                                        #   in Loop: Header=BB0_49 Depth=2
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	%r14, 80(%rsp)                  # 8-byte Spill
	movq	48(%rsp), %rbx                  # 8-byte Reload
	xorl	%r8d, %r8d
	jmp	.LBB0_51
	.p2align	4, 0x90
.LBB0_52:                               # %._crit_edge42.us
                                        #   in Loop: Header=BB0_51 Depth=3
	movq	72(%rsp), %r8                   # 8-byte Reload
	.loc	1 530 12 is_stmt 1              # <stdin>:530:12
	addq	$1, %r8
	.loc	1 465 5                         # <stdin>:465:5
	addq	$460800, %rbx                   # imm = 0x70800
	.loc	1 464 12                        # <stdin>:464:12
	cmpq	$69, %r8
	.loc	1 465 5                         # <stdin>:465:5
	je	.LBB0_53
.LBB0_51:                               # %.lr.ph41.us
                                        #   Parent Loop BB0_48 Depth=1
                                        #     Parent Loop BB0_49 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_69 Depth 4
                                        #           Child Loop BB0_70 Depth 5
                                        #             Child Loop BB0_71 Depth 6
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	%r8, 72(%rsp)                   # 8-byte Spill
	shlq	$5, %r8
	cmpq	$2168, %r8                      # imm = 0x878
	movl	$2168, %eax                     # imm = 0x878
	cmovbq	%r8, %rax
	addq	$32, %rax
	movq	64(%rsp), %rdi                  # 8-byte Reload
	movq	88(%rsp), %r13                  # 8-byte Reload
	cmpq	%rax, %r8
	.loc	1 474 5 is_stmt 1               # <stdin>:474:5
	jae	.LBB0_52
	.p2align	4, 0x90
.LBB0_69:                               # %.lr.ph41.split.us.us.us
                                        #   Parent Loop BB0_48 Depth=1
                                        #     Parent Loop BB0_49 Depth=2
                                        #       Parent Loop BB0_51 Depth=3
                                        # =>      This Loop Header: Depth=4
                                        #           Child Loop BB0_70 Depth 5
                                        #             Child Loop BB0_71 Depth 6
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	imulq	$2200, %r13, %r10               # imm = 0x898
	movq	%rbx, %r9
	movq	%r8, %rdx
	.p2align	4, 0x90
.LBB0_70:                               # %.lr.ph.us.us.us.us
                                        #   Parent Loop BB0_48 Depth=1
                                        #     Parent Loop BB0_49 Depth=2
                                        #       Parent Loop BB0_51 Depth=3
                                        #         Parent Loop BB0_69 Depth=4
                                        # =>        This Loop Header: Depth=5
                                        #             Child Loop BB0_71 Depth 6
	leaq	(%rdx,%r10), %rbp
	movsd	(%r15,%rbp,8), %xmm1            # xmm1 = mem[0],zero
	mulsd	%xmm0, %xmm1
	xorl	%ebp, %ebp
	.p2align	4, 0x90
.LBB0_71:                               #   Parent Loop BB0_48 Depth=1
                                        #     Parent Loop BB0_49 Depth=2
                                        #       Parent Loop BB0_51 Depth=3
                                        #         Parent Loop BB0_69 Depth=4
                                        #           Parent Loop BB0_70 Depth=5
                                        # =>          This Inner Loop Header: Depth=6
	movsd	(%r9,%rbp,8), %xmm2             # xmm2 = mem[0],zero
	.loc	1 513 12 is_stmt 1              # <stdin>:513:12
	mulsd	%xmm1, %xmm2
	.loc	1 514 12                        # <stdin>:514:12
	addsd	(%rdi,%rbp,8), %xmm2
	.loc	1 520 5                         # <stdin>:520:5
	movsd	%xmm2, (%rdi,%rbp,8)
	.loc	1 491 12                        # <stdin>:491:12
	leaq	(%rsi,%rbp), %r14
	addq	$1, %r14
	addq	$1, %rbp
	cmpq	%rcx, %r14
	.loc	1 492 5                         # <stdin>:492:5
	jb	.LBB0_71
# %bb.72:                               # %._crit_edge.us.us.us.us
                                        #   in Loop: Header=BB0_70 Depth=5
	.loc	1 524 12                        # <stdin>:524:12
	addq	$1, %rdx
	.loc	1 483 5                         # <stdin>:483:5
	addq	$14400, %r9                     # imm = 0x3840
	.loc	1 482 12                        # <stdin>:482:12
	cmpq	%rax, %rdx
	.loc	1 483 5                         # <stdin>:483:5
	jb	.LBB0_70
# %bb.68:                               # %._crit_edge39.us.loopexit.us.us
                                        #   in Loop: Header=BB0_69 Depth=4
	.loc	1 527 12                        # <stdin>:527:12
	addq	$1, %r13
	.loc	1 474 5                         # <stdin>:474:5
	addq	$14400, %rdi                    # imm = 0x3840
	.loc	1 473 12                        # <stdin>:473:12
	cmpq	%r11, %r13
	jne	.LBB0_69
	jmp	.LBB0_52
	.p2align	4, 0x90
.LBB0_53:                               # %.lr.ph50.preheader
                                        #   in Loop: Header=BB0_49 Depth=2
	.loc	1 0 12 is_stmt 0                # <stdin>:0:12
	cmpq	%rcx, 40(%rsp)                  # 8-byte Folded Reload
	movq	80(%rsp), %r14                  # 8-byte Reload
	.loc	1 543 5 is_stmt 1               # <stdin>:543:5
	jae	.LBB0_62
# %bb.54:                               # %.lr.ph50.us.preheader
                                        #   in Loop: Header=BB0_49 Depth=2
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	32(%rsp), %r8                   # 8-byte Reload
	movq	%r14, %r10
	xorl	%r9d, %r9d
	.p2align	4, 0x90
.LBB0_55:                               # %.lr.ph50.us
                                        #   Parent Loop BB0_48 Depth=1
                                        #     Parent Loop BB0_49 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_56 Depth 4
                                        #           Child Loop BB0_57 Depth 5
                                        #             Child Loop BB0_58 Depth 6
	movq	%r8, %rax
	movq	88(%rsp), %r13                  # 8-byte Reload
	.p2align	4, 0x90
.LBB0_56:                               # %.lr.ph44.us.us.preheader.us
                                        #   Parent Loop BB0_48 Depth=1
                                        #     Parent Loop BB0_49 Depth=2
                                        #       Parent Loop BB0_55 Depth=3
                                        # =>      This Loop Header: Depth=4
                                        #           Child Loop BB0_57 Depth 5
                                        #             Child Loop BB0_58 Depth 6
	imulq	$1800, %r13, %rdi               # imm = 0x708
	movq	%r10, %rbx
	movq	40(%rsp), %rbp                  # 8-byte Reload
	.p2align	4, 0x90
.LBB0_57:                               # %.lr.ph44.us.us.us
                                        #   Parent Loop BB0_48 Depth=1
                                        #     Parent Loop BB0_49 Depth=2
                                        #       Parent Loop BB0_55 Depth=3
                                        #         Parent Loop BB0_56 Depth=4
                                        # =>        This Loop Header: Depth=5
                                        #             Child Loop BB0_58 Depth 6
	leaq	(%rdi,%rbp), %rdx
	movsd	(%r12,%rdx,8), %xmm1            # xmm1 = mem[0],zero
	xorl	%edx, %edx
	.p2align	4, 0x90
.LBB0_58:                               #   Parent Loop BB0_48 Depth=1
                                        #     Parent Loop BB0_49 Depth=2
                                        #       Parent Loop BB0_55 Depth=3
                                        #         Parent Loop BB0_56 Depth=4
                                        #           Parent Loop BB0_57 Depth=5
                                        # =>          This Inner Loop Header: Depth=6
	movsd	(%rbx,%rdx,8), %xmm2            # xmm2 = mem[0],zero
	.loc	1 581 12 is_stmt 1              # <stdin>:581:12
	mulsd	%xmm1, %xmm2
	.loc	1 582 12                        # <stdin>:582:12
	addsd	(%rax,%rdx,8), %xmm2
	.loc	1 588 5                         # <stdin>:588:5
	movsd	%xmm2, (%rax,%rdx,8)
	.loc	1 560 12                        # <stdin>:560:12
	addq	$1, %rdx
	cmpq	$32, %rdx
	.loc	1 561 5                         # <stdin>:561:5
	jne	.LBB0_58
# %bb.59:                               # %._crit_edge45.us.us.us
                                        #   in Loop: Header=BB0_57 Depth=5
	.loc	1 592 12                        # <stdin>:592:12
	addq	$1, %rbp
	.loc	1 552 5                         # <stdin>:552:5
	addq	$19200, %rbx                    # imm = 0x4B00
	.loc	1 551 12                        # <stdin>:551:12
	cmpq	%rcx, %rbp
	.loc	1 552 5                         # <stdin>:552:5
	jb	.LBB0_57
# %bb.60:                               # %._crit_edge48.us.us
                                        #   in Loop: Header=BB0_56 Depth=4
	.loc	1 595 12                        # <stdin>:595:12
	addq	$1, %r13
	.loc	1 543 5                         # <stdin>:543:5
	addq	$19200, %rax                    # imm = 0x4B00
	.loc	1 542 12                        # <stdin>:542:12
	cmpq	%r11, %r13
	.loc	1 543 5                         # <stdin>:543:5
	jne	.LBB0_56
# %bb.61:                               # %._crit_edge51.loopexit.us
                                        #   in Loop: Header=BB0_55 Depth=3
	.loc	1 598 12                        # <stdin>:598:12
	addq	$1, %r9
	.loc	1 534 5                         # <stdin>:534:5
	addq	$256, %r10                      # imm = 0x100
	addq	$256, %r8                       # imm = 0x100
	.loc	1 533 12                        # <stdin>:533:12
	cmpq	$75, %r9
	.loc	1 534 5                         # <stdin>:534:5
	jne	.LBB0_55
	jmp	.LBB0_62
.LBB0_64:
	.loc	1 607 5                         # <stdin>:607:5
	callq	polybench_timer_stop
	.loc	1 608 5                         # <stdin>:608:5
	callq	polybench_timer_print
	.loc	1 609 12                        # <stdin>:609:12
	cmpl	$43, 124(%rsp)                  # 4-byte Folded Reload
	.loc	1 610 5                         # <stdin>:610:5
	jl	.LBB0_67
# %bb.65:
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	160(%rsp), %rax                 # 8-byte Reload
	.loc	1 612 12 is_stmt 1              # <stdin>:612:12
	movq	(%rax), %rax
	.loc	1 621 5                         # <stdin>:621:5
	testb	$1, (%rax)
	jne	.LBB0_67
# %bb.66:
	.loc	1 630 5                         # <stdin>:630:5
	movl	$1600, %edi                     # imm = 0x640
	movl	$2400, %esi                     # imm = 0x960
	movq	128(%rsp), %rcx                 # 8-byte Reload
	callq	print_array
.LBB0_67:                               # %.critedge
	.loc	1 633 5                         # <stdin>:633:5
	xorl	%eax, %eax
	addq	$216, %rsp
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
.Ltmp1:
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
                                        # -- End function
	.globl	print_array                     # -- Begin function print_array
	.p2align	4, 0x90
	.type	print_array,@function
print_array:                            # @print_array
.Lfunc_begin1:
	.loc	1 638 0                         # <stdin>:638:0
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
	pushq	%rax
	.cfi_def_cfa_offset 64
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rcx, %r12
	movl	%esi, %r14d
	movl	%edi, %ebp
.Ltmp2:
	.loc	1 651 11 prologue_end           # <stdin>:651:11
	movq	stderr(%rip), %rcx
	.loc	1 655 11                        # <stdin>:655:11
	movl	$str1, %edi
	movl	$22, %esi
	movl	$1, %edx
	callq	fwrite
	.loc	1 657 11                        # <stdin>:657:11
	movq	stderr(%rip), %rdi
	.loc	1 662 11                        # <stdin>:662:11
	movl	$str2, %esi
	movl	$str3, %edx
	xorl	%eax, %eax
	callq	fprintf
	.loc	1 665 11                        # <stdin>:665:11
	testl	%ebp, %ebp
	.loc	1 667 5                         # <stdin>:667:5
	jle	.LBB1_8
# %bb.1:
	testl	%r14d, %r14d
	jle	.LBB1_8
# %bb.2:                                # %.preheader.us.preheader
	.loc	1 665 11                        # <stdin>:665:11
	movl	%ebp, %r15d
	movl	%r14d, %r14d
	xorl	%ebp, %ebp
	xorl	%r13d, %r13d
.LBB1_3:                                # %.preheader.us
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB1_4 Depth 2
	.loc	1 0 11 is_stmt 0                # <stdin>:0:11
	xorl	%ebx, %ebx
.LBB1_4:                                #   Parent Loop BB1_3 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	.loc	1 689 11 is_stmt 1              # <stdin>:689:11
	leal	(%rbx,%rbp), %eax
	.loc	1 690 11                        # <stdin>:690:11
	imull	$-858993459, %eax, %eax         # imm = 0xCCCCCCCD
	addl	$429496728, %eax                # imm = 0x19999998
	rorl	$2, %eax
	cmpl	$214748364, %eax                # imm = 0xCCCCCCC
	.loc	1 691 5                         # <stdin>:691:5
	ja	.LBB1_6
# %bb.5:                                #   in Loop: Header=BB1_4 Depth=2
	.loc	1 694 11                        # <stdin>:694:11
	movq	stderr(%rip), %rsi
	.loc	1 697 11                        # <stdin>:697:11
	movl	$10, %edi
	callq	fputc
.LBB1_6:                                #   in Loop: Header=BB1_4 Depth=2
	.loc	1 701 11                        # <stdin>:701:11
	movq	stderr(%rip), %rdi
	.loc	1 709 11                        # <stdin>:709:11
	movsd	(%r12,%rbx,8), %xmm0            # xmm0 = mem[0],zero
	.loc	1 710 11                        # <stdin>:710:11
	movl	$str5, %esi
	movb	$1, %al
	callq	fprintf
	.loc	1 711 11                        # <stdin>:711:11
	addq	$1, %rbx
	.loc	1 683 11                        # <stdin>:683:11
	cmpq	%rbx, %r14
	.loc	1 685 5                         # <stdin>:685:5
	jne	.LBB1_4
# %bb.7:                                # %._crit_edge.us
                                        #   in Loop: Header=BB1_3 Depth=1
	.loc	1 714 11                        # <stdin>:714:11
	addq	$1, %r13
	.loc	1 667 5                         # <stdin>:667:5
	addq	$19200, %r12                    # imm = 0x4B00
	addq	%r15, %rbp
	.loc	1 665 11                        # <stdin>:665:11
	cmpq	%r15, %r13
	.loc	1 667 5                         # <stdin>:667:5
	jne	.LBB1_3
.LBB1_8:                                # %._crit_edge1
	.loc	1 670 11                        # <stdin>:670:11
	movq	stderr(%rip), %rdi
	.loc	1 675 11                        # <stdin>:675:11
	movl	$str6, %esi
	movl	$str3, %edx
	xorl	%eax, %eax
	callq	fprintf
	.loc	1 677 11                        # <stdin>:677:11
	movq	stderr(%rip), %rcx
	.loc	1 680 11                        # <stdin>:680:11
	movl	$str7, %edi
	movl	$22, %esi
	movl	$1, %edx
	addq	$8, %rsp
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
	jmp	fwrite                          # TAILCALL
.Ltmp3:
.Lfunc_end1:
	.size	print_array, .Lfunc_end1-print_array
	.cfi_endproc
                                        # -- End function
	.globl	S0                              # -- Begin function S0
	.p2align	4, 0x90
	.type	S0,@function
S0:                                     # @S0
.Lfunc_begin2:
	.loc	1 717 0                         # <stdin>:717:0
	.cfi_startproc
# %bb.0:
	.loc	1 729 11 prologue_end           # <stdin>:729:11
	imulq	$1800, 16(%rsp), %rax           # imm = 0x708
	.loc	1 730 11                        # <stdin>:730:11
	addq	24(%rsp), %rax
	.loc	1 732 5                         # <stdin>:732:5
	movq	$0, (%rsi,%rax,8)
	.loc	1 733 5                         # <stdin>:733:5
	retq
.Ltmp4:
.Lfunc_end2:
	.size	S0, .Lfunc_end2-S0
	.cfi_endproc
                                        # -- End function
	.globl	S1                              # -- Begin function S1
	.p2align	4, 0x90
	.type	S1,@function
S1:                                     # @S1
.Lfunc_begin3:
	.loc	1 735 0                         # <stdin>:735:0
	.cfi_startproc
# %bb.0:
	movq	40(%rsp), %r8
	movq	104(%rsp), %r9
	movq	88(%rsp), %rdx
	movq	24(%rsp), %rdi
	movq	16(%rsp), %rax
.Ltmp5:
	.loc	1 762 11 prologue_end           # <stdin>:762:11
	imulq	$1800, %rax, %rcx               # imm = 0x708
	.loc	1 768 11                        # <stdin>:768:11
	imulq	$2200, %rax, %rax               # imm = 0x898
	.loc	1 769 11                        # <stdin>:769:11
	addq	%rdx, %rax
	.loc	1 772 11                        # <stdin>:772:11
	mulsd	(%r9,%rax,8), %xmm0
	.loc	1 775 11                        # <stdin>:775:11
	imulq	$1800, %rdx, %rax               # imm = 0x708
	.loc	1 776 11                        # <stdin>:776:11
	addq	%rdi, %rax
	.loc	1 779 11                        # <stdin>:779:11
	mulsd	(%r8,%rax,8), %xmm0
	.loc	1 763 11                        # <stdin>:763:11
	addq	%rdi, %rcx
	.loc	1 780 11                        # <stdin>:780:11
	addsd	(%rsi,%rcx,8), %xmm0
	.loc	1 786 5                         # <stdin>:786:5
	movsd	%xmm0, (%rsi,%rcx,8)
	.loc	1 787 5                         # <stdin>:787:5
	retq
.Ltmp6:
.Lfunc_end3:
	.size	S1, .Lfunc_end3-S1
	.cfi_endproc
                                        # -- End function
	.globl	S2                              # -- Begin function S2
	.p2align	4, 0x90
	.type	S2,@function
S2:                                     # @S2
.Lfunc_begin4:
	.loc	1 789 0                         # <stdin>:789:0
	.cfi_startproc
# %bb.0:
	.loc	1 800 11 prologue_end           # <stdin>:800:11
	imulq	$2400, 16(%rsp), %rax           # imm = 0x960
	.loc	1 801 11                        # <stdin>:801:11
	addq	24(%rsp), %rax
	.loc	1 804 11                        # <stdin>:804:11
	mulsd	(%rsi,%rax,8), %xmm0
	.loc	1 810 5                         # <stdin>:810:5
	movsd	%xmm0, (%rsi,%rax,8)
	.loc	1 811 5                         # <stdin>:811:5
	retq
.Ltmp7:
.Lfunc_end4:
	.size	S2, .Lfunc_end4-S2
	.cfi_endproc
                                        # -- End function
	.globl	S3                              # -- Begin function S3
	.p2align	4, 0x90
	.type	S3,@function
S3:                                     # @S3
.Lfunc_begin5:
	.loc	1 813 0                         # <stdin>:813:0
	.cfi_startproc
# %bb.0:
	movq	40(%rsp), %r8
	movq	104(%rsp), %r9
	movq	88(%rsp), %rdx
	movq	24(%rsp), %rdi
	movq	16(%rsp), %rax
.Ltmp8:
	.loc	1 840 11 prologue_end           # <stdin>:840:11
	imulq	$2400, %rax, %rcx               # imm = 0x960
	.loc	1 846 11                        # <stdin>:846:11
	imulq	$1800, %rax, %rax               # imm = 0x708
	.loc	1 847 11                        # <stdin>:847:11
	addq	%rdx, %rax
	.loc	1 849 11                        # <stdin>:849:11
	movsd	(%r9,%rax,8), %xmm0             # xmm0 = mem[0],zero
	.loc	1 852 11                        # <stdin>:852:11
	imulq	$2400, %rdx, %rax               # imm = 0x960
	.loc	1 853 11                        # <stdin>:853:11
	addq	%rdi, %rax
	.loc	1 856 11                        # <stdin>:856:11
	mulsd	(%r8,%rax,8), %xmm0
	.loc	1 841 11                        # <stdin>:841:11
	addq	%rdi, %rcx
	.loc	1 857 11                        # <stdin>:857:11
	addsd	(%rsi,%rcx,8), %xmm0
	.loc	1 863 5                         # <stdin>:863:5
	movsd	%xmm0, (%rsi,%rcx,8)
	.loc	1 864 5                         # <stdin>:864:5
	retq
.Ltmp9:
.Lfunc_end5:
	.size	S3, .Lfunc_end5-S3
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
	.asciz	"D"
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
	.byte	3                               # Abbreviation Code
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
	.byte	5                               # DW_FORM_data2
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
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
	.byte	1                               # Abbrev [1] 0xb:0xbb DW_TAG_compile_unit
	.long	.Linfo_string0                  # DW_AT_producer
	.short	2                               # DW_AT_language
	.long	.Linfo_string1                  # DW_AT_name
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Linfo_string2                  # DW_AT_comp_dir
                                        # DW_AT_GNU_pubnames
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end5-.Lfunc_begin0       # DW_AT_high_pc
	.byte	2                               # Abbrev [2] 0x2a:0x19 DW_TAG_subprogram
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
	.long	.Linfo_string3                  # DW_AT_linkage_name
	.long	.Linfo_string3                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	3                               # Abbrev [3] 0x43:0x1a DW_TAG_subprogram
	.quad	.Lfunc_begin1                   # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
	.long	.Linfo_string4                  # DW_AT_linkage_name
	.long	.Linfo_string4                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.short	638                             # DW_AT_decl_line
                                        # DW_AT_external
	.byte	3                               # Abbrev [3] 0x5d:0x1a DW_TAG_subprogram
	.quad	.Lfunc_begin2                   # DW_AT_low_pc
	.long	.Lfunc_end2-.Lfunc_begin2       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
	.long	.Linfo_string5                  # DW_AT_linkage_name
	.long	.Linfo_string5                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.short	717                             # DW_AT_decl_line
                                        # DW_AT_external
	.byte	3                               # Abbrev [3] 0x77:0x1a DW_TAG_subprogram
	.quad	.Lfunc_begin3                   # DW_AT_low_pc
	.long	.Lfunc_end3-.Lfunc_begin3       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
	.long	.Linfo_string6                  # DW_AT_linkage_name
	.long	.Linfo_string6                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.short	735                             # DW_AT_decl_line
                                        # DW_AT_external
	.byte	3                               # Abbrev [3] 0x91:0x1a DW_TAG_subprogram
	.quad	.Lfunc_begin4                   # DW_AT_low_pc
	.long	.Lfunc_end4-.Lfunc_begin4       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
	.long	.Linfo_string7                  # DW_AT_linkage_name
	.long	.Linfo_string7                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.short	789                             # DW_AT_decl_line
                                        # DW_AT_external
	.byte	3                               # Abbrev [3] 0xab:0x1a DW_TAG_subprogram
	.quad	.Lfunc_begin5                   # DW_AT_low_pc
	.long	.Lfunc_end5-.Lfunc_begin5       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
	.long	.Linfo_string8                  # DW_AT_linkage_name
	.long	.Linfo_string8                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.short	813                             # DW_AT_decl_line
                                        # DW_AT_external
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
	.asciz	"main"                          # string offset=25
.Linfo_string4:
	.asciz	"print_array"                   # string offset=30
.Linfo_string5:
	.asciz	"S0"                            # string offset=42
.Linfo_string6:
	.asciz	"S1"                            # string offset=45
.Linfo_string7:
	.asciz	"S2"                            # string offset=48
.Linfo_string8:
	.asciz	"S3"                            # string offset=51
	.section	.debug_pubnames,"",@progbits
	.long	.LpubNames_end0-.LpubNames_begin0 # Length of Public Names Info
.LpubNames_begin0:
	.short	2                               # DWARF Version
	.long	.Lcu_begin0                     # Offset of Compilation Unit Info
	.long	198                             # Compilation Unit Length
	.long	93                              # DIE offset
	.asciz	"S0"                            # External Name
	.long	119                             # DIE offset
	.asciz	"S1"                            # External Name
	.long	42                              # DIE offset
	.asciz	"main"                          # External Name
	.long	145                             # DIE offset
	.asciz	"S2"                            # External Name
	.long	171                             # DIE offset
	.asciz	"S3"                            # External Name
	.long	67                              # DIE offset
	.asciz	"print_array"                   # External Name
	.long	0                               # End Mark
.LpubNames_end0:
	.section	.debug_pubtypes,"",@progbits
	.long	.LpubTypes_end0-.LpubTypes_begin0 # Length of Public Types Info
.LpubTypes_begin0:
	.short	2                               # DWARF Version
	.long	.Lcu_begin0                     # Offset of Compilation Unit Info
	.long	198                             # Compilation Unit Length
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
