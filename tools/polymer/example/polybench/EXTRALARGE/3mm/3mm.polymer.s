	.text
	.file	"LLVMDialectModule"
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
.Lfunc_begin0:
	.file	1 "/home/ubuntu/polymer/example/polybench/EXTRALARGE/3mm" "<stdin>"
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
	subq	$232, %rsp
	.cfi_def_cfa_offset 288
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rsi, 176(%rsp)                 # 8-byte Spill
	movl	%edi, 148(%rsp)                 # 4-byte Spill
.Ltmp0:
	.loc	1 53 11 prologue_end            # <stdin>:53:11
	movl	$23040000, %edi                 # imm = 0x15F9000
	callq	malloc
	movq	%rax, %r14
	.loc	1 71 11                         # <stdin>:71:11
	movl	$25600000, %edi                 # imm = 0x186A000
	callq	malloc
	movq	%rax, %r13
	.loc	1 89 11                         # <stdin>:89:11
	movl	$28800000, %edi                 # imm = 0x1B77400
	callq	malloc
	movq	%rax, 160(%rsp)                 # 8-byte Spill
	.loc	1 107 11                        # <stdin>:107:11
	movl	$31680000, %edi                 # imm = 0x1E36600
	callq	malloc
	movq	%rax, 120(%rsp)                 # 8-byte Spill
	.loc	1 125 12                        # <stdin>:125:12
	movl	$34560000, %edi                 # imm = 0x20F5800
	callq	malloc
	movq	%rax, %r15
	.loc	1 143 12                        # <stdin>:143:12
	movl	$42240000, %edi                 # imm = 0x2848800
	callq	malloc
	movq	%rax, 152(%rsp)                 # 8-byte Spill
	.loc	1 161 12                        # <stdin>:161:12
	movl	$28160000, %edi                 # imm = 0x1ADB000
	callq	malloc
	movq	%rax, 104(%rsp)                 # 8-byte Spill
	xorl	%eax, %eax
	movsd	.LCPI0_0(%rip), %xmm0           # xmm0 = mem[0],zero
	movq	%r13, %rcx
	.p2align	4, 0x90
.LBB0_1:                                # %.preheader42
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_2 Depth 2
	.loc	1 0 12 is_stmt 0                # <stdin>:0:12
	movl	$1, %ebp
	xorl	%esi, %esi
	.p2align	4, 0x90
.LBB0_2:                                #   Parent Loop BB0_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	.loc	1 182 12 is_stmt 1              # <stdin>:182:12
	movl	%ebp, %edi
	imulq	$1374389535, %rdi, %rdi         # imm = 0x51EB851F
	shrq	$41, %rdi
	imull	$1600, %edi, %edi               # imm = 0x640
	movl	%ebp, %edx
	subl	%edi, %edx
	.loc	1 185 12                        # <stdin>:185:12
	xorps	%xmm1, %xmm1
	cvtsi2sd	%edx, %xmm1
	.loc	1 187 12                        # <stdin>:187:12
	divsd	%xmm0, %xmm1
	.loc	1 193 5                         # <stdin>:193:5
	movsd	%xmm1, (%rcx,%rsi,8)
	.loc	1 194 12                        # <stdin>:194:12
	addq	$1, %rsi
	.loc	1 178 12                        # <stdin>:178:12
	addl	%eax, %ebp
	cmpq	$2000, %rsi                     # imm = 0x7D0
	.loc	1 180 5                         # <stdin>:180:5
	jne	.LBB0_2
# %bb.3:                                #   in Loop: Header=BB0_1 Depth=1
	.loc	1 197 12                        # <stdin>:197:12
	addq	$1, %rax
	.loc	1 176 5                         # <stdin>:176:5
	addq	$16000, %rcx                    # imm = 0x3E80
	.loc	1 174 12                        # <stdin>:174:12
	cmpq	$1600, %rax                     # imm = 0x640
	.loc	1 176 5                         # <stdin>:176:5
	jne	.LBB0_1
# %bb.4:                                # %.preheader40.preheader
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movl	$2, %r8d
	xorl	%ecx, %ecx
	movl	$2443359173, %edx               # imm = 0x91A2B3C5
	movsd	.LCPI0_1(%rip), %xmm0           # xmm0 = mem[0],zero
	movq	160(%rsp), %rsi                 # 8-byte Reload
	.p2align	4, 0x90
.LBB0_5:                                # %.preheader40
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_6 Depth 2
	movl	%r8d, %eax
	xorl	%ebp, %ebp
	.p2align	4, 0x90
.LBB0_6:                                #   Parent Loop BB0_5 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	.loc	1 209 12 is_stmt 1              # <stdin>:209:12
	movl	%eax, %ebx
	imulq	%rdx, %rbx
	shrq	$42, %rbx
	imull	$1800, %ebx, %ebx               # imm = 0x708
	movl	%eax, %edi
	subl	%ebx, %edi
	.loc	1 212 12                        # <stdin>:212:12
	xorps	%xmm1, %xmm1
	cvtsi2sd	%edi, %xmm1
	.loc	1 214 12                        # <stdin>:214:12
	divsd	%xmm0, %xmm1
	.loc	1 220 5                         # <stdin>:220:5
	movsd	%xmm1, (%rsi,%rbp,8)
	.loc	1 208 12                        # <stdin>:208:12
	addq	$1, %rbp
	.loc	1 204 12                        # <stdin>:204:12
	addl	%ecx, %eax
	cmpq	$1800, %rbp                     # imm = 0x708
	.loc	1 206 5                         # <stdin>:206:5
	jne	.LBB0_6
# %bb.7:                                #   in Loop: Header=BB0_5 Depth=1
	.loc	1 223 12                        # <stdin>:223:12
	addq	$1, %rcx
	.loc	1 202 5                         # <stdin>:202:5
	addq	$14400, %rsi                    # imm = 0x3840
	addl	$1, %r8d
	.loc	1 200 12                        # <stdin>:200:12
	cmpq	$2000, %rcx                     # imm = 0x7D0
	.loc	1 202 5                         # <stdin>:202:5
	jne	.LBB0_5
# %bb.8:                                # %.preheader38.preheader
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	xorl	%eax, %eax
	movsd	.LCPI0_2(%rip), %xmm0           # xmm0 = mem[0],zero
	movq	%r15, %rcx
	xorl	%edx, %edx
	.p2align	4, 0x90
.LBB0_9:                                # %.preheader38
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_10 Depth 2
	movl	%eax, %ebx
	xorl	%edi, %edi
	.p2align	4, 0x90
.LBB0_10:                               #   Parent Loop BB0_9 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	.loc	1 234 12 is_stmt 1              # <stdin>:234:12
	movl	%ebx, %ebp
	shrl	$3, %ebp
	imulq	$499778013, %rbp, %rbp          # imm = 0x1DCA01DD
	shrq	$37, %rbp
	imull	$2200, %ebp, %ebp               # imm = 0x898
	movl	%ebx, %esi
	subl	%ebp, %esi
	.loc	1 237 12                        # <stdin>:237:12
	xorps	%xmm1, %xmm1
	cvtsi2sd	%esi, %xmm1
	.loc	1 239 12                        # <stdin>:239:12
	divsd	%xmm0, %xmm1
	.loc	1 245 5                         # <stdin>:245:5
	movsd	%xmm1, (%rcx,%rdi,8)
	.loc	1 246 12                        # <stdin>:246:12
	addq	$1, %rdi
	.loc	1 230 12                        # <stdin>:230:12
	addl	%edx, %ebx
	cmpq	$2400, %rdi                     # imm = 0x960
	.loc	1 232 5                         # <stdin>:232:5
	jne	.LBB0_10
# %bb.11:                               #   in Loop: Header=BB0_9 Depth=1
	.loc	1 249 12                        # <stdin>:249:12
	addq	$1, %rdx
	.loc	1 228 5                         # <stdin>:228:5
	addl	$3, %eax
	addq	$19200, %rcx                    # imm = 0x4B00
	.loc	1 226 12                        # <stdin>:226:12
	cmpq	$1800, %rdx                     # imm = 0x708
	.loc	1 228 5                         # <stdin>:228:5
	jne	.LBB0_9
# %bb.12:                               # %.preheader36.preheader
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movl	$2, %ebx
	xorl	%ecx, %ecx
	movsd	.LCPI0_3(%rip), %xmm0           # xmm0 = mem[0],zero
	movq	152(%rsp), %rdx                 # 8-byte Reload
	.p2align	4, 0x90
.LBB0_13:                               # %.preheader36
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_14 Depth 2
	movl	%ebx, %eax
	xorl	%edi, %edi
	.p2align	4, 0x90
.LBB0_14:                               #   Parent Loop BB0_13 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	.loc	1 263 12 is_stmt 1              # <stdin>:263:12
	movl	%eax, %ebp
	imulq	$274877907, %rbp, %rbp          # imm = 0x10624DD3
	shrq	$39, %rbp
	imull	$2000, %ebp, %ebp               # imm = 0x7D0
	movl	%eax, %esi
	subl	%ebp, %esi
	.loc	1 267 12                        # <stdin>:267:12
	xorps	%xmm1, %xmm1
	cvtsi2sd	%esi, %xmm1
	.loc	1 269 12                        # <stdin>:269:12
	divsd	%xmm0, %xmm1
	.loc	1 275 5                         # <stdin>:275:5
	movsd	%xmm1, (%rdx,%rdi,8)
	.loc	1 276 12                        # <stdin>:276:12
	addq	$1, %rdi
	.loc	1 259 12                        # <stdin>:259:12
	addl	%ecx, %eax
	cmpq	$2200, %rdi                     # imm = 0x898
	.loc	1 261 5                         # <stdin>:261:5
	jne	.LBB0_14
# %bb.15:                               #   in Loop: Header=BB0_13 Depth=1
	.loc	1 279 12                        # <stdin>:279:12
	addq	$1, %rcx
	.loc	1 254 5                         # <stdin>:254:5
	addl	$2, %ebx
	addq	$17600, %rdx                    # imm = 0x44C0
	.loc	1 252 12                        # <stdin>:252:12
	cmpq	$2400, %rcx                     # imm = 0x960
	.loc	1 254 5                         # <stdin>:254:5
	jne	.LBB0_13
# %bb.16:
	.loc	1 256 5                         # <stdin>:256:5
	callq	polybench_timer_start
	movl	$32, %edx
	xorl	%ecx, %ecx
	movq	104(%rsp), %r12                 # 8-byte Reload
	movq	120(%rsp), %r10                 # 8-byte Reload
	xorl	%ebx, %ebx
	jmp	.LBB0_17
	.p2align	4, 0x90
.LBB0_44:                               # %.us-lcssa35.us
                                        #   in Loop: Header=BB0_17 Depth=1
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	184(%rsp), %rbx                 # 8-byte Reload
	.loc	1 385 12 is_stmt 1              # <stdin>:385:12
	addq	$1, %rbx
	movq	208(%rsp), %rdx                 # 8-byte Reload
	.loc	1 283 5                         # <stdin>:283:5
	addq	$32, %rdx
	movq	96(%rsp), %rcx                  # 8-byte Reload
	addq	$32, %rcx
	movq	192(%rsp), %r10                 # 8-byte Reload
	addq	$563200, %r10                   # imm = 0x89800
	movq	200(%rsp), %r12                 # 8-byte Reload
	addq	$563200, %r12                   # imm = 0x89800
	.loc	1 282 12                        # <stdin>:282:12
	cmpq	$107, %rbx
	.loc	1 283 5                         # <stdin>:283:5
	je	.LBB0_45
.LBB0_17:                               # %.preheader35
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_32 Depth 2
                                        #     Child Loop BB0_37 Depth 2
                                        #       Child Loop BB0_39 Depth 3
                                        #     Child Loop BB0_19 Depth 2
                                        #       Child Loop BB0_21 Depth 3
                                        #       Child Loop BB0_25 Depth 3
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	%rcx, 96(%rsp)                  # 8-byte Spill
	cmpq	$1600, %rcx                     # imm = 0x640
	movl	$1600, %eax                     # imm = 0x640
	cmovaq	%rcx, %rax
	imulq	$17600, %rax, %rbp              # imm = 0x44C0
	addq	120(%rsp), %rbp                 # 8-byte Folded Reload
	cmpq	$1600, %rdx                     # imm = 0x640
	movl	$1600, %r11d                    # imm = 0x640
	cmovbq	%rdx, %r11
	cmpq	$1800, %rcx                     # imm = 0x708
	movl	$1800, %r8d                     # imm = 0x708
	cmovaq	%rcx, %r8
	movq	%rbx, %rcx
	shlq	$5, %rcx
	cmpq	$1800, %rcx                     # imm = 0x708
	movl	$1800, %esi                     # imm = 0x708
	cmovaq	%rcx, %rsi
	movq	%rsi, 72(%rsp)                  # 8-byte Spill
	imulq	$17600, %rsi, %rax              # imm = 0x44C0
	movq	%rax, 168(%rsp)                 # 8-byte Spill
	cmpq	$1600, %rcx                     # imm = 0x640
	movl	$1600, %r9d                     # imm = 0x640
	cmovaq	%rcx, %r9
	leaq	32(%rcx), %rdi
	cmpq	$1600, %rdi                     # imm = 0x640
	movl	$1600, %eax                     # imm = 0x640
	cmovbq	%rdi, %rax
	cmpq	$1800, %rdi                     # imm = 0x708
	movl	$1800, %esi                     # imm = 0x708
	cmovaeq	%rsi, %rdi
	movq	%rax, 112(%rsp)                 # 8-byte Spill
	cmpq	%rax, %rcx
	movq	%rdi, 56(%rsp)                  # 8-byte Spill
	movq	%rdx, 208(%rsp)                 # 8-byte Spill
	movq	%r12, 200(%rsp)                 # 8-byte Spill
	movq	%r10, 192(%rsp)                 # 8-byte Spill
	movq	%rbx, 184(%rsp)                 # 8-byte Spill
	.loc	1 286 5 is_stmt 1               # <stdin>:286:5
	jae	.LBB0_30
# %bb.18:                               # %.preheader35.split.us.preheader
                                        #   in Loop: Header=BB0_17 Depth=1
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	%r9, 128(%rsp)                  # 8-byte Spill
	movq	%r12, %rbx
	xorl	%eax, %eax
	movq	%r11, %r12
	movq	%r11, 216(%rsp)                 # 8-byte Spill
	jmp	.LBB0_19
	.p2align	4, 0x90
.LBB0_29:                               # %._crit_edge90.us
                                        #   in Loop: Header=BB0_19 Depth=2
	.loc	1 382 12 is_stmt 1              # <stdin>:382:12
	addq	$1, %rax
	.loc	1 286 5                         # <stdin>:286:5
	addq	$256, %r10                      # imm = 0x100
	addq	$256, %rbx                      # imm = 0x100
	addq	$256, %rbp                      # imm = 0x100
	.loc	1 285 12                        # <stdin>:285:12
	cmpq	$69, %rax
	.loc	1 286 5                         # <stdin>:286:5
	je	.LBB0_44
.LBB0_19:                               # %.preheader35.split.us
                                        #   Parent Loop BB0_17 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_21 Depth 3
                                        #       Child Loop BB0_25 Depth 3
	.loc	1 295 5                         # <stdin>:295:5
	movq	%rax, %rdx
	shlq	$5, %rdx
	cmpq	$2168, %rdx                     # imm = 0x878
	movl	$2168, %esi                     # imm = 0x878
	cmovbq	%rdx, %rsi
	addq	$32, %rsi
	movq	%rdx, %rcx
	subq	%rsi, %rcx
	jae	.LBB0_29
# %bb.20:                               # %.lr.ph77.split.us.us.preheader
                                        #   in Loop: Header=BB0_19 Depth=2
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	%rsi, 88(%rsp)                  # 8-byte Spill
	movq	%rdx, 48(%rsp)                  # 8-byte Spill
	movq	%rbp, 80(%rsp)                  # 8-byte Spill
	movq	%rax, 136(%rsp)                 # 8-byte Spill
	shlq	$8, %rax
	addq	168(%rsp), %rax                 # 8-byte Folded Reload
	movq	%rax, 224(%rsp)                 # 8-byte Spill
	negq	%rcx
	shlq	$3, %rcx
	movq	%rbx, 64(%rsp)                  # 8-byte Spill
	xorl	%ebx, %ebx
	movq	%r10, 32(%rsp)                  # 8-byte Spill
	.p2align	4, 0x90
.LBB0_21:                               # %.lr.ph77.split.us.us
                                        #   Parent Loop BB0_17 Depth=1
                                        #     Parent Loop BB0_19 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movq	64(%rsp), %rax                  # 8-byte Reload
	addq	%rbx, %rax
	movq	%rax, 40(%rsp)                  # 8-byte Spill
	movq	32(%rsp), %rax                  # 8-byte Reload
	.loc	1 311 5 is_stmt 1               # <stdin>:311:5
	leaq	(%rax,%rbx), %rdi
	xorl	%esi, %esi
	movq	%rcx, %rbp
	movq	%rcx, %rdx
	callq	memset
	movq	40(%rsp), %rdi                  # 8-byte Reload
	.loc	1 317 5                         # <stdin>:317:5
	xorl	%esi, %esi
	movq	%rbp, %rdx
	callq	memset
	movq	%rbp, %rcx
	.loc	1 294 12                        # <stdin>:294:12
	addq	$-1, %r12
	addq	$17600, %rbx                    # imm = 0x44C0
	cmpq	%r12, 96(%rsp)                  # 8-byte Folded Reload
	.loc	1 295 5                         # <stdin>:295:5
	jne	.LBB0_21
# %bb.22:                               # %._crit_edge78.us
                                        #   in Loop: Header=BB0_19 Depth=2
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	56(%rsp), %rax                  # 8-byte Reload
	cmpq	%rax, 128(%rsp)                 # 8-byte Folded Reload
	.loc	1 331 5 is_stmt 1               # <stdin>:331:5
	jge	.LBB0_26
# %bb.23:                               # %._crit_edge78.us
                                        #   in Loop: Header=BB0_19 Depth=2
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	88(%rsp), %rax                  # 8-byte Reload
	.loc	1 331 5                         # <stdin>:331:5
	cmpq	%rax, 48(%rsp)                  # 8-byte Folded Reload
	jae	.LBB0_26
# %bb.24:                               # %.lr.ph83.split.us.us.preheader
                                        #   in Loop: Header=BB0_19 Depth=2
	.loc	1 0 5                           # <stdin>:0:5
	movq	80(%rsp), %r12                  # 8-byte Reload
	movq	128(%rsp), %rbx                 # 8-byte Reload
	.p2align	4, 0x90
.LBB0_25:                               # %.lr.ph83.split.us.us
                                        #   Parent Loop BB0_17 Depth=1
                                        #     Parent Loop BB0_19 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	.loc	1 347 5 is_stmt 1               # <stdin>:347:5
	movq	%r12, %rdi
	xorl	%esi, %esi
	movq	%rbp, %rdx
	callq	memset
	movq	%rbp, %rcx
	.loc	1 351 12                        # <stdin>:351:12
	addq	$1, %rbx
	.loc	1 330 12                        # <stdin>:330:12
	addq	$17600, %r12                    # imm = 0x44C0
	cmpq	56(%rsp), %rbx                  # 8-byte Folded Reload
	.loc	1 331 5                         # <stdin>:331:5
	jb	.LBB0_25
.LBB0_26:                               # %._crit_edge84.us
                                        #   in Loop: Header=BB0_19 Depth=2
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	112(%rsp), %rax                 # 8-byte Reload
	cmpq	%rax, 72(%rsp)                  # 8-byte Folded Reload
	movq	216(%rsp), %r12                 # 8-byte Reload
	movq	64(%rsp), %rbx                  # 8-byte Reload
	movq	32(%rsp), %r10                  # 8-byte Reload
	movq	136(%rsp), %rax                 # 8-byte Reload
	movq	%rcx, %rdx
	movq	48(%rsp), %rcx                  # 8-byte Reload
	movq	88(%rsp), %rsi                  # 8-byte Reload
	movq	80(%rsp), %rbp                  # 8-byte Reload
	.loc	1 359 5 is_stmt 1               # <stdin>:359:5
	jge	.LBB0_29
# %bb.27:                               # %._crit_edge84.us
                                        #   in Loop: Header=BB0_19 Depth=2
	cmpq	%rsi, %rcx
	jae	.LBB0_29
# %bb.28:                               # %._crit_edge90.loopexit.us
                                        #   in Loop: Header=BB0_19 Depth=2
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	224(%rsp), %rdi                 # 8-byte Reload
	addq	104(%rsp), %rdi                 # 8-byte Folded Reload
	.loc	1 375 5 is_stmt 1               # <stdin>:375:5
	xorl	%esi, %esi
	callq	memset
	movq	136(%rsp), %rax                 # 8-byte Reload
	movq	32(%rsp), %r10                  # 8-byte Reload
	jmp	.LBB0_29
	.p2align	4, 0x90
.LBB0_30:                               # %.preheader35.split
                                        #   in Loop: Header=BB0_17 Depth=1
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	cmpq	%rdi, %r9
	.loc	1 286 5 is_stmt 1               # <stdin>:286:5
	jge	.LBB0_31
# %bb.36:                               # %.preheader35.split.split.us.preheader
                                        #   in Loop: Header=BB0_17 Depth=1
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	%r9, 128(%rsp)                  # 8-byte Spill
	xorl	%ebx, %ebx
	jmp	.LBB0_37
	.p2align	4, 0x90
.LBB0_43:                               # %._crit_edge90.us54
                                        #   in Loop: Header=BB0_37 Depth=2
	.loc	1 382 12 is_stmt 1              # <stdin>:382:12
	addq	$1, %rbx
	.loc	1 286 5                         # <stdin>:286:5
	addq	$256, %rbp                      # imm = 0x100
	.loc	1 285 12                        # <stdin>:285:12
	cmpq	$69, %rbx
	.loc	1 286 5                         # <stdin>:286:5
	je	.LBB0_44
.LBB0_37:                               # %.preheader35.split.split.us
                                        #   Parent Loop BB0_17 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_39 Depth 3
	.loc	1 295 5                         # <stdin>:295:5
	movq	%rbx, %rax
	shlq	$5, %rax
	cmpq	$2168, %rax                     # imm = 0x878
	movl	$2168, %ecx                     # imm = 0x878
	cmovbq	%rax, %rcx
	addq	$32, %rcx
	movq	%rax, %rdx
	subq	%rcx, %rdx
	.loc	1 331 5                         # <stdin>:331:5
	jae	.LBB0_43
# %bb.38:                               # %.lr.ph83.split.us.us44.preheader
                                        #   in Loop: Header=BB0_37 Depth=2
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	%rcx, 64(%rsp)                  # 8-byte Spill
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	movq	%rbx, 40(%rsp)                  # 8-byte Spill
	shlq	$8, %rbx
	addq	168(%rsp), %rbx                 # 8-byte Folded Reload
	movq	%rbx, 48(%rsp)                  # 8-byte Spill
	negq	%rdx
	shlq	$3, %rdx
	movq	%rbp, 80(%rsp)                  # 8-byte Spill
	movq	128(%rsp), %r12                 # 8-byte Reload
	.p2align	4, 0x90
.LBB0_39:                               # %.lr.ph83.split.us.us44
                                        #   Parent Loop BB0_17 Depth=1
                                        #     Parent Loop BB0_37 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	.loc	1 347 5 is_stmt 1               # <stdin>:347:5
	movq	%rbp, %rdi
	xorl	%esi, %esi
	movq	%rdx, %rbx
	callq	memset
	movq	%rbx, %rdx
	.loc	1 351 12                        # <stdin>:351:12
	addq	$1, %r12
	.loc	1 330 12                        # <stdin>:330:12
	addq	$17600, %rbp                    # imm = 0x44C0
	cmpq	56(%rsp), %r12                  # 8-byte Folded Reload
	.loc	1 331 5                         # <stdin>:331:5
	jb	.LBB0_39
# %bb.40:                               # %._crit_edge84.us50
                                        #   in Loop: Header=BB0_37 Depth=2
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	112(%rsp), %rax                 # 8-byte Reload
	cmpq	%rax, 72(%rsp)                  # 8-byte Folded Reload
	movq	80(%rsp), %rbp                  # 8-byte Reload
	movq	40(%rsp), %rbx                  # 8-byte Reload
	movq	32(%rsp), %rax                  # 8-byte Reload
	movq	64(%rsp), %rcx                  # 8-byte Reload
	.loc	1 359 5 is_stmt 1               # <stdin>:359:5
	jge	.LBB0_43
# %bb.41:                               # %._crit_edge84.us50
                                        #   in Loop: Header=BB0_37 Depth=2
	cmpq	%rcx, %rax
	jae	.LBB0_43
# %bb.42:                               # %._crit_edge90.loopexit.us52
                                        #   in Loop: Header=BB0_37 Depth=2
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	48(%rsp), %rdi                  # 8-byte Reload
	addq	104(%rsp), %rdi                 # 8-byte Folded Reload
	.loc	1 375 5 is_stmt 1               # <stdin>:375:5
	xorl	%esi, %esi
	callq	memset
	jmp	.LBB0_43
	.p2align	4, 0x90
.LBB0_31:                               # %.preheader35.split.split.preheader
                                        #   in Loop: Header=BB0_17 Depth=1
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	imulq	$17600, %r8, %r12               # imm = 0x44C0
	addq	104(%rsp), %r12                 # 8-byte Folded Reload
	movl	$32, %ebp
	xorl	%ebx, %ebx
	jmp	.LBB0_32
	.p2align	4, 0x90
.LBB0_35:                               # %._crit_edge90
                                        #   in Loop: Header=BB0_32 Depth=2
	.loc	1 285 12 is_stmt 1              # <stdin>:285:12
	addq	$32, %rbx
	addq	$-32, %rbp
	cmpq	$2208, %rbx                     # imm = 0x8A0
	je	.LBB0_44
.LBB0_32:                               # %.preheader35.split.split
                                        #   Parent Loop BB0_17 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	.loc	1 295 5                         # <stdin>:295:5
	cmpq	$2168, %rbx                     # imm = 0x878
	movl	$2168, %edx                     # imm = 0x878
	cmovbq	%rbx, %rdx
	movq	112(%rsp), %rax                 # 8-byte Reload
	cmpq	%rax, 72(%rsp)                  # 8-byte Folded Reload
	.loc	1 359 5                         # <stdin>:359:5
	jge	.LBB0_35
# %bb.33:                               # %.preheader35.split.split
                                        #   in Loop: Header=BB0_32 Depth=2
	.loc	1 0 0 is_stmt 0                 # <stdin>:0:0
	leaq	32(%rdx), %rax
	.loc	1 359 5                         # <stdin>:359:5
	cmpq	%rax, %rbx
	jae	.LBB0_35
# %bb.34:                               # %._crit_edge90.loopexit
                                        #   in Loop: Header=BB0_32 Depth=2
	.loc	1 0 0                           # <stdin>:0:0
	addq	%rbp, %rdx
	.loc	1 295 5 is_stmt 1               # <stdin>:295:5
	shlq	$3, %rdx
	.loc	1 375 5                         # <stdin>:375:5
	leaq	(%r12,%rbx,8), %rdi
	xorl	%esi, %esi
	callq	memset
	jmp	.LBB0_35
.LBB0_45:                               # %.preheader33.preheader
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	xorl	%eax, %eax
	movq	120(%rsp), %rcx                 # 8-byte Reload
	jmp	.LBB0_46
	.p2align	4, 0x90
.LBB0_58:                               #   in Loop: Header=BB0_46 Depth=1
	movq	88(%rsp), %rax                  # 8-byte Reload
	.loc	1 465 12 is_stmt 1              # <stdin>:465:12
	addq	$1, %rax
	movq	72(%rsp), %rcx                  # 8-byte Reload
	.loc	1 389 5                         # <stdin>:389:5
	addq	$563200, %rcx                   # imm = 0x89800
	.loc	1 388 12                        # <stdin>:388:12
	cmpq	$57, %rax
	.loc	1 389 5                         # <stdin>:389:5
	je	.LBB0_59
.LBB0_46:                               # %.preheader33
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_47 Depth 2
                                        #       Child Loop BB0_49 Depth 3
                                        #         Child Loop BB0_51 Depth 4
                                        #           Child Loop BB0_52 Depth 5
                                        #             Child Loop BB0_53 Depth 6
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	%rax, 88(%rsp)                  # 8-byte Spill
	shlq	$5, %rax
	cmpq	$1768, %rax                     # imm = 0x6E8
	movl	$1768, %edx                     # imm = 0x6E8
	movq	%rax, 56(%rsp)                  # 8-byte Spill
	cmovbq	%rax, %rdx
	addq	$32, %rdx
	movq	%rdx, 40(%rsp)                  # 8-byte Spill
	movq	%rcx, 72(%rsp)                  # 8-byte Spill
	movq	%rcx, 96(%rsp)                  # 8-byte Spill
	movq	152(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 48(%rsp)                  # 8-byte Spill
	xorl	%eax, %eax
	xorl	%ecx, %ecx
	jmp	.LBB0_47
	.p2align	4, 0x90
.LBB0_57:                               # %.us-lcssa26.us
                                        #   in Loop: Header=BB0_47 Depth=2
	movq	80(%rsp), %rcx                  # 8-byte Reload
	.loc	1 462 12 is_stmt 1              # <stdin>:462:12
	addq	$1, %rcx
	.loc	1 392 5                         # <stdin>:392:5
	addq	$32, %rax
	addq	$256, 48(%rsp)                  # 8-byte Folded Spill
                                        # imm = 0x100
	addq	$256, 96(%rsp)                  # 8-byte Folded Spill
                                        # imm = 0x100
	.loc	1 391 12                        # <stdin>:391:12
	cmpq	$69, %rcx
	.loc	1 392 5                         # <stdin>:392:5
	je	.LBB0_58
.LBB0_47:                               # %.preheader32
                                        #   Parent Loop BB0_46 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_49 Depth 3
                                        #         Child Loop BB0_51 Depth 4
                                        #           Child Loop BB0_52 Depth 5
                                        #             Child Loop BB0_53 Depth 6
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	%rcx, 80(%rsp)                  # 8-byte Spill
	shlq	$5, %rcx
	cmpq	$2168, %rcx                     # imm = 0x878
	movl	$2168, %edx                     # imm = 0x878
	movq	%rcx, 64(%rsp)                  # 8-byte Spill
	cmovbq	%rcx, %rdx
	movq	40(%rsp), %rcx                  # 8-byte Reload
	cmpq	%rcx, 56(%rsp)                  # 8-byte Folded Reload
	.loc	1 395 5 is_stmt 1               # <stdin>:395:5
	jae	.LBB0_57
# %bb.48:                               # %.preheader32.split.us.preheader
                                        #   in Loop: Header=BB0_47 Depth=2
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	addq	$32, %rdx
	movl	$32, %edi
	movq	48(%rsp), %r10                  # 8-byte Reload
	xorl	%ecx, %ecx
	movq	%rcx, 32(%rsp)                  # 8-byte Spill
	jmp	.LBB0_49
	.p2align	4, 0x90
.LBB0_56:                               # %._crit_edge72.loopexit.us
                                        #   in Loop: Header=BB0_49 Depth=3
	movq	32(%rsp), %rsi                  # 8-byte Reload
	.loc	1 459 12 is_stmt 1              # <stdin>:459:12
	addq	$1, %rsi
	.loc	1 395 5                         # <stdin>:395:5
	addq	$32, %rdi
	addq	$563200, %r10                   # imm = 0x89800
	movq	%rsi, %rcx
	movq	%rsi, 32(%rsp)                  # 8-byte Spill
	.loc	1 394 12                        # <stdin>:394:12
	cmpq	$75, %rsi
	.loc	1 395 5                         # <stdin>:395:5
	je	.LBB0_57
.LBB0_49:                               # %.preheader32.split.us
                                        #   Parent Loop BB0_46 Depth=1
                                        #     Parent Loop BB0_47 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_51 Depth 4
                                        #           Child Loop BB0_52 Depth 5
                                        #             Child Loop BB0_53 Depth 6
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	cmpq	%rdx, 64(%rsp)                  # 8-byte Folded Reload
	.loc	1 413 5 is_stmt 1               # <stdin>:413:5
	jae	.LBB0_56
# %bb.50:                               #   in Loop: Header=BB0_49 Depth=3
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	32(%rsp), %r9                   # 8-byte Reload
	shlq	$5, %r9
	movq	96(%rsp), %r12                  # 8-byte Reload
	movq	56(%rsp), %rbx                  # 8-byte Reload
	.p2align	4, 0x90
.LBB0_51:                               # %.lr.ph71.split.us.us.us
                                        #   Parent Loop BB0_46 Depth=1
                                        #     Parent Loop BB0_47 Depth=2
                                        #       Parent Loop BB0_49 Depth=3
                                        # =>      This Loop Header: Depth=4
                                        #           Child Loop BB0_52 Depth 5
                                        #             Child Loop BB0_53 Depth 6
	imulq	$2400, %rbx, %r11               # imm = 0x960
	movq	%r10, %rcx
	movq	%r9, %r8
	.p2align	4, 0x90
.LBB0_52:                               # %.lr.ph65.us.us.us.us
                                        #   Parent Loop BB0_46 Depth=1
                                        #     Parent Loop BB0_47 Depth=2
                                        #       Parent Loop BB0_49 Depth=3
                                        #         Parent Loop BB0_51 Depth=4
                                        # =>        This Loop Header: Depth=5
                                        #             Child Loop BB0_53 Depth 6
	leaq	(%r8,%r11), %rbp
	movsd	(%r15,%rbp,8), %xmm0            # xmm0 = mem[0],zero
	xorl	%ebp, %ebp
	.p2align	4, 0x90
.LBB0_53:                               #   Parent Loop BB0_46 Depth=1
                                        #     Parent Loop BB0_47 Depth=2
                                        #       Parent Loop BB0_49 Depth=3
                                        #         Parent Loop BB0_51 Depth=4
                                        #           Parent Loop BB0_52 Depth=5
                                        # =>          This Inner Loop Header: Depth=6
	movsd	(%rcx,%rbp,8), %xmm1            # xmm1 = mem[0],zero
	.loc	1 442 12 is_stmt 1              # <stdin>:442:12
	mulsd	%xmm0, %xmm1
	.loc	1 443 12                        # <stdin>:443:12
	addsd	(%r12,%rbp,8), %xmm1
	.loc	1 449 5                         # <stdin>:449:5
	movsd	%xmm1, (%r12,%rbp,8)
	.loc	1 421 12                        # <stdin>:421:12
	leaq	(%rax,%rbp), %rsi
	addq	$1, %rsi
	addq	$1, %rbp
	cmpq	%rdx, %rsi
	.loc	1 422 5                         # <stdin>:422:5
	jb	.LBB0_53
# %bb.54:                               # %._crit_edge66.us.us.us.us
                                        #   in Loop: Header=BB0_52 Depth=5
	.loc	1 453 12                        # <stdin>:453:12
	addq	$1, %r8
	.loc	1 413 5                         # <stdin>:413:5
	addq	$17600, %rcx                    # imm = 0x44C0
	.loc	1 412 12                        # <stdin>:412:12
	cmpq	%rdi, %r8
	.loc	1 413 5                         # <stdin>:413:5
	jne	.LBB0_52
# %bb.55:                               # %._crit_edge69.us.loopexit.us.us
                                        #   in Loop: Header=BB0_51 Depth=4
	.loc	1 456 12                        # <stdin>:456:12
	addq	$1, %rbx
	.loc	1 404 5                         # <stdin>:404:5
	addq	$17600, %r12                    # imm = 0x44C0
	.loc	1 403 12                        # <stdin>:403:12
	cmpq	40(%rsp), %rbx                  # 8-byte Folded Reload
	.loc	1 404 5                         # <stdin>:404:5
	jb	.LBB0_51
	jmp	.LBB0_56
.LBB0_59:                               # %.preheader30.split.us.preheader.preheader
	.loc	1 481 5                         # <stdin>:481:5
	movq	%r14, %rbx
	addq	$446400, %rbx                   # imm = 0x6CFC0
	xorl	%eax, %eax
	jmp	.LBB0_60
	.p2align	4, 0x90
.LBB0_64:                               # %.us-lcssa.us
                                        #   in Loop: Header=BB0_60 Depth=1
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	32(%rsp), %rax                  # 8-byte Reload
	.loc	1 507 12 is_stmt 1              # <stdin>:507:12
	addq	$1, %rax
	.loc	1 469 5                         # <stdin>:469:5
	addq	$460800, %rbx                   # imm = 0x70800
	.loc	1 468 12                        # <stdin>:468:12
	cmpq	$50, %rax
	.loc	1 469 5                         # <stdin>:469:5
	je	.LBB0_65
.LBB0_60:                               # %.preheader30.split.us.preheader
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_61 Depth 2
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	movl	$32, %ebp
	xorl	%r12d, %r12d
	jmp	.LBB0_61
	.p2align	4, 0x90
.LBB0_63:                               # %._crit_edge63.us
                                        #   in Loop: Header=BB0_61 Depth=2
	.loc	1 471 12 is_stmt 1              # <stdin>:471:12
	addq	$32, %r12
	addq	$-32, %rbp
	cmpq	$1824, %r12                     # imm = 0x720
	.loc	1 472 5                         # <stdin>:472:5
	je	.LBB0_64
.LBB0_61:                               # %.preheader30.split.us
                                        #   Parent Loop BB0_60 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	cmpq	$1768, %r12                     # imm = 0x6E8
	movl	$1768, %r15d                    # imm = 0x6E8
	cmovbq	%r12, %r15
	leaq	32(%r15), %rax
	cmpq	%rax, %r12
	.loc	1 481 5 is_stmt 1               # <stdin>:481:5
	jae	.LBB0_63
# %bb.62:                               # %.lr.ph59.us.us.preheader
                                        #   in Loop: Header=BB0_61 Depth=2
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	addq	%rbp, %r15
	shlq	$3, %r15
	.loc	1 497 5 is_stmt 1               # <stdin>:497:5
	leaq	(%rbx,%r12,8), %rax
	movq	%rax, 40(%rsp)                  # 8-byte Spill
	leaq	(%rbx,%r12,8), %rdi
	addq	$-446400, %rdi                  # imm = 0xFFF93040
	xorl	%esi, %esi
	movq	%r15, %rdx
	callq	memset
	leaq	(%rbx,%r12,8), %rdi
	addq	$-432000, %rdi                  # imm = 0xFFF96880
	xorl	%esi, %esi
	movq	%r15, %rdx
	callq	memset
	leaq	(%rbx,%r12,8), %rdi
	addq	$-417600, %rdi                  # imm = 0xFFF9A0C0
	xorl	%esi, %esi
	movq	%r15, %rdx
	callq	memset
	leaq	(%rbx,%r12,8), %rdi
	addq	$-403200, %rdi                  # imm = 0xFFF9D900
	xorl	%esi, %esi
	movq	%r15, %rdx
	callq	memset
	leaq	(%rbx,%r12,8), %rdi
	addq	$-388800, %rdi                  # imm = 0xFFFA1140
	xorl	%esi, %esi
	movq	%r15, %rdx
	callq	memset
	leaq	(%rbx,%r12,8), %rdi
	addq	$-374400, %rdi                  # imm = 0xFFFA4980
	xorl	%esi, %esi
	movq	%r15, %rdx
	callq	memset
	leaq	(%rbx,%r12,8), %rdi
	addq	$-360000, %rdi                  # imm = 0xFFFA81C0
	xorl	%esi, %esi
	movq	%r15, %rdx
	callq	memset
	leaq	(%rbx,%r12,8), %rdi
	addq	$-345600, %rdi                  # imm = 0xFFFABA00
	xorl	%esi, %esi
	movq	%r15, %rdx
	callq	memset
	leaq	(%rbx,%r12,8), %rdi
	addq	$-331200, %rdi                  # imm = 0xFFFAF240
	xorl	%esi, %esi
	movq	%r15, %rdx
	callq	memset
	leaq	(%rbx,%r12,8), %rdi
	addq	$-316800, %rdi                  # imm = 0xFFFB2A80
	xorl	%esi, %esi
	movq	%r15, %rdx
	callq	memset
	leaq	(%rbx,%r12,8), %rdi
	addq	$-302400, %rdi                  # imm = 0xFFFB62C0
	xorl	%esi, %esi
	movq	%r15, %rdx
	callq	memset
	leaq	(%rbx,%r12,8), %rdi
	addq	$-288000, %rdi                  # imm = 0xFFFB9B00
	xorl	%esi, %esi
	movq	%r15, %rdx
	callq	memset
	leaq	(%rbx,%r12,8), %rdi
	addq	$-273600, %rdi                  # imm = 0xFFFBD340
	xorl	%esi, %esi
	movq	%r15, %rdx
	callq	memset
	leaq	(%rbx,%r12,8), %rdi
	addq	$-259200, %rdi                  # imm = 0xFFFC0B80
	xorl	%esi, %esi
	movq	%r15, %rdx
	callq	memset
	leaq	(%rbx,%r12,8), %rdi
	addq	$-244800, %rdi                  # imm = 0xFFFC43C0
	xorl	%esi, %esi
	movq	%r15, %rdx
	callq	memset
	leaq	(%rbx,%r12,8), %rdi
	addq	$-230400, %rdi                  # imm = 0xFFFC7C00
	xorl	%esi, %esi
	movq	%r15, %rdx
	callq	memset
	leaq	(%rbx,%r12,8), %rdi
	addq	$-216000, %rdi                  # imm = 0xFFFCB440
	xorl	%esi, %esi
	movq	%r15, %rdx
	callq	memset
	leaq	(%rbx,%r12,8), %rdi
	addq	$-201600, %rdi                  # imm = 0xFFFCEC80
	xorl	%esi, %esi
	movq	%r15, %rdx
	callq	memset
	leaq	(%rbx,%r12,8), %rdi
	addq	$-187200, %rdi                  # imm = 0xFFFD24C0
	xorl	%esi, %esi
	movq	%r15, %rdx
	callq	memset
	leaq	(%rbx,%r12,8), %rdi
	addq	$-172800, %rdi                  # imm = 0xFFFD5D00
	xorl	%esi, %esi
	movq	%r15, %rdx
	callq	memset
	leaq	(%rbx,%r12,8), %rdi
	addq	$-158400, %rdi                  # imm = 0xFFFD9540
	xorl	%esi, %esi
	movq	%r15, %rdx
	callq	memset
	leaq	(%rbx,%r12,8), %rdi
	addq	$-144000, %rdi                  # imm = 0xFFFDCD80
	xorl	%esi, %esi
	movq	%r15, %rdx
	callq	memset
	leaq	(%rbx,%r12,8), %rdi
	addq	$-129600, %rdi                  # imm = 0xFFFE05C0
	xorl	%esi, %esi
	movq	%r15, %rdx
	callq	memset
	leaq	(%rbx,%r12,8), %rdi
	addq	$-115200, %rdi                  # imm = 0xFFFE3E00
	xorl	%esi, %esi
	movq	%r15, %rdx
	callq	memset
	leaq	(%rbx,%r12,8), %rdi
	addq	$-100800, %rdi                  # imm = 0xFFFE7640
	xorl	%esi, %esi
	movq	%r15, %rdx
	callq	memset
	leaq	(%rbx,%r12,8), %rdi
	addq	$-86400, %rdi                   # imm = 0xFFFEAE80
	xorl	%esi, %esi
	movq	%r15, %rdx
	callq	memset
	leaq	(%rbx,%r12,8), %rdi
	addq	$-72000, %rdi                   # imm = 0xFFFEE6C0
	xorl	%esi, %esi
	movq	%r15, %rdx
	callq	memset
	leaq	(%rbx,%r12,8), %rdi
	addq	$-57600, %rdi                   # imm = 0xFFFF1F00
	xorl	%esi, %esi
	movq	%r15, %rdx
	callq	memset
	leaq	(%rbx,%r12,8), %rdi
	addq	$-43200, %rdi                   # imm = 0xFFFF5740
	xorl	%esi, %esi
	movq	%r15, %rdx
	callq	memset
	leaq	(%rbx,%r12,8), %rdi
	addq	$-28800, %rdi                   # imm = 0x8F80
	xorl	%esi, %esi
	movq	%r15, %rdx
	callq	memset
	leaq	(%rbx,%r12,8), %rdi
	addq	$-14400, %rdi                   # imm = 0xC7C0
	xorl	%esi, %esi
	movq	%r15, %rdx
	callq	memset
	movq	40(%rsp), %rdi                  # 8-byte Reload
	xorl	%esi, %esi
	movq	%r15, %rdx
	callq	memset
	jmp	.LBB0_63
.LBB0_65:                               # %.preheader28.preheader
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movl	$32, %r8d
	xorl	%eax, %eax
	movq	104(%rsp), %rcx                 # 8-byte Reload
	movq	%rcx, 72(%rsp)                  # 8-byte Spill
	movq	%r14, %rcx
	jmp	.LBB0_66
	.p2align	4, 0x90
.LBB0_76:                               #   in Loop: Header=BB0_66 Depth=1
	movq	112(%rsp), %rax                 # 8-byte Reload
	.loc	1 655 12 is_stmt 1              # <stdin>:655:12
	addq	$1, %rax
	.loc	1 511 5                         # <stdin>:511:5
	addq	$32, %r8
	movq	136(%rsp), %rcx                 # 8-byte Reload
	addq	$460800, %rcx                   # imm = 0x70800
	addq	$563200, 72(%rsp)               # 8-byte Folded Spill
                                        # imm = 0x89800
	.loc	1 510 12                        # <stdin>:510:12
	cmpq	$50, %rax
	.loc	1 511 5                         # <stdin>:511:5
	je	.LBB0_77
.LBB0_66:                               # %.preheader28
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_67 Depth 2
                                        #       Child Loop BB0_69 Depth 3
                                        #         Child Loop BB0_82 Depth 4
                                        #           Child Loop BB0_83 Depth 5
                                        #             Child Loop BB0_84 Depth 6
                                        #       Child Loop BB0_73 Depth 3
                                        #         Child Loop BB0_87 Depth 4
                                        #           Child Loop BB0_88 Depth 5
                                        #             Child Loop BB0_89 Depth 6
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	%rax, 112(%rsp)                 # 8-byte Spill
	shlq	$5, %rax
	movq	%rax, 64(%rsp)                  # 8-byte Spill
	movq	120(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 88(%rsp)                  # 8-byte Spill
	movq	%rcx, 136(%rsp)                 # 8-byte Spill
	movq	%rcx, 56(%rsp)                  # 8-byte Spill
	movq	160(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 48(%rsp)                  # 8-byte Spill
	xorl	%esi, %esi
	xorl	%eax, %eax
	jmp	.LBB0_67
	.p2align	4, 0x90
.LBB0_75:                               # %.us-lcssa.us25
                                        #   in Loop: Header=BB0_67 Depth=2
	movq	80(%rsp), %rax                  # 8-byte Reload
	.loc	1 652 12 is_stmt 1              # <stdin>:652:12
	addq	$1, %rax
	.loc	1 514 5                         # <stdin>:514:5
	addq	$32, %rsi
	addq	$256, 48(%rsp)                  # 8-byte Folded Spill
                                        # imm = 0x100
	addq	$256, 56(%rsp)                  # 8-byte Folded Spill
                                        # imm = 0x100
	addq	$563200, 88(%rsp)               # 8-byte Folded Spill
                                        # imm = 0x89800
	.loc	1 513 12                        # <stdin>:513:12
	cmpq	$57, %rax
	.loc	1 514 5                         # <stdin>:514:5
	je	.LBB0_76
.LBB0_67:                               # %.preheader27
                                        #   Parent Loop BB0_66 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_69 Depth 3
                                        #         Child Loop BB0_82 Depth 4
                                        #           Child Loop BB0_83 Depth 5
                                        #             Child Loop BB0_84 Depth 6
                                        #       Child Loop BB0_73 Depth 3
                                        #         Child Loop BB0_87 Depth 4
                                        #           Child Loop BB0_88 Depth 5
                                        #             Child Loop BB0_89 Depth 6
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	%rax, 80(%rsp)                  # 8-byte Spill
	shlq	$5, %rax
	cmpq	$1768, %rax                     # imm = 0x6E8
	movl	$1768, %ecx                     # imm = 0x6E8
	cmovbq	%rax, %rcx
	addq	$32, %rcx
	movq	%rax, 40(%rsp)                  # 8-byte Spill
	cmpq	%rcx, %rax
	.loc	1 517 5 is_stmt 1               # <stdin>:517:5
	jae	.LBB0_75
# %bb.68:                               # %.lr.ph47.us.preheader
                                        #   in Loop: Header=BB0_67 Depth=2
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	48(%rsp), %r10                  # 8-byte Reload
	xorl	%r11d, %r11d
	jmp	.LBB0_69
	.p2align	4, 0x90
.LBB0_70:                               # %._crit_edge48.us
                                        #   in Loop: Header=BB0_69 Depth=3
	movq	32(%rsp), %r11                  # 8-byte Reload
	.loc	1 581 12 is_stmt 1              # <stdin>:581:12
	addq	$1, %r11
	.loc	1 517 5                         # <stdin>:517:5
	addq	$460800, %r10                   # imm = 0x70800
	.loc	1 516 12                        # <stdin>:516:12
	cmpq	$63, %r11
	.loc	1 517 5                         # <stdin>:517:5
	je	.LBB0_71
.LBB0_69:                               # %.lr.ph47.us
                                        #   Parent Loop BB0_66 Depth=1
                                        #     Parent Loop BB0_67 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_82 Depth 4
                                        #           Child Loop BB0_83 Depth 5
                                        #             Child Loop BB0_84 Depth 6
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	%r11, 32(%rsp)                  # 8-byte Spill
	shlq	$5, %r11
	cmpq	$1968, %r11                     # imm = 0x7B0
	movl	$1968, %r15d                    # imm = 0x7B0
	cmovbq	%r11, %r15
	addq	$32, %r15
	movq	56(%rsp), %rbx                  # 8-byte Reload
	movq	64(%rsp), %r12                  # 8-byte Reload
	cmpq	%r15, %r11
	.loc	1 526 5 is_stmt 1               # <stdin>:526:5
	jae	.LBB0_70
	.p2align	4, 0x90
.LBB0_82:                               # %.lr.ph47.split.us.us.us
                                        #   Parent Loop BB0_66 Depth=1
                                        #     Parent Loop BB0_67 Depth=2
                                        #       Parent Loop BB0_69 Depth=3
                                        # =>      This Loop Header: Depth=4
                                        #           Child Loop BB0_83 Depth 5
                                        #             Child Loop BB0_84 Depth 6
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	imulq	$2000, %r12, %rdx               # imm = 0x7D0
	movq	%r10, %rbp
	movq	%r11, %rdi
	.p2align	4, 0x90
.LBB0_83:                               # %.lr.ph.us.us.us.us
                                        #   Parent Loop BB0_66 Depth=1
                                        #     Parent Loop BB0_67 Depth=2
                                        #       Parent Loop BB0_69 Depth=3
                                        #         Parent Loop BB0_82 Depth=4
                                        # =>        This Loop Header: Depth=5
                                        #             Child Loop BB0_84 Depth 6
	leaq	(%rdi,%rdx), %rax
	movsd	(%r13,%rax,8), %xmm0            # xmm0 = mem[0],zero
	xorl	%eax, %eax
	.p2align	4, 0x90
.LBB0_84:                               #   Parent Loop BB0_66 Depth=1
                                        #     Parent Loop BB0_67 Depth=2
                                        #       Parent Loop BB0_69 Depth=3
                                        #         Parent Loop BB0_82 Depth=4
                                        #           Parent Loop BB0_83 Depth=5
                                        # =>          This Inner Loop Header: Depth=6
	movsd	(%rbp,%rax,8), %xmm1            # xmm1 = mem[0],zero
	.loc	1 564 12 is_stmt 1              # <stdin>:564:12
	mulsd	%xmm0, %xmm1
	.loc	1 565 12                        # <stdin>:565:12
	addsd	(%rbx,%rax,8), %xmm1
	.loc	1 571 5                         # <stdin>:571:5
	movsd	%xmm1, (%rbx,%rax,8)
	.loc	1 543 12                        # <stdin>:543:12
	leaq	(%rsi,%rax), %r9
	addq	$1, %r9
	addq	$1, %rax
	cmpq	%rcx, %r9
	.loc	1 544 5                         # <stdin>:544:5
	jb	.LBB0_84
# %bb.85:                               # %._crit_edge.us.us.us.us
                                        #   in Loop: Header=BB0_83 Depth=5
	.loc	1 575 12                        # <stdin>:575:12
	addq	$1, %rdi
	.loc	1 535 5                         # <stdin>:535:5
	addq	$14400, %rbp                    # imm = 0x3840
	.loc	1 534 12                        # <stdin>:534:12
	cmpq	%r15, %rdi
	.loc	1 535 5                         # <stdin>:535:5
	jb	.LBB0_83
# %bb.81:                               # %._crit_edge45.us.loopexit.us.us
                                        #   in Loop: Header=BB0_82 Depth=4
	.loc	1 578 12                        # <stdin>:578:12
	addq	$1, %r12
	.loc	1 526 5                         # <stdin>:526:5
	addq	$14400, %rbx                    # imm = 0x3840
	.loc	1 525 12                        # <stdin>:525:12
	cmpq	%r8, %r12
	jne	.LBB0_82
	jmp	.LBB0_70
	.p2align	4, 0x90
.LBB0_71:                               # %.lr.ph56.preheader
                                        #   in Loop: Header=BB0_67 Depth=2
	.loc	1 0 12 is_stmt 0                # <stdin>:0:12
	cmpq	%rcx, 40(%rsp)                  # 8-byte Folded Reload
	.loc	1 594 5 is_stmt 1               # <stdin>:594:5
	jae	.LBB0_75
# %bb.72:                               # %.lr.ph56.us.preheader
                                        #   in Loop: Header=BB0_67 Depth=2
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	72(%rsp), %rdx                  # 8-byte Reload
	movq	88(%rsp), %r11                  # 8-byte Reload
	xorl	%r9d, %r9d
	xorl	%edi, %edi
	jmp	.LBB0_73
	.p2align	4, 0x90
.LBB0_74:                               # %._crit_edge57.loopexit.us
                                        #   in Loop: Header=BB0_73 Depth=3
	movq	96(%rsp), %rdi                  # 8-byte Reload
	.loc	1 649 12 is_stmt 1              # <stdin>:649:12
	addq	$1, %rdi
	.loc	1 585 5                         # <stdin>:585:5
	addq	$32, %r9
	addq	$256, %r11                      # imm = 0x100
	movq	32(%rsp), %rdx                  # 8-byte Reload
	addq	$256, %rdx                      # imm = 0x100
	.loc	1 584 12                        # <stdin>:584:12
	cmpq	$69, %rdi
	.loc	1 585 5                         # <stdin>:585:5
	je	.LBB0_75
.LBB0_73:                               # %.lr.ph56.us
                                        #   Parent Loop BB0_66 Depth=1
                                        #     Parent Loop BB0_67 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_87 Depth 4
                                        #           Child Loop BB0_88 Depth 5
                                        #             Child Loop BB0_89 Depth 6
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	%rdi, 96(%rsp)                  # 8-byte Spill
	shlq	$5, %rdi
	cmpq	$2168, %rdi                     # imm = 0x878
	movl	$2168, %eax                     # imm = 0x878
	cmovbq	%rdi, %rax
	addq	$32, %rax
	movq	%rdx, 32(%rsp)                  # 8-byte Spill
	movq	64(%rsp), %rbx                  # 8-byte Reload
	cmpq	%rax, %rdi
	.loc	1 603 5 is_stmt 1               # <stdin>:603:5
	jae	.LBB0_74
	.p2align	4, 0x90
.LBB0_87:                               # %.lr.ph56.split.us.us.us
                                        #   Parent Loop BB0_66 Depth=1
                                        #     Parent Loop BB0_67 Depth=2
                                        #       Parent Loop BB0_73 Depth=3
                                        # =>      This Loop Header: Depth=4
                                        #           Child Loop BB0_88 Depth 5
                                        #             Child Loop BB0_89 Depth 6
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	imulq	$1800, %rbx, %r12               # imm = 0x708
	movq	%r11, %rbp
	movq	40(%rsp), %r15                  # 8-byte Reload
	.p2align	4, 0x90
.LBB0_88:                               # %.lr.ph50.us.us.us.us
                                        #   Parent Loop BB0_66 Depth=1
                                        #     Parent Loop BB0_67 Depth=2
                                        #       Parent Loop BB0_73 Depth=3
                                        #         Parent Loop BB0_87 Depth=4
                                        # =>        This Loop Header: Depth=5
                                        #             Child Loop BB0_89 Depth 6
	leaq	(%r15,%r12), %rdi
	movsd	(%r14,%rdi,8), %xmm0            # xmm0 = mem[0],zero
	xorl	%edi, %edi
	.p2align	4, 0x90
.LBB0_89:                               #   Parent Loop BB0_66 Depth=1
                                        #     Parent Loop BB0_67 Depth=2
                                        #       Parent Loop BB0_73 Depth=3
                                        #         Parent Loop BB0_87 Depth=4
                                        #           Parent Loop BB0_88 Depth=5
                                        # =>          This Inner Loop Header: Depth=6
	movsd	(%rbp,%rdi,8), %xmm1            # xmm1 = mem[0],zero
	.loc	1 632 12 is_stmt 1              # <stdin>:632:12
	mulsd	%xmm0, %xmm1
	.loc	1 633 12                        # <stdin>:633:12
	addsd	(%rdx,%rdi,8), %xmm1
	.loc	1 639 5                         # <stdin>:639:5
	movsd	%xmm1, (%rdx,%rdi,8)
	.loc	1 611 12                        # <stdin>:611:12
	leaq	(%r9,%rdi), %r10
	addq	$1, %r10
	addq	$1, %rdi
	cmpq	%rax, %r10
	.loc	1 612 5                         # <stdin>:612:5
	jb	.LBB0_89
# %bb.90:                               # %._crit_edge51.us.us.us.us
                                        #   in Loop: Header=BB0_88 Depth=5
	.loc	1 643 12                        # <stdin>:643:12
	addq	$1, %r15
	.loc	1 603 5                         # <stdin>:603:5
	addq	$17600, %rbp                    # imm = 0x44C0
	.loc	1 602 12                        # <stdin>:602:12
	cmpq	%rcx, %r15
	.loc	1 603 5                         # <stdin>:603:5
	jb	.LBB0_88
# %bb.86:                               # %._crit_edge54.us.loopexit.us.us
                                        #   in Loop: Header=BB0_87 Depth=4
	.loc	1 646 12                        # <stdin>:646:12
	addq	$1, %rbx
	.loc	1 594 5                         # <stdin>:594:5
	addq	$17600, %rdx                    # imm = 0x44C0
	.loc	1 593 12                        # <stdin>:593:12
	cmpq	%r8, %rbx
	jne	.LBB0_87
	jmp	.LBB0_74
.LBB0_77:
	.loc	1 658 5                         # <stdin>:658:5
	callq	polybench_timer_stop
	.loc	1 659 5                         # <stdin>:659:5
	callq	polybench_timer_print
	.loc	1 660 12                        # <stdin>:660:12
	cmpl	$43, 148(%rsp)                  # 4-byte Folded Reload
	.loc	1 661 5                         # <stdin>:661:5
	jl	.LBB0_80
# %bb.78:
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	176(%rsp), %rax                 # 8-byte Reload
	.loc	1 663 12 is_stmt 1              # <stdin>:663:12
	movq	(%rax), %rax
	.loc	1 672 5                         # <stdin>:672:5
	testb	$1, (%rax)
	jne	.LBB0_80
# %bb.79:
	.loc	1 681 5                         # <stdin>:681:5
	movl	$1600, %edi                     # imm = 0x640
	movl	$2200, %esi                     # imm = 0x898
	movq	104(%rsp), %rcx                 # 8-byte Reload
	callq	print_array
.LBB0_80:                               # %.critedge
	.loc	1 684 5                         # <stdin>:684:5
	xorl	%eax, %eax
	addq	$232, %rsp
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
	.loc	1 689 0                         # <stdin>:689:0
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
	.loc	1 702 11 prologue_end           # <stdin>:702:11
	movq	stderr(%rip), %rcx
	.loc	1 706 11                        # <stdin>:706:11
	movl	$str1, %edi
	movl	$22, %esi
	movl	$1, %edx
	callq	fwrite
	.loc	1 708 11                        # <stdin>:708:11
	movq	stderr(%rip), %rdi
	.loc	1 713 11                        # <stdin>:713:11
	movl	$str2, %esi
	movl	$str3, %edx
	xorl	%eax, %eax
	callq	fprintf
	.loc	1 716 11                        # <stdin>:716:11
	testl	%ebp, %ebp
	.loc	1 718 5                         # <stdin>:718:5
	jle	.LBB1_8
# %bb.1:
	testl	%r14d, %r14d
	jle	.LBB1_8
# %bb.2:                                # %.preheader.us.preheader
	.loc	1 716 11                        # <stdin>:716:11
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
	.loc	1 740 11 is_stmt 1              # <stdin>:740:11
	leal	(%rbx,%rbp), %eax
	.loc	1 741 11                        # <stdin>:741:11
	imull	$-858993459, %eax, %eax         # imm = 0xCCCCCCCD
	addl	$429496728, %eax                # imm = 0x19999998
	rorl	$2, %eax
	cmpl	$214748364, %eax                # imm = 0xCCCCCCC
	.loc	1 742 5                         # <stdin>:742:5
	ja	.LBB1_6
# %bb.5:                                #   in Loop: Header=BB1_4 Depth=2
	.loc	1 745 11                        # <stdin>:745:11
	movq	stderr(%rip), %rsi
	.loc	1 748 11                        # <stdin>:748:11
	movl	$10, %edi
	callq	fputc
.LBB1_6:                                #   in Loop: Header=BB1_4 Depth=2
	.loc	1 752 11                        # <stdin>:752:11
	movq	stderr(%rip), %rdi
	.loc	1 760 11                        # <stdin>:760:11
	movsd	(%r12,%rbx,8), %xmm0            # xmm0 = mem[0],zero
	.loc	1 761 11                        # <stdin>:761:11
	movl	$str5, %esi
	movb	$1, %al
	callq	fprintf
	.loc	1 762 11                        # <stdin>:762:11
	addq	$1, %rbx
	.loc	1 734 11                        # <stdin>:734:11
	cmpq	%rbx, %r14
	.loc	1 736 5                         # <stdin>:736:5
	jne	.LBB1_4
# %bb.7:                                # %._crit_edge.us
                                        #   in Loop: Header=BB1_3 Depth=1
	.loc	1 765 11                        # <stdin>:765:11
	addq	$1, %r13
	.loc	1 718 5                         # <stdin>:718:5
	addq	$17600, %r12                    # imm = 0x44C0
	addq	%r15, %rbp
	.loc	1 716 11                        # <stdin>:716:11
	cmpq	%r15, %r13
	.loc	1 718 5                         # <stdin>:718:5
	jne	.LBB1_3
.LBB1_8:                                # %._crit_edge1
	.loc	1 721 11                        # <stdin>:721:11
	movq	stderr(%rip), %rdi
	.loc	1 726 11                        # <stdin>:726:11
	movl	$str6, %esi
	movl	$str3, %edx
	xorl	%eax, %eax
	callq	fprintf
	.loc	1 728 11                        # <stdin>:728:11
	movq	stderr(%rip), %rcx
	.loc	1 731 11                        # <stdin>:731:11
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
	.loc	1 768 0                         # <stdin>:768:0
	.cfi_startproc
# %bb.0:
	.loc	1 780 11 prologue_end           # <stdin>:780:11
	imulq	$1800, 16(%rsp), %rax           # imm = 0x708
	.loc	1 781 11                        # <stdin>:781:11
	addq	24(%rsp), %rax
	.loc	1 783 5                         # <stdin>:783:5
	movq	$0, (%rsi,%rax,8)
	.loc	1 784 5                         # <stdin>:784:5
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
	.loc	1 786 0                         # <stdin>:786:0
	.cfi_startproc
# %bb.0:
	movq	40(%rsp), %r8
	movq	104(%rsp), %r9
	movq	88(%rsp), %rdx
	movq	24(%rsp), %rdi
	movq	16(%rsp), %rax
.Ltmp5:
	.loc	1 813 11 prologue_end           # <stdin>:813:11
	imulq	$1800, %rax, %rcx               # imm = 0x708
	.loc	1 819 11                        # <stdin>:819:11
	imulq	$2000, %rax, %rax               # imm = 0x7D0
	.loc	1 820 11                        # <stdin>:820:11
	addq	%rdx, %rax
	.loc	1 822 11                        # <stdin>:822:11
	movsd	(%r9,%rax,8), %xmm0             # xmm0 = mem[0],zero
	.loc	1 825 11                        # <stdin>:825:11
	imulq	$1800, %rdx, %rax               # imm = 0x708
	.loc	1 826 11                        # <stdin>:826:11
	addq	%rdi, %rax
	.loc	1 829 11                        # <stdin>:829:11
	mulsd	(%r8,%rax,8), %xmm0
	.loc	1 814 11                        # <stdin>:814:11
	addq	%rdi, %rcx
	.loc	1 830 11                        # <stdin>:830:11
	addsd	(%rsi,%rcx,8), %xmm0
	.loc	1 836 5                         # <stdin>:836:5
	movsd	%xmm0, (%rsi,%rcx,8)
	.loc	1 837 5                         # <stdin>:837:5
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
	.loc	1 839 0                         # <stdin>:839:0
	.cfi_startproc
# %bb.0:
	.loc	1 851 11 prologue_end           # <stdin>:851:11
	imulq	$2200, 16(%rsp), %rax           # imm = 0x898
	.loc	1 852 11                        # <stdin>:852:11
	addq	24(%rsp), %rax
	.loc	1 854 5                         # <stdin>:854:5
	movq	$0, (%rsi,%rax,8)
	.loc	1 855 5                         # <stdin>:855:5
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
	.loc	1 857 0                         # <stdin>:857:0
	.cfi_startproc
# %bb.0:
	movq	40(%rsp), %r8
	movq	104(%rsp), %r9
	movq	88(%rsp), %rdx
	movq	24(%rsp), %rdi
	movq	16(%rsp), %rax
.Ltmp8:
	.loc	1 884 11 prologue_end           # <stdin>:884:11
	imulq	$2200, %rax, %rcx               # imm = 0x898
	.loc	1 890 11                        # <stdin>:890:11
	imulq	$2400, %rax, %rax               # imm = 0x960
	.loc	1 891 11                        # <stdin>:891:11
	addq	%rdx, %rax
	.loc	1 893 11                        # <stdin>:893:11
	movsd	(%r9,%rax,8), %xmm0             # xmm0 = mem[0],zero
	.loc	1 896 11                        # <stdin>:896:11
	imulq	$2200, %rdx, %rax               # imm = 0x898
	.loc	1 897 11                        # <stdin>:897:11
	addq	%rdi, %rax
	.loc	1 900 11                        # <stdin>:900:11
	mulsd	(%r8,%rax,8), %xmm0
	.loc	1 885 11                        # <stdin>:885:11
	addq	%rdi, %rcx
	.loc	1 901 11                        # <stdin>:901:11
	addsd	(%rsi,%rcx,8), %xmm0
	.loc	1 907 5                         # <stdin>:907:5
	movsd	%xmm0, (%rsi,%rcx,8)
	.loc	1 908 5                         # <stdin>:908:5
	retq
.Ltmp9:
.Lfunc_end5:
	.size	S3, .Lfunc_end5-S3
	.cfi_endproc
                                        # -- End function
	.globl	S4                              # -- Begin function S4
	.p2align	4, 0x90
	.type	S4,@function
S4:                                     # @S4
.Lfunc_begin6:
	.loc	1 910 0                         # <stdin>:910:0
	.cfi_startproc
# %bb.0:
	.loc	1 922 11 prologue_end           # <stdin>:922:11
	imulq	$2200, 16(%rsp), %rax           # imm = 0x898
	.loc	1 923 11                        # <stdin>:923:11
	addq	24(%rsp), %rax
	.loc	1 925 5                         # <stdin>:925:5
	movq	$0, (%rsi,%rax,8)
	.loc	1 926 5                         # <stdin>:926:5
	retq
.Ltmp10:
.Lfunc_end6:
	.size	S4, .Lfunc_end6-S4
	.cfi_endproc
                                        # -- End function
	.globl	S5                              # -- Begin function S5
	.p2align	4, 0x90
	.type	S5,@function
S5:                                     # @S5
.Lfunc_begin7:
	.loc	1 928 0                         # <stdin>:928:0
	.cfi_startproc
# %bb.0:
	movq	40(%rsp), %r8
	movq	104(%rsp), %r9
	movq	88(%rsp), %rdx
	movq	24(%rsp), %rdi
	movq	16(%rsp), %rax
.Ltmp11:
	.loc	1 955 11 prologue_end           # <stdin>:955:11
	imulq	$2200, %rax, %rcx               # imm = 0x898
	.loc	1 961 11                        # <stdin>:961:11
	imulq	$1800, %rax, %rax               # imm = 0x708
	.loc	1 962 11                        # <stdin>:962:11
	addq	%rdx, %rax
	.loc	1 964 11                        # <stdin>:964:11
	movsd	(%r9,%rax,8), %xmm0             # xmm0 = mem[0],zero
	.loc	1 967 11                        # <stdin>:967:11
	imulq	$2200, %rdx, %rax               # imm = 0x898
	.loc	1 968 11                        # <stdin>:968:11
	addq	%rdi, %rax
	.loc	1 971 11                        # <stdin>:971:11
	mulsd	(%r8,%rax,8), %xmm0
	.loc	1 956 11                        # <stdin>:956:11
	addq	%rdi, %rcx
	.loc	1 972 11                        # <stdin>:972:11
	addsd	(%rsi,%rcx,8), %xmm0
	.loc	1 978 5                         # <stdin>:978:5
	movsd	%xmm0, (%rsi,%rcx,8)
	.loc	1 979 5                         # <stdin>:979:5
	retq
.Ltmp12:
.Lfunc_end7:
	.size	S5, .Lfunc_end7-S5
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
	.asciz	"G"
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
	.byte	1                               # Abbrev [1] 0xb:0xef DW_TAG_compile_unit
	.long	.Linfo_string0                  # DW_AT_producer
	.short	2                               # DW_AT_language
	.long	.Linfo_string1                  # DW_AT_name
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Linfo_string2                  # DW_AT_comp_dir
                                        # DW_AT_GNU_pubnames
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end7-.Lfunc_begin0       # DW_AT_high_pc
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
	.short	689                             # DW_AT_decl_line
                                        # DW_AT_external
	.byte	3                               # Abbrev [3] 0x5d:0x1a DW_TAG_subprogram
	.quad	.Lfunc_begin2                   # DW_AT_low_pc
	.long	.Lfunc_end2-.Lfunc_begin2       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
	.long	.Linfo_string5                  # DW_AT_linkage_name
	.long	.Linfo_string5                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.short	768                             # DW_AT_decl_line
                                        # DW_AT_external
	.byte	3                               # Abbrev [3] 0x77:0x1a DW_TAG_subprogram
	.quad	.Lfunc_begin3                   # DW_AT_low_pc
	.long	.Lfunc_end3-.Lfunc_begin3       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
	.long	.Linfo_string6                  # DW_AT_linkage_name
	.long	.Linfo_string6                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.short	786                             # DW_AT_decl_line
                                        # DW_AT_external
	.byte	3                               # Abbrev [3] 0x91:0x1a DW_TAG_subprogram
	.quad	.Lfunc_begin4                   # DW_AT_low_pc
	.long	.Lfunc_end4-.Lfunc_begin4       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
	.long	.Linfo_string7                  # DW_AT_linkage_name
	.long	.Linfo_string7                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.short	839                             # DW_AT_decl_line
                                        # DW_AT_external
	.byte	3                               # Abbrev [3] 0xab:0x1a DW_TAG_subprogram
	.quad	.Lfunc_begin5                   # DW_AT_low_pc
	.long	.Lfunc_end5-.Lfunc_begin5       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
	.long	.Linfo_string8                  # DW_AT_linkage_name
	.long	.Linfo_string8                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.short	857                             # DW_AT_decl_line
                                        # DW_AT_external
	.byte	3                               # Abbrev [3] 0xc5:0x1a DW_TAG_subprogram
	.quad	.Lfunc_begin6                   # DW_AT_low_pc
	.long	.Lfunc_end6-.Lfunc_begin6       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
	.long	.Linfo_string9                  # DW_AT_linkage_name
	.long	.Linfo_string9                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.short	910                             # DW_AT_decl_line
                                        # DW_AT_external
	.byte	3                               # Abbrev [3] 0xdf:0x1a DW_TAG_subprogram
	.quad	.Lfunc_begin7                   # DW_AT_low_pc
	.long	.Lfunc_end7-.Lfunc_begin7       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
	.long	.Linfo_string10                 # DW_AT_linkage_name
	.long	.Linfo_string10                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.short	928                             # DW_AT_decl_line
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
.Linfo_string9:
	.asciz	"S4"                            # string offset=54
.Linfo_string10:
	.asciz	"S5"                            # string offset=57
	.section	.debug_pubnames,"",@progbits
	.long	.LpubNames_end0-.LpubNames_begin0 # Length of Public Names Info
.LpubNames_begin0:
	.short	2                               # DWARF Version
	.long	.Lcu_begin0                     # Offset of Compilation Unit Info
	.long	250                             # Compilation Unit Length
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
	.long	197                             # DIE offset
	.asciz	"S4"                            # External Name
	.long	223                             # DIE offset
	.asciz	"S5"                            # External Name
	.long	67                              # DIE offset
	.asciz	"print_array"                   # External Name
	.long	0                               # End Mark
.LpubNames_end0:
	.section	.debug_pubtypes,"",@progbits
	.long	.LpubTypes_end0-.LpubTypes_begin0 # Length of Public Types Info
.LpubTypes_begin0:
	.short	2                               # DWARF Version
	.long	.Lcu_begin0                     # Offset of Compilation Unit Info
	.long	250                             # Compilation Unit Length
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
