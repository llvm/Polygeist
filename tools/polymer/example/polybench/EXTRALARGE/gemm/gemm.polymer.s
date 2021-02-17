	.text
	.file	"LLVMDialectModule"
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3                               # -- Begin function main
.LCPI0_0:
	.quad	0x409f400000000000              # double 2000
.LCPI0_1:
	.quad	0x40a4500000000000              # double 2600
.LCPI0_2:
	.quad	0x40a1f80000000000              # double 2300
.LCPI0_3:
	.quad	0x3ff3333333333333              # double 1.2
.LCPI0_4:
	.quad	0x3ff8000000000000              # double 1.5
	.text
	.globl	main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
.Lfunc_begin0:
	.file	1 "/home/ubuntu/polymer/example/polybench/EXTRALARGE/gemm" "<stdin>"
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
	subq	$136, %rsp
	.cfi_def_cfa_offset 192
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rsi, 72(%rsp)                  # 8-byte Spill
	movl	%edi, 12(%rsp)                  # 4-byte Spill
.Ltmp0:
	.loc	1 68 11 prologue_end            # <stdin>:68:11
	movl	$36800000, %edi                 # imm = 0x2318600
	callq	malloc@PLT
	movq	%rax, %r13
	.loc	1 86 11                         # <stdin>:86:11
	movl	$41600000, %edi                 # imm = 0x27AC400
	callq	malloc@PLT
	movq	%rax, %rbp
	.loc	1 104 11                        # <stdin>:104:11
	movl	$47840000, %edi                 # imm = 0x2D9FB00
	callq	malloc@PLT
	movq	%rax, 24(%rsp)                  # 8-byte Spill
	xorl	%eax, %eax
	movsd	.LCPI0_0(%rip), %xmm0           # xmm0 = mem[0],zero
	movq	%r13, %rcx
	.p2align	4, 0x90
.LBB0_1:                                # %.preheader19
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_2 Depth 2
	.loc	1 0 11 is_stmt 0                # <stdin>:0:11
	movl	$1, %ebx
	xorl	%esi, %esi
	.p2align	4, 0x90
.LBB0_2:                                #   Parent Loop BB0_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	.loc	1 131 12 is_stmt 1              # <stdin>:131:12
	movl	%ebx, %edi
	imulq	$274877907, %rdi, %rdi          # imm = 0x10624DD3
	shrq	$39, %rdi
	imull	$2000, %edi, %edi               # imm = 0x7D0
	movl	%ebx, %edx
	subl	%edi, %edx
	.loc	1 134 12                        # <stdin>:134:12
	xorps	%xmm1, %xmm1
	cvtsi2sd	%edx, %xmm1
	.loc	1 136 12                        # <stdin>:136:12
	divsd	%xmm0, %xmm1
	.loc	1 142 5                         # <stdin>:142:5
	movsd	%xmm1, (%rcx,%rsi,8)
	.loc	1 143 12                        # <stdin>:143:12
	addq	$1, %rsi
	.loc	1 127 12                        # <stdin>:127:12
	addl	%eax, %ebx
	cmpq	$2300, %rsi                     # imm = 0x8FC
	.loc	1 129 5                         # <stdin>:129:5
	jne	.LBB0_2
# %bb.3:                                #   in Loop: Header=BB0_1 Depth=1
	.loc	1 146 12                        # <stdin>:146:12
	addq	$1, %rax
	.loc	1 125 5                         # <stdin>:125:5
	addq	$18400, %rcx                    # imm = 0x47E0
	.loc	1 123 12                        # <stdin>:123:12
	cmpq	$2000, %rax                     # imm = 0x7D0
	.loc	1 125 5                         # <stdin>:125:5
	jne	.LBB0_1
# %bb.4:                                # %.preheader17.preheader
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	xorl	%eax, %eax
	movl	$3383112701, %r8d               # imm = 0xC9A633FD
	movsd	.LCPI0_1(%rip), %xmm0           # xmm0 = mem[0],zero
	movq	%rbp, %rdx
	.p2align	4, 0x90
.LBB0_5:                                # %.preheader17
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_6 Depth 2
	movl	%eax, %ecx
	xorl	%edi, %edi
	.p2align	4, 0x90
.LBB0_6:                                #   Parent Loop BB0_5 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	.loc	1 158 12 is_stmt 1              # <stdin>:158:12
	movl	%ecx, %ebx
	imulq	%r8, %rbx
	shrq	$43, %rbx
	imull	$2600, %ebx, %ebx               # imm = 0xA28
	movl	%ecx, %esi
	subl	%ebx, %esi
	.loc	1 160 12                        # <stdin>:160:12
	xorps	%xmm1, %xmm1
	cvtsi2sd	%esi, %xmm1
	.loc	1 162 12                        # <stdin>:162:12
	divsd	%xmm0, %xmm1
	.loc	1 168 5                         # <stdin>:168:5
	movsd	%xmm1, (%rdx,%rdi,8)
	.loc	1 157 12                        # <stdin>:157:12
	addq	$1, %rdi
	.loc	1 153 12                        # <stdin>:153:12
	addl	%eax, %ecx
	cmpq	$2600, %rdi                     # imm = 0xA28
	.loc	1 155 5                         # <stdin>:155:5
	jne	.LBB0_6
# %bb.7:                                #   in Loop: Header=BB0_5 Depth=1
	.loc	1 171 12                        # <stdin>:171:12
	addq	$1, %rax
	.loc	1 151 5                         # <stdin>:151:5
	addq	$20800, %rdx                    # imm = 0x5140
	.loc	1 149 12                        # <stdin>:149:12
	cmpq	$2000, %rax                     # imm = 0x7D0
	.loc	1 151 5                         # <stdin>:151:5
	jne	.LBB0_5
# %bb.8:                                # %.preheader15.preheader
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	xorl	%r8d, %r8d
	movl	$3824388271, %r9d               # imm = 0xE3F388AF
	movsd	.LCPI0_2(%rip), %xmm0           # xmm0 = mem[0],zero
	movq	24(%rsp), %rdx                  # 8-byte Reload
	xorl	%esi, %esi
	.p2align	4, 0x90
.LBB0_9:                                # %.preheader15
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_10 Depth 2
	movl	%r8d, %ecx
	xorl	%ebx, %ebx
	.p2align	4, 0x90
.LBB0_10:                               #   Parent Loop BB0_9 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	.loc	1 191 12 is_stmt 1              # <stdin>:191:12
	movl	%ecx, %eax
	imulq	%r9, %rax
	shrq	$43, %rax
	imull	$2300, %eax, %eax               # imm = 0x8FC
	movl	%ecx, %edi
	subl	%eax, %edi
	.loc	1 194 12                        # <stdin>:194:12
	xorps	%xmm1, %xmm1
	cvtsi2sd	%edi, %xmm1
	.loc	1 196 12                        # <stdin>:196:12
	divsd	%xmm0, %xmm1
	.loc	1 202 5                         # <stdin>:202:5
	movsd	%xmm1, (%rdx,%rbx,8)
	.loc	1 203 12                        # <stdin>:203:12
	addq	$1, %rbx
	.loc	1 187 12                        # <stdin>:187:12
	addl	%esi, %ecx
	cmpq	$2300, %rbx                     # imm = 0x8FC
	.loc	1 189 5                         # <stdin>:189:5
	jne	.LBB0_10
# %bb.11:                               #   in Loop: Header=BB0_9 Depth=1
	.loc	1 206 12                        # <stdin>:206:12
	addq	$1, %rsi
	.loc	1 176 5                         # <stdin>:176:5
	addq	$18400, %rdx                    # imm = 0x47E0
	addl	$2, %r8d
	.loc	1 174 12                        # <stdin>:174:12
	cmpq	$2600, %rsi                     # imm = 0xA28
	.loc	1 176 5                         # <stdin>:176:5
	jne	.LBB0_9
# %bb.12:
	.loc	1 178 5                         # <stdin>:178:5
	callq	polybench_timer_start@PLT
	movl	$32, %r9d
	xorl	%r10d, %r10d
	movsd	.LCPI0_3(%rip), %xmm0           # xmm0 = mem[0],zero
	movq	%r13, %r8
	jmp	.LBB0_13
	.p2align	4, 0x90
.LBB0_21:                               # %.us-lcssa.us
                                        #   in Loop: Header=BB0_13 Depth=1
	.loc	1 255 12                        # <stdin>:255:12
	addq	$1, %r10
	.loc	1 210 5                         # <stdin>:210:5
	addq	$32, %r9
	addq	$588800, %r8                    # imm = 0x8FC00
	.loc	1 209 12                        # <stdin>:209:12
	cmpq	$63, %r10
	.loc	1 210 5                         # <stdin>:210:5
	je	.LBB0_22
.LBB0_13:                               # %.preheader14
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_15 Depth 2
                                        #       Child Loop BB0_17 Depth 3
                                        #         Child Loop BB0_18 Depth 4
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	cmpq	$2000, %r9                      # imm = 0x7D0
	movl	$2000, %esi                     # imm = 0x7D0
	cmovbq	%r9, %rsi
	movq	%r10, %r11
	shlq	$5, %r11
	cmpq	$1968, %r11                     # imm = 0x7B0
	movl	$1968, %eax                     # imm = 0x7B0
	cmovbq	%r11, %rax
	addq	$32, %rax
	cmpq	%rax, %r11
	.loc	1 213 5 is_stmt 1               # <stdin>:213:5
	jae	.LBB0_21
# %bb.14:                               # %.preheader14.split.us.preheader
                                        #   in Loop: Header=BB0_13 Depth=1
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movl	$32, %r12d
	movq	%r8, %r14
	xorl	%r15d, %r15d
	xorl	%ecx, %ecx
	jmp	.LBB0_15
	.p2align	4, 0x90
.LBB0_20:                               # %._crit_edge31.us
                                        #   in Loop: Header=BB0_15 Depth=2
	.loc	1 252 12 is_stmt 1              # <stdin>:252:12
	addq	$1, %rcx
	.loc	1 213 5                         # <stdin>:213:5
	addq	$32, %r12
	addq	$-32, %r15
	addq	$256, %r14                      # imm = 0x100
	.loc	1 212 12                        # <stdin>:212:12
	cmpq	$72, %rcx
	.loc	1 213 5                         # <stdin>:213:5
	je	.LBB0_21
.LBB0_15:                               # %.preheader14.split.us
                                        #   Parent Loop BB0_13 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_17 Depth 3
                                        #         Child Loop BB0_18 Depth 4
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	cmpq	$2300, %r12                     # imm = 0x8FC
	movl	$2300, %edi                     # imm = 0x8FC
	cmovbq	%r12, %rdi
	movq	%rcx, %rax
	shlq	$5, %rax
	cmpq	$2268, %rax                     # imm = 0x8DC
	movl	$2268, %edx                     # imm = 0x8DC
	cmovbq	%rax, %rdx
	addq	$32, %rdx
	cmpq	%rdx, %rax
	.loc	1 222 5 is_stmt 1               # <stdin>:222:5
	jae	.LBB0_20
# %bb.16:                               #   in Loop: Header=BB0_15 Depth=2
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	addq	%r15, %rdi
	movq	%r14, %rdx
	movq	%r11, %rax
	.p2align	4, 0x90
.LBB0_17:                               # %.lr.ph27.us.us
                                        #   Parent Loop BB0_13 Depth=1
                                        #     Parent Loop BB0_15 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_18 Depth 4
	xorl	%ebx, %ebx
	.p2align	4, 0x90
.LBB0_18:                               #   Parent Loop BB0_13 Depth=1
                                        #     Parent Loop BB0_15 Depth=2
                                        #       Parent Loop BB0_17 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	movsd	(%rdx,%rbx,8), %xmm1            # xmm1 = mem[0],zero
	.loc	1 239 12 is_stmt 1              # <stdin>:239:12
	mulsd	%xmm0, %xmm1
	.loc	1 245 5                         # <stdin>:245:5
	movsd	%xmm1, (%rdx,%rbx,8)
	.loc	1 230 12                        # <stdin>:230:12
	addq	$1, %rbx
	cmpq	%rbx, %rdi
	.loc	1 231 5                         # <stdin>:231:5
	jne	.LBB0_18
# %bb.19:                               # %._crit_edge28.us.us
                                        #   in Loop: Header=BB0_17 Depth=3
	.loc	1 249 12                        # <stdin>:249:12
	addq	$1, %rax
	.loc	1 222 5                         # <stdin>:222:5
	addq	$18400, %rdx                    # imm = 0x47E0
	.loc	1 221 12                        # <stdin>:221:12
	cmpq	%rsi, %rax
	.loc	1 222 5                         # <stdin>:222:5
	jne	.LBB0_17
	jmp	.LBB0_20
.LBB0_22:                               # %.preheader12.preheader
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movl	$32, %eax
	xorl	%ecx, %ecx
	movsd	.LCPI0_4(%rip), %xmm0           # xmm0 = mem[0],zero
	movq	%r13, 16(%rsp)                  # 8-byte Spill
	jmp	.LBB0_23
	.p2align	4, 0x90
.LBB0_30:                               #   in Loop: Header=BB0_23 Depth=1
	movq	88(%rsp), %rcx                  # 8-byte Reload
	.loc	1 336 12 is_stmt 1              # <stdin>:336:12
	addq	$1, %rcx
	movq	96(%rsp), %rax                  # 8-byte Reload
	.loc	1 259 5                         # <stdin>:259:5
	addq	$32, %rax
	movq	80(%rsp), %r13                  # 8-byte Reload
	addq	$588800, %r13                   # imm = 0x8FC00
	.loc	1 258 12                        # <stdin>:258:12
	cmpq	$63, %rcx
	.loc	1 259 5                         # <stdin>:259:5
	je	.LBB0_31
.LBB0_23:                               # %.preheader12
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_24 Depth 2
                                        #       Child Loop BB0_26 Depth 3
                                        #         Child Loop BB0_42 Depth 4
                                        #           Child Loop BB0_43 Depth 5
                                        #             Child Loop BB0_44 Depth 6
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	cmpq	$2000, %rax                     # imm = 0x7D0
	movl	$2000, %r14d                    # imm = 0x7D0
	movq	%rax, 96(%rsp)                  # 8-byte Spill
	cmovbq	%rax, %r14
	movq	%rcx, 88(%rsp)                  # 8-byte Spill
	shlq	$5, %rcx
	cmpq	$1968, %rcx                     # imm = 0x7B0
	movl	$1968, %eax                     # imm = 0x7B0
	movq	%rcx, 56(%rsp)                  # 8-byte Spill
	cmovbq	%rcx, %rax
	addq	$32, %rax
	movq	%rax, 104(%rsp)                 # 8-byte Spill
	movq	%r13, 80(%rsp)                  # 8-byte Spill
	movq	%r13, 48(%rsp)                  # 8-byte Spill
	movq	24(%rsp), %rax                  # 8-byte Reload
	movq	%rax, 40(%rsp)                  # 8-byte Spill
	xorl	%eax, %eax
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	movl	$32, %eax
	xorl	%ecx, %ecx
	jmp	.LBB0_24
	.p2align	4, 0x90
.LBB0_29:                               # %.us-lcssa.us20
                                        #   in Loop: Header=BB0_24 Depth=2
	movq	112(%rsp), %rcx                 # 8-byte Reload
	.loc	1 333 12 is_stmt 1              # <stdin>:333:12
	addq	$1, %rcx
	movq	120(%rsp), %rax                 # 8-byte Reload
	.loc	1 262 5                         # <stdin>:262:5
	addq	$32, %rax
	addq	$-32, 32(%rsp)                  # 8-byte Folded Spill
	addq	$256, 40(%rsp)                  # 8-byte Folded Spill
                                        # imm = 0x100
	addq	$256, 48(%rsp)                  # 8-byte Folded Spill
                                        # imm = 0x100
	.loc	1 261 12                        # <stdin>:261:12
	cmpq	$72, %rcx
	.loc	1 262 5                         # <stdin>:262:5
	je	.LBB0_30
.LBB0_24:                               # %.preheader
                                        #   Parent Loop BB0_23 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_26 Depth 3
                                        #         Child Loop BB0_42 Depth 4
                                        #           Child Loop BB0_43 Depth 5
                                        #             Child Loop BB0_44 Depth 6
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	cmpq	$2300, %rax                     # imm = 0x8FC
	movl	$2300, %r15d                    # imm = 0x8FC
	movq	%rax, 120(%rsp)                 # 8-byte Spill
	cmovbq	%rax, %r15
	movq	%rcx, 112(%rsp)                 # 8-byte Spill
	shlq	$5, %rcx
	cmpq	$2268, %rcx                     # imm = 0x8DC
	movl	$2268, %eax                     # imm = 0x8DC
	movq	%rcx, 128(%rsp)                 # 8-byte Spill
	cmovbq	%rcx, %rax
	movq	%rax, 64(%rsp)                  # 8-byte Spill
	movq	104(%rsp), %rax                 # 8-byte Reload
	cmpq	%rax, 56(%rsp)                  # 8-byte Folded Reload
	.loc	1 265 5 is_stmt 1               # <stdin>:265:5
	jae	.LBB0_29
# %bb.25:                               # %.preheader.split.us.preheader
                                        #   in Loop: Header=BB0_24 Depth=2
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	addq	32(%rsp), %r15                  # 8-byte Folded Reload
	addq	$32, 64(%rsp)                   # 8-byte Folded Spill
	movl	$32, %r8d
	movq	40(%rsp), %rbx                  # 8-byte Reload
	xorl	%r9d, %r9d
	jmp	.LBB0_26
	.p2align	4, 0x90
.LBB0_28:                               # %._crit_edge25.us
                                        #   in Loop: Header=BB0_26 Depth=3
	.loc	1 330 12 is_stmt 1              # <stdin>:330:12
	addq	$1, %r9
	.loc	1 265 5                         # <stdin>:265:5
	addq	$32, %r8
	addq	$588800, %rbx                   # imm = 0x8FC00
	.loc	1 264 12                        # <stdin>:264:12
	cmpq	$82, %r9
	.loc	1 265 5                         # <stdin>:265:5
	je	.LBB0_29
.LBB0_26:                               # %.preheader.split.us
                                        #   Parent Loop BB0_23 Depth=1
                                        #     Parent Loop BB0_24 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_42 Depth 4
                                        #           Child Loop BB0_43 Depth 5
                                        #             Child Loop BB0_44 Depth 6
	.loc	1 274 5                         # <stdin>:274:5
	cmpq	$2600, %r8                      # imm = 0xA28
	movl	$2600, %r11d                    # imm = 0xA28
	cmovbq	%r8, %r11
	movq	%r9, %rcx
	shlq	$5, %rcx
	cmpq	$2568, %rcx                     # imm = 0xA08
	movl	$2568, %eax                     # imm = 0xA08
	cmovbq	%rcx, %rax
	addq	$32, %rax
	cmpq	%rax, %rcx
	jae	.LBB0_28
# %bb.27:                               # %.preheader.split.us
                                        #   in Loop: Header=BB0_26 Depth=3
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	48(%rsp), %rdx                  # 8-byte Reload
	movq	56(%rsp), %rdi                  # 8-byte Reload
	movq	64(%rsp), %rax                  # 8-byte Reload
	.loc	1 274 5                         # <stdin>:274:5
	cmpq	%rax, 128(%rsp)                 # 8-byte Folded Reload
	jae	.LBB0_28
	.p2align	4, 0x90
.LBB0_42:                               # %.lr.ph24.split.us.us.us
                                        #   Parent Loop BB0_23 Depth=1
                                        #     Parent Loop BB0_24 Depth=2
                                        #       Parent Loop BB0_26 Depth=3
                                        # =>      This Loop Header: Depth=4
                                        #           Child Loop BB0_43 Depth 5
                                        #             Child Loop BB0_44 Depth 6
	.loc	1 0 5                           # <stdin>:0:5
	imulq	$2600, %rdi, %r12               # imm = 0xA28
	movq	%rbx, %rax
	movq	%rcx, %r10
	.p2align	4, 0x90
.LBB0_43:                               # %.lr.ph.us.us.us.us
                                        #   Parent Loop BB0_23 Depth=1
                                        #     Parent Loop BB0_24 Depth=2
                                        #       Parent Loop BB0_26 Depth=3
                                        #         Parent Loop BB0_42 Depth=4
                                        # =>        This Loop Header: Depth=5
                                        #             Child Loop BB0_44 Depth 6
	leaq	(%r10,%r12), %rsi
	movsd	(%rbp,%rsi,8), %xmm1            # xmm1 = mem[0],zero
	mulsd	%xmm0, %xmm1
	xorl	%r13d, %r13d
	.p2align	4, 0x90
.LBB0_44:                               #   Parent Loop BB0_23 Depth=1
                                        #     Parent Loop BB0_24 Depth=2
                                        #       Parent Loop BB0_26 Depth=3
                                        #         Parent Loop BB0_42 Depth=4
                                        #           Parent Loop BB0_43 Depth=5
                                        # =>          This Inner Loop Header: Depth=6
	movsd	(%rax,%r13,8), %xmm2            # xmm2 = mem[0],zero
	.loc	1 313 12 is_stmt 1              # <stdin>:313:12
	mulsd	%xmm1, %xmm2
	.loc	1 314 12                        # <stdin>:314:12
	addsd	(%rdx,%r13,8), %xmm2
	.loc	1 320 5                         # <stdin>:320:5
	movsd	%xmm2, (%rdx,%r13,8)
	.loc	1 291 12                        # <stdin>:291:12
	addq	$1, %r13
	cmpq	%r13, %r15
	.loc	1 292 5                         # <stdin>:292:5
	jne	.LBB0_44
# %bb.45:                               # %._crit_edge.us.us.us.us
                                        #   in Loop: Header=BB0_43 Depth=5
	.loc	1 324 12                        # <stdin>:324:12
	addq	$1, %r10
	.loc	1 283 5                         # <stdin>:283:5
	addq	$18400, %rax                    # imm = 0x47E0
	.loc	1 282 12                        # <stdin>:282:12
	cmpq	%r11, %r10
	.loc	1 283 5                         # <stdin>:283:5
	jne	.LBB0_43
# %bb.41:                               # %._crit_edge22.us.loopexit.us.us
                                        #   in Loop: Header=BB0_42 Depth=4
	.loc	1 327 12                        # <stdin>:327:12
	addq	$1, %rdi
	.loc	1 274 5                         # <stdin>:274:5
	addq	$18400, %rdx                    # imm = 0x47E0
	.loc	1 273 12                        # <stdin>:273:12
	cmpq	%r14, %rdi
	jne	.LBB0_42
	jmp	.LBB0_28
.LBB0_31:
	.loc	1 339 5                         # <stdin>:339:5
	callq	polybench_timer_stop@PLT
	.loc	1 340 5                         # <stdin>:340:5
	callq	polybench_timer_print@PLT
	.loc	1 341 12                        # <stdin>:341:12
	cmpl	$43, 12(%rsp)                   # 4-byte Folded Reload
	.loc	1 342 5                         # <stdin>:342:5
	jl	.LBB0_40
# %bb.32:
	.loc	1 0 5 is_stmt 0                 # <stdin>:0:5
	movq	72(%rsp), %rax                  # 8-byte Reload
	.loc	1 344 12 is_stmt 1              # <stdin>:344:12
	movq	(%rax), %rax
	.loc	1 353 5                         # <stdin>:353:5
	testb	$1, (%rax)
	je	.LBB0_33
.LBB0_40:                               # %.critedge
	.loc	1 365 5                         # <stdin>:365:5
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
.LBB0_33:
	.cfi_def_cfa_offset 192
.Ltmp1:
	.loc	1 383 11                        # <stdin>:383:11
	movq	stderr@GOTPCREL(%rip), %r15
	movq	(%r15), %rcx
	.loc	1 387 11                        # <stdin>:387:11
	movl	$str1, %edi
	movl	$22, %esi
	movl	$1, %edx
	callq	fwrite@PLT
	.loc	1 389 11                        # <stdin>:389:11
	movq	(%r15), %rdi
	xorl	%r12d, %r12d
	.loc	1 394 11                        # <stdin>:394:11
	movl	$str2, %esi
	movl	$str3, %edx
	xorl	%eax, %eax
	callq	fprintf@PLT
	xorl	%ebx, %ebx
	xorl	%r13d, %r13d
.LBB0_34:                               # %.preheader.i
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_35 Depth 2
	.loc	1 0 11 is_stmt 0                # <stdin>:0:11
	movl	%r12d, %ebp
	xorl	%r14d, %r14d
.LBB0_35:                               #   Parent Loop BB0_34 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	.loc	1 420 11 is_stmt 1              # <stdin>:420:11
	movl	%ebp, %eax
	movl	$3435973837, %ecx               # imm = 0xCCCCCCCD
	imulq	%rcx, %rax
	shrq	$36, %rax
	leal	(%rax,%rax,4), %eax
	leal	(%rbx,%rax,4), %eax
	.loc	1 422 11                        # <stdin>:422:11
	cmpl	%r14d, %eax
	.loc	1 423 5                         # <stdin>:423:5
	jne	.LBB0_37
# %bb.36:                               #   in Loop: Header=BB0_35 Depth=2
	.loc	1 426 11                        # <stdin>:426:11
	movq	(%r15), %rsi
	.loc	1 429 11                        # <stdin>:429:11
	movl	$10, %edi
	callq	fputc@PLT
.LBB0_37:                               #   in Loop: Header=BB0_35 Depth=2
	.loc	1 433 11                        # <stdin>:433:11
	movq	(%r15), %rdi
	movq	16(%rsp), %rax                  # 8-byte Reload
	.loc	1 441 11                        # <stdin>:441:11
	movsd	(%rax,%r14,8), %xmm0            # xmm0 = mem[0],zero
	.loc	1 442 11                        # <stdin>:442:11
	movl	$str5, %esi
	movb	$1, %al
	callq	fprintf@PLT
	.loc	1 443 11                        # <stdin>:443:11
	addq	$1, %r14
	.loc	1 415 11                        # <stdin>:415:11
	addl	$1, %ebp
	cmpq	$2300, %r14                     # imm = 0x8FC
	.loc	1 417 5                         # <stdin>:417:5
	jne	.LBB0_35
# %bb.38:                               #   in Loop: Header=BB0_34 Depth=1
	.loc	1 446 11                        # <stdin>:446:11
	addq	$1, %r13
	.loc	1 399 5                         # <stdin>:399:5
	addq	$18400, 16(%rsp)                # 8-byte Folded Spill
                                        # imm = 0x47E0
	addl	$-2000, %ebx                    # imm = 0xF830
	addl	$2000, %r12d                    # imm = 0x7D0
	.loc	1 397 11                        # <stdin>:397:11
	cmpq	$2000, %r13                     # imm = 0x7D0
	.loc	1 399 5                         # <stdin>:399:5
	jne	.LBB0_34
# %bb.39:                               # %print_array.exit
	.loc	1 402 11                        # <stdin>:402:11
	movq	(%r15), %rdi
	.loc	1 407 11                        # <stdin>:407:11
	movl	$str6, %esi
	movl	$str3, %edx
	xorl	%eax, %eax
	callq	fprintf@PLT
	.loc	1 409 11                        # <stdin>:409:11
	movq	(%r15), %rcx
	.loc	1 412 11                        # <stdin>:412:11
	movl	$str7, %edi
	movl	$22, %esi
	movl	$1, %edx
	callq	fwrite@PLT
	jmp	.LBB0_40
.Ltmp2:
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
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
	.asciz	"C"
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
	.byte	5                               # DW_FORM_data2
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
	.byte	1                               # Abbrev [1] 0xb:0x5c DW_TAG_compile_unit
	.long	.Linfo_string0                  # DW_AT_producer
	.short	2                               # DW_AT_language
	.long	.Linfo_string1                  # DW_AT_name
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Linfo_string2                  # DW_AT_comp_dir
                                        # DW_AT_GNU_pubnames
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	2                               # Abbrev [2] 0x2a:0xd DW_TAG_subprogram
	.long	.Linfo_string3                  # DW_AT_linkage_name
	.long	.Linfo_string3                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.short	370                             # DW_AT_decl_line
                                        # DW_AT_external
	.byte	1                               # DW_AT_inline
	.byte	3                               # Abbrev [3] 0x37:0x2f DW_TAG_subprogram
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
	.long	.Linfo_string4                  # DW_AT_linkage_name
	.long	.Linfo_string4                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	4                               # Abbrev [4] 0x50:0x15 DW_TAG_inlined_subroutine
	.long	42                              # DW_AT_abstract_origin
	.quad	.Ltmp1                          # DW_AT_low_pc
	.long	.Ltmp2-.Ltmp1                   # DW_AT_high_pc
	.byte	1                               # DW_AT_call_file
	.short	362                             # DW_AT_call_line
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
	.asciz	"print_array"                   # string offset=25
.Linfo_string4:
	.asciz	"main"                          # string offset=37
	.section	.debug_pubnames,"",@progbits
	.long	.LpubNames_end0-.LpubNames_begin0 # Length of Public Names Info
.LpubNames_begin0:
	.short	2                               # DWARF Version
	.long	.Lcu_begin0                     # Offset of Compilation Unit Info
	.long	103                             # Compilation Unit Length
	.long	55                              # DIE offset
	.asciz	"main"                          # External Name
	.long	42                              # DIE offset
	.asciz	"print_array"                   # External Name
	.long	0                               # End Mark
.LpubNames_end0:
	.section	.debug_pubtypes,"",@progbits
	.long	.LpubTypes_end0-.LpubTypes_begin0 # Length of Public Types Info
.LpubTypes_begin0:
	.short	2                               # DWARF Version
	.long	.Lcu_begin0                     # Offset of Compilation Unit Info
	.long	103                             # Compilation Unit Length
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
