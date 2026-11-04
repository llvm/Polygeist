/*
+--------------------------------------------------------------------------+
| CHStone : a suite of benchmark programs for C-based High-Level Synthesis |
| ======================================================================== |
|                                                                          |
| * Collected and Modified : Y. Hara, H. Tomiyama, S. Honda,               |
|                            H. Takada and K. Ishii                        |
|                            Nagoya University, Japan                      |
|                                                                          |
| * Remark :                                                               |
|    1. This source code is modified to unify the formats of the benchmark |
|       programs in CHStone.                                               |
|    2. Test vectors are added for CHStone.                                |
|    3. If "main_result" is 0 at the end of the program, the program is    |
|       correctly executed.                                                |
|    4. Please follow the copyright of each benchmark program.             |
+--------------------------------------------------------------------------+
*/
/*
 * Copyright 1992 by Jutta Degener and Carsten Bormann, Technische
 * Universitaet Berlin.  See the accompanying file "COPYRIGHT" for
 * details.  THERE IS ABSOLUTELY NO WARRANTY FOR THIS SOFTWARE.
 */

/*$Header: /home/kbs/jutta/src/gsm/gsm-1.0/inc/RCS/private.h,v 1.4 1994/11/28 20:25:03 jutta Exp $*/

#ifndef	PRIVATE_H
#define	PRIVATE_H

typedef short word;		/* 16 bit signed int    */
typedef long longword;		/* 32 bit signed int    */

#define	MIN_WORD	((-32767)-1)
#define	MAX_WORD	( 32767)

#define	SASR(x, by)	((x) >> (by))

word gsm_add(word a, word b);
word gsm_mult(word a, word b);
word gsm_mult_r(word a, word b);
word gsm_abs(word a);
word gsm_norm(longword a);
word gsm_div(word num, word denum);

#define GSM_MULT_R(a, b)	gsm_mult_r(a, b)
#define GSM_MULT(a, b)		gsm_mult(a, b)
#define GSM_ADD(a, b)		gsm_add(a, b)
#define GSM_ABS(a)		gsm_abs(a)

#endif /* PRIVATE_H */
