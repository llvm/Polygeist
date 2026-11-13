/* Bug 7 minimal reproducer (netlib DGEMM with ALPHA = 0: the
 * `IF (ALPHA.EQ.ZERO) ... RETURN` quick return becomes `if (1) { ...; return; }`
 * and the whole general path below it is dead).
 *   cgeist bug7_min.c --function=bug7_min --resource-dir=/usr/lib/clang/14 -O0 -S -o out.mlir
 * -> SIGSEGV in polygeist-canonicalize (scf.if RemoveStaticCondition ->
 *    RewriterBase::eraseOp -> SuccessorRange(Block* = nullptr)).
 * Needs >= 3 nested loops in the dead code; a non-constant condition does not crash. */
void bug7_min(int n) {
  if (1)
    return;
  for (int j = 0; j < n; j++)
    for (int i = 0; i < n; i++)
      for (int l = 0; l < n; l++)
        ;
}
