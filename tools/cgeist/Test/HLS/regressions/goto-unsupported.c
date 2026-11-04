// `goto` is reported as an error: cgeist's code for it was silently wrong
// (a forward goto out of an if was dropped, a backward one made a body run
// once, `goto skip; z = 7; skip:` ran `z = 7`), CIRCT OPEN_ISSUES O27
// (held-out chstone dfadd / dfsin).
// RUN: not cgeist %s %resourcedir --function='*' -S -o /dev/null 2>&1 | FileCheck %s
// CHECK: error: 'goto' is not supported by cgeist (its code would be wrong)
int top(int a, int b) {
  int z;
  if (a > b) goto big;
  z = b - a;
  goto done;
big:
  z = a - b;
done:
  return z * 3;
}
