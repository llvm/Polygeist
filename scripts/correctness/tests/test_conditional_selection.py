"""Bounded ordered selection rules preserve exact floating selections."""
import math
import os
from pathlib import Path
import struct
import sys
import unittest
from unittest import mock

# The staged module is first for private validation; production uses the same test.
HERE = Path(__file__).resolve().parent
if (HERE / 'kernel_match.py').exists():
    sys.path.insert(0, str(HERE))
else:
    sys.path.insert(0, str(HERE.parent))
import kernel_match as K


@mock.patch.dict(os.environ, {'POLYGEIST_NUMERIC_MODEL': 'strict'})
class ConditionalSelection(unittest.TestCase):
    def setUp(self):
        K.configure_matcher('egglog')

    @staticmethod
    def terms(kind, reverse, literal, bound):
        T = K.Term
        p, value, alternative = T.In(0), T.In(1), T.In(2)
        lit, lim = T.Lit(literal), T.Lit(bound)
        selected = T.Select(p, lit, value)
        cmp = lambda x: T.Cmp(kind, lim, x) if reverse else T.Cmp(kind, x, lim)
        return (T.Select(cmp(selected), alternative, selected),
                T.Select(p, lit, T.Select(cmp(value), alternative, value)))

    def test_constant_guards_and_exact_predicates(self):
        positives = negatives = 0
        for kind in ('olt', 'ole'):
            for reverse in (False, True):
                for literal, bound in ((0., 6.), (6., 0.), (-2., 5.),
                                       (5., -2.), (0., 0.), (-1., 4.)):
                    left, right = self.terms(kind, reverse, literal, bound)
                    x, y = (bound, literal) if reverse else (literal, bound)
                    expected = x >= y if kind == 'olt' else x > y
                    self.assertEqual(K.body_matches_template(left, right) == {}, expected,
                                     (kind, reverse, literal, bound))
                    positives += expected
                    negatives += not expected
        self.assertEqual((positives, negatives), (12, 12))
        # Unordered comparisons, integer predicates and an unsupported float
        # predicate must not borrow ordered-float constant guards.
        for kind in ('ult_float', 'ule_float', 'slt', 'ult', 'ogt'):
            left, right = self.terms(kind, True, 0., 6.)
            self.assertIsNone(K.body_matches_template(left, right), kind)

    def test_leaf_filter_and_rejection_controls(self):
        T = K.Term
        left, right = self.terms('olt', True, 0., 6.)
        self.assertEqual(K.body_matches_template(left, right), {})
        # Same leaf set does not prove equivalence: swapping branches is wrong.
        wrong = T.Select(T.In(0), T.Lit(0.),
                         T.Select(T.Cmp('olt', T.Lit(6.), T.In(1)), T.In(1), T.In(2)))
        self.assertIsNone(K.body_matches_template(left, wrong))
        changed = T.Select(T.In(0), T.Lit(0.),
                           T.Select(T.Cmp('olt', T.Lit(6.), T.In(3)), T.In(2), T.In(3)))
        self.assertIsNone(K.body_matches_template(left, changed))
        # Arithmetic order remains exact in strict mode.
        a, b, c = T.In(0), T.In(1), T.In(2)
        self.assertIsNone(K.body_matches_template((a + b) + c, a + (b + c)))
        large = ('In', 0)
        for i in range(33):
            large = ('Unary', str(i), large)
        self.assertFalse(K._conditional_proof_candidate(
            ('Select', ('In', 1), large, ('Lit', 0.)),
            ('Select', ('In', 1), ('Lit', 0.), large)))

    def test_exhaustive_small_grid_and_special_values(self):
        # Exhaust the chosen scalar grid, both predicates, both comparison
        # directions, both condition values, and independent branch values.
        values = [float(x) / 2 for x in range(-8, 9)] + [
            -0., float('inf'), -float('inf'), float('nan'),
            1e-45, -1e-45, 3.4028234663852886e38, -3.4028234663852886e38]
        constants = [-float('inf'), -2., -0., 0., 3., 6., float('inf')]
        bits = lambda x: struct.pack('d', x)
        checks = 0
        for kind in ('olt', 'ole'):
            cmp = (lambda a, b: a < b) if kind == 'olt' else (lambda a, b: a <= b)
            for reverse in (False, True):
                for literal in constants:
                    for bound in constants:
                        x, y = (bound, literal) if reverse else (literal, bound)
                        if not (x >= y if kind == 'olt' else x > y):
                            continue
                        compare = lambda a: cmp(bound, a) if reverse else cmp(a, bound)
                        for condition in (False, True):
                            for value in values:
                                for alternative in values:
                                    selected = literal if condition else value
                                    before = alternative if compare(selected) else selected
                                    after = literal if condition else (alternative if compare(value) else value)
                                    self.assertEqual(bits(before), bits(after))
                                    checks += 1
        self.assertEqual(checks, 122500)


if __name__ == '__main__':
    unittest.main()
