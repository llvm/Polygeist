"""Honor upstream arith.mulf/muli Commutative without admitting reassociation."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kernel_match as K

@patch.dict(os.environ, {'POLYGEIST_NUMERIC_MODEL': 'strict'})
class StrictMultiplicationCommutativity(unittest.TestCase):
    def setUp(self):
        K.configure_matcher('egglog')

    def test_operand_exchange_is_an_eqsat_proof(self):
        T = K.Term
        a, b, c = T.In(0), T.In(1), T.Out(0)
        self.assertTrue(K.equivalent(a * b, b * a))
        self.assertEqual(K.body_matches_template(c + a * b, c + b * a), {})
        self.assertEqual(K.body_matches_template(a * T.Cap('scale'),
                                                T.Cap('scale') * a),
                         {'scale': ('Cap', 'scale')})
        # The no-EqSat mode remains syntactic, so this is measurable separately.
        K.configure_matcher('syntactic')
        self.assertIsNone(K.body_matches_template(a * b, b * a))

    def test_other_arithmetic_changes_remain_rejected(self):
        T = K.Term
        a, b, c = T.In(0), T.In(1), T.In(2)
        for left, right in [((a * b) * c, a * (b * c)),
                            ((a + b) + c, a + (b + c)),
                            (a / b, b / a), (a - b, b - a),
                            (a * T.Lit(0.), T.Lit(0.)),
                            (a * T.Lit(1.), a),
                            (a * b, T.Unary('fpext:f32 to f64', a) * b),
                            (a * K.literal(-0.), a * T.Lit(0.))]:
            with self.subTest(left=left, right=right):
                self.assertFalse(K.equivalent(left, right))
                self.assertIsNone(K.body_matches_template(left, right))

if __name__ == '__main__':
    unittest.main()
