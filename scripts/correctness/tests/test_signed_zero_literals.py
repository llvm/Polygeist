import os
from pathlib import Path
import sys
import unittest
from unittest import mock
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE if (HERE/'kernel_match.py').exists() else HERE.parent))
import kernel_match as K

@mock.patch.dict(os.environ,{'POLYGEIST_NUMERIC_MODEL':'strict'})
class SignedZeroLiterals(unittest.TestCase):
    def setUp(self): K.configure_matcher('egglog')

    def test_distinct_literal_signs(self):
        T=K.Term
        for left,right in [(K.literal(-0.0),T.Lit(0.0)),
                           (T.In(0)+K.literal(-0.0),T.In(0)+T.Lit(0.0))]:
            self.assertIsNone(K.body_matches_template(left,right))
            self.assertFalse(K.equivalent(left,right))
        exact=T.In(0)+K.literal(-0.0)
        self.assertEqual(K.body_matches_template(exact,exact),{})
        self.assertTrue(K.equivalent(exact,exact))

    def test_capture_binding_preserves_sign(self):
        T=K.Term
        bound=K.body_matches_template(T.In(0)*K.literal(-0.0),T.In(0)*T.Cap('scale'))
        self.assertIsNotNone(bound)
        self.assertEqual(K._literal_identity(bound['scale']),('LiteralBits','-0x0.0p+0'))
        self.assertIsNone(K.body_matches_template(K.literal(-0.0)+T.Lit(0.0),
                                                 T.Cap('x')+T.Cap('x')))

    def test_syntactic_identity_and_conservative_eqsat(self):
        T=K.Term
        K.configure_matcher('syntactic')
        self.assertIsNone(K.body_matches_template(K.literal(-0.0),T.Lit(0.0)))
        self.assertFalse(K.equivalent(K.literal(-0.0),T.Lit(0.0)))
        K.configure_matcher('egglog')
        # This conditional identity is valid, but is conservatively declined
        # while f64 e-graph atoms cannot preserve the negative-zero literal.
        p,x,d=T.In(0),T.In(1),T.In(2);z=K.literal(-0.0);limit=T.Lit(6.0)
        selected=T.Select(p,z,x)
        left=T.Select(T.Cmp('olt',limit,selected),d,selected)
        right=T.Select(p,z,T.Select(T.Cmp('olt',limit,x),d,x))
        self.assertIsNone(K.body_matches_template(left,right))

    def test_raised_literal_encoding_keeps_both_signs(self):
        def encode(value):
            text='''%constant = arith.constant VALUE : f32
  %result = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}
    outs(%out : tensor<4xf32>) {
  ^bb0(%unused: f32):
    linalg.yield %constant : f32
  } -> tensor<4xf32>'''.replace('VALUE',value)
            return K.encode_body(K.parse_generics(text,infer_outputs_from_yield=True)[0])
        negative,positive=encode('-0.000000e+00'),encode('0.000000e+00')
        self.assertIn('NegativeZero',str(negative))
        self.assertIsNone(K.body_matches_template(negative,positive))
        self.assertEqual(K.body_matches_template(negative,negative),{})

if __name__=='__main__': unittest.main()
