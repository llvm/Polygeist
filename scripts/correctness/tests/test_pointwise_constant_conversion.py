"""FP32 backend constants preserve explicit source precision conversions."""
import math
import unittest

import test_pointwise_scalar_closure as scalar_tests
import abi_v2 as A


class ConstantConversion(unittest.TestCase):
    compile = scalar_tests.PointwiseScalarClosure.compile

    def converted_graph(self, literal, source='f64', cast_source='f64'):
        return self.compile(
            '%answer = arith.mulf %x, %narrow : f32',
            '%wide = arith.constant ' + literal + ' : ' + source + '\n'
            '%narrow = arith.truncf %wide : ' + cast_source + ' to f32')

    def test_round_to_even_and_subnormal_boundaries(self):
        # Exact hexadecimal expectations, independent of the conversion helper.
        for literal, expected in [
            ('1.000000059604644775390625', '0x1.0000000000000p+0'),
            ('1.000000178813934326171875', '0x1.0000040000000p+0'),
            ('0.79788456080286541', '0x1.9884540000000p-1'),
            ('7.006492321624085e-46', '0x0.0p+0'),
            ('1.401298464324817e-45', '0x1.0000000000000p-149'),
        ]:
            with self.subTest(literal=literal):
                graph = self.converted_graph(literal)
                self.assertIn(expected, [v.hex() for v in graph['scalars']])
                self.assertEqual(graph['nodes'], 1)

    def test_negative_zero_is_not_rebound_to_positive_zero(self):
        graph = self.converted_graph('-0.0')
        self.assertIn('-0x0.0p+0', [v.hex() for v in graph['scalars']])
        operand = (graph['words'][0] >> 8) & 255
        self.assertEqual(math.copysign(1., graph['scalars'][operand - 4]), -1.)

    def test_source_type_cannot_be_invented(self):
        with self.assertRaises(A.AbiError):
            self.converted_graph('1.0', source='f32')
        with self.assertRaises(A.AbiError):
            self.converted_graph('1.0', source='i64', cast_source='i64')

    def test_dynamic_conversion_and_direct_fp64_arithmetic_are_rejected(self):
        for outer, body in [
            ('%narrow = arith.truncf %limit : f64 to f32',
             '%answer = arith.mulf %x, %narrow : f32'),
            ('%wide = arith.constant 1.5 : f64',
             '%answer = arith.mulf %x, %wide : f32'),
        ]:
            with self.subTest(outer=outer), self.assertRaises(A.AbiError):
                self.compile(body, outer, scalar='%limit' in outer)

    def test_overflow_and_nonfinite_constants_are_rejected(self):
        for literal in ['1.0e100', '-1.0e100', 'inf', 'nan']:
            with self.subTest(literal=literal), self.assertRaises(A.AbiError):
                self.converted_graph(literal)


if __name__ == '__main__':
    unittest.main()
