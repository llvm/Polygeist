"""Exact aliasing requires the validated per-element reference/backend path."""
from pathlib import Path
import copy
import sys
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE / 'lib_v2')]
import abi_v2 as A
import runtime_adapter as R
import lower_v2 as L


class PointwiseAliasContract(unittest.TestCase):
    def contract(self, spec, policy):
        calls, specs, errors = A.build_table([spec])
        self.assertFalse(errors)
        reference = specs[spec]
        target = copy.deepcopy(A.manifest_entry(spec)[1]['target'])
        target['runtime_adapter']['alias_policy'] = policy
        return R.contract(A, spec, target, reference.ret, reference.args)

    def test_identity_pointwise_body_accepts_exact_alias_contract(self):
        _, _, contract = self.contract('aten_add_f32', 'exact_or_disjoint')
        self.assertEqual(contract['alias_policy'], 'exact_or_disjoint')

    def test_other_runtime_adapters_cannot_assert_alias_safety(self):
        for spec in ['cutlass_rmsnorm_f32', 'aten_tan_f32']:
            with self.subTest(spec=spec), self.assertRaises(A.AbiError):
                self.contract(spec, 'exact_or_disjoint')

    def test_lowering_derives_alias_capability_from_validated_contract(self):
        calls, specs, errors = A.build_table(['aten_add_f32'])
        self.assertFalse(errors)
        text = ("module {\n  %r = kernel.launch @aten_add_f32(%n, %x, %y, %out) : "
                "(i32, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>\n}\n")
        prepared, _ = L.prepare(text, specs, calls)
        self.assertIn("{polygeist.elementwise_exact_alias}", prepared)
        self.assertEqual(len([m for line in prepared.splitlines()
                              if (m := L.LAUNCH_RE.match(line))]), 1)

    def test_input_cannot_assert_unvalidated_alias_capability(self):
        calls, specs, errors = A.build_table(['aten_mul_f32'])
        self.assertFalse(errors)
        text = ("module {\n  %r = kernel.launch @aten_mul_f32(%n, %x, %y, %out) "
                "{polygeist.elementwise_exact_alias, keep = 1 : i64} : "
                "(i32, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>\n}\n")
        prepared, _ = L.prepare(text, specs, calls)
        self.assertNotIn("polygeist.elementwise_exact_alias", prepared)
        self.assertIn("keep = 1 : i64", prepared)

    def test_partial_overlap_is_not_an_available_policy(self):
        with self.assertRaises(A.AbiError):
            self.contract('aten_add_f32', 'arbitrary_overlap')


if __name__ == '__main__':
    unittest.main()
