"""Two proven source layouts can share a flat-pointer vendor ABI."""
from pathlib import Path
import copy
import sys
import unittest
from unittest.mock import patch
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / 'lib_v2')]
import abi_v2 as A
import gen_v2_runtime as G

class SharedFlatPointerABI(unittest.TestCase):
    old = 'openblas_sgemm_nn_m1_b0_alpha1'
    general = 'openblas_sgemm_nn_b0_alpha1'

    def test_both_layouts_keep_the_same_vendor_wrapper(self):
        before, _, before_errors = A.build_table([self.old])
        calls, specs, errors = A.build_table([self.old, self.general])
        self.assertFalse(before_errors)
        self.assertFalse(errors)
        self.assertEqual(set(specs), {self.old, self.general})
        self.assertEqual(set(calls), {'cublasSgemm'})
        self.assertEqual(G.gen_cuda(calls), G.gen_cuda(before))
        self.assertEqual(G.gen_cpu(calls), G.gen_cpu(before))
        arrays = {p.name: p.rank for p in calls['cublasSgemm'].params if p.tkind == 'array'}
        self.assertEqual(arrays, {'A': None, 'B': 2, 'C': None})

    def test_unsupported_rank_and_missing_layout_fail_closed(self):
        sas = [A.bind_spec(self.old), A.bind_spec(self.general)]
        invalid = copy.deepcopy(sas)
        next(a for a in invalid[1].args if a.name == 'A').rank = 3
        with self.assertRaisesRegex(A.AbiError, 'array rank'):
            A._build_call('cublasSgemm', invalid)
        with patch.object(A, '_layout_links', return_value={}):
            with self.assertRaisesRegex(A.AbiError, 'layout'):
                A._build_call('cublasSgemm', sas)

if __name__ == '__main__':
    unittest.main()
