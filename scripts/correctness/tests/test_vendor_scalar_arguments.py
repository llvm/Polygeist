"""Vendor runtime bindings preserve FP32 scalar values and reject mistyped ABI maps."""
import copy
from pathlib import Path
import sys
import unittest
HERE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(HERE),str(HERE/'lib_v2')]
import abi_v2 as A
import gen_v2_runtime as G
import runtime_adapter as R

class VendorScalarArguments(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calls,cls.specs,errors=A.build_table(['cutlass_rmsnorm_parameterized_f32','cutlass_rmsnorm_f32'])
        if errors:raise AssertionError(errors)

    def test_dynamic_scalar_reaches_typed_runtime_and_old_api_remains(self):
        source=G.gen_cuda(self.calls)
        self.assertIn('float epsilon',source)
        self.assertIn('polygeist_rmsnorm_f32_epsilon(n,',source)
        self.assertIn(', epsilon);',source)
        self.assertIn('polygeist_rmsnorm_f32(n,',source)
        old=self.specs['cutlass_rmsnorm_f32']
        self.assertNotIn('epsilon',[a.name for a in old.args])

    def test_fp64_scalar_does_not_silently_narrow(self):
        spec=self.specs['cutlass_rmsnorm_parameterized_f32']
        args=copy.deepcopy(spec.args)
        next(a for a in args if a.name=='epsilon').ctype='double'
        target=A.manifest_entry(spec.spec)[1]['target']
        with self.assertRaises(A.AbiError):R.contract(A,spec.spec,target,spec.ret,args)

    def test_array_cannot_supply_scalar_slot(self):
        spec=self.specs['cutlass_rmsnorm_parameterized_f32']
        target=copy.deepcopy(A.manifest_entry(spec.spec)[1]['target'])
        target['runtime_adapter']['arguments']['Epsilon']='input'
        with self.assertRaises(A.AbiError):R.contract(A,spec.spec,target,spec.ret,spec.args)

if __name__=='__main__':unittest.main()
