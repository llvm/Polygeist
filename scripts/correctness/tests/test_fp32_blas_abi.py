"""Precision is part of the imported API contract, including scalar parameters."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'lib_v2'))
import abi_v2 as A
import lower_v2 as L
import gen_v2_runtime as G

class FP32BlasABI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calls,cls.specs,errors=A.build_table(['openblas_sgemv_t_beta0_alpha1','openblas_dgemv_t_beta0'])
        if errors:raise AssertionError(errors)

    def test_both_precisions_derive_scalar_types(self):
        for call,ty in [('cublasSgemv','float'),('cublasDgemv','double')]:
            ca=self.calls[call]
            self.assertEqual([p.elem for p in ca.params if p.tkind=='scalar'],[ty,ty])
            self.assertIn(ty+' alpha',G.proto(ca))
            self.assertIn('const '+ty+' *',G.gen_cpu({call:ca}))

    def test_no_implicit_scalar_precision_conversion(self):
        e=L.Emitter('','test')
        for source,target in [('f32','f64'),('f64','f32'),('bf16','f32')]:
            with self.assertRaises(L.LowerError):L.cast_scalar(e,'%x',source,target)
        self.assertEqual(L.cast_scalar(e,'%x','f32','f32'),'%x')
        self.assertEqual(e.lines,[])

    def test_reference_and_vendor_precision_must_agree(self):
        sa=copy.deepcopy(self.specs['openblas_sgemv_t_beta0_alpha1'])
        next(a for a in sa.args if a.name=='A').ctype='double'
        with self.assertRaises(A.AbiError):A._build_call(sa.call,[sa])

    def test_fortran_precision_must_agree(self):
        ca=copy.deepcopy(self.calls['cublasSgemv'])
        next(p for p in ca.params if p.name=='alpha').elem='double'
        with self.assertRaises(A.AbiError):A._fortran_binding(ca,[self.specs['openblas_sgemv_t_beta0_alpha1']])

if __name__=='__main__':unittest.main()
