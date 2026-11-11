"""Imported scalar primitives bind only to exact typed vendor interfaces."""
from pathlib import Path
import sys
import tempfile
import types
import unittest

HERE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(HERE),str(HERE/'lib_v2')]
import abi_v2 as A
import runtime_adapter as R


class UnaryVendorBinding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calls,cls.specs,errors=A.build_table(['aten_asin_f32'])
        if errors:raise AssertionError(errors)

    def test_real_vendor_header_and_runtime_choose_same_primitive(self):
        p=A.resolve_proto('cuTENSOR','cutensorPermute')
        self.assertEqual([x.name for x in p.params],['handle','plan','alpha','A','B','stream'])
        graph=next(iter(self.calls.values())).runtime_adapter['primitive']
        self.assertEqual(graph['vendor_opcode'],'CUTENSOR_OP_ASIN')
        self.assertEqual(graph['opcode'],'POLYGEIST_CUTENSOR_UNARY_ASIN')

    def reject_modified_call(self,old,new):
        text=(A.SPEC_DIR.parent/'kernel_library_v3.mlir').read_text()
        self.assertIn(old,text)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'kernel_library_v3.mlir').write_text(text.replace(old,new))
            provider=types.SimpleNamespace(REPO=A.REPO,SPEC_DIR=root/'lib_v3',
                AbiError=A.AbiError,enum_of_constant=A.enum_of_constant)
            with self.assertRaises(A.AbiError):
                R.unary_reference(provider,'aten_asin_f32',self.specs['aten_asin_f32'].args,'n')

    def test_double_symbol_cannot_be_bound_to_float_opcode(self):
        self.reject_modified_call('func.call @asinf(', 'func.call @asin(')

    def test_unknown_callee_cannot_be_inferred_from_argument_shape(self):
        self.reject_modified_call('func.call @asinf(', 'func.call @unknown(')


if __name__=='__main__':unittest.main()
