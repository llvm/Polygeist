from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'scripts/correctness/lib_v2').is_dir())
sys.path[:0]=[str(HERE),str(ROOT/'scripts/correctness/lib_v2')]
import mlir_lite as ML
import runtime_adapter as RA

class ReferenceParseCache(unittest.TestCase):
    def setUp(self): RA._parsed_reference_module.cache_clear()
    def tearDown(self): RA._parsed_reference_module.cache_clear()

    @staticmethod
    def body(number):
        return '''module {
  func.func @reference(%input: f32) -> f32 {
    %c = arith.constant NUMBER : f32
    %sum = arith.addf %input, %c : f32
    return %sum : f32
  }
}'''.replace('NUMBER',str(number)+'.000000e+00')

    def test_exact_text_reuse_and_same_path_content_change(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(ML,'Module',wraps=ML.Module) as parse:
            path=Path(tmp)/'library.mlir';path.write_text(self.body(1))
            first=RA._parsed_reference_module(path.read_text())
            self.assertIs(first,RA._parsed_reference_module(path.read_text()))
            self.assertEqual(parse.call_count,1)
            path.write_text(self.body(2))
            changed=RA._parsed_reference_module(path.read_text())
            self.assertIsNot(first,changed)
            self.assertEqual(parse.call_count,2)
            self.assertIn('2.000000e+00',changed.funcs[0].ops[0].text)
            self.assertIn('1.000000e+00',first.funcs[0].ops[0].text)

    def test_old_catalogs_are_evicted(self):
        with mock.patch.object(ML,'Module',wraps=ML.Module) as parse:
            for value in range(5):RA._parsed_reference_module(self.body(value))
            self.assertEqual(RA._parsed_reference_module.cache_info().currsize,4)
            RA._parsed_reference_module(self.body(0))
            self.assertEqual(parse.call_count,6)

if __name__=='__main__':unittest.main()
