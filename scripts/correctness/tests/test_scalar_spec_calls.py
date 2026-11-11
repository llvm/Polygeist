"""Pure scalar calls retain their typed declaration through spec import."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib_v2'))
import build_lib_v2 as B


class ScalarSpecCalls(unittest.TestCase):
    body = '%r = func.call @scalar_helper(%x) : (f32) -> f32'

    def test_pure_typed_declaration_is_preserved_without_callee_name_rules(self):
        declaration = 'func.func private @scalar_helper(f32) -> f32 attributes {polygeist.pure}'
        self.assertEqual(B.scalar_call_declarations(declaration, self.body), [declaration])

    def test_observable_or_missing_declaration_cannot_enter_reference(self):
        for text in ['', 'func.func private @scalar_helper(f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>}']:
            with self.assertRaises(B.DefnError):
                B.scalar_call_declarations(text, self.body)

    def test_memory_argument_is_not_a_scalar_reference_call(self):
        declaration = 'func.func private @scalar_helper(memref<?xf32>) -> f32 attributes {polygeist.pure}'
        with self.assertRaises(B.DefnError):
            B.scalar_call_declarations(declaration, self.body)


if __name__ == '__main__':
    unittest.main()
