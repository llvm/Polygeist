from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from region_match.matcher import _c_prototype_dims, _load_specs_uncached


class ReferenceDimensionsTest(unittest.TestCase):
    def dimensions(self, source):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'reference.c'
            path.write_text(source)
            return _c_prototype_dims(str(path))

    def test_typedef_and_type_macro(self):
        for scalar in ['double', 'float', 'element_t', 'LIBRARY_SCALAR']:
            self.assertEqual(self.dimensions(
                f'void f(int pitch, const {scalar} A[][pitch]);'),
                {'pitch': ('A', 1)})

    def test_parameter_macros_retain_leading_dimensions(self):
        self.assertEqual(self.dimensions(
            '#define INPUTS int ld, SCALAR A[][ld],\n'
            '#define OUTPUTS SCALAR C[][out_ld]\n'
            'void f(INPUTS int out_ld, OUTPUTS) {}'),
            {'ld': ('A', 1), 'out_ld': ('C', 1)})

    def test_comments_and_return_expressions_are_not_declarations(self):
        self.assertEqual(self.dimensions(
            '/* double fake[][wrong]; */\n'
            'double f(double A[][pitch]) { return A[i][j]; }'),
            {'pitch': ('A', 1)})

    def test_macro_expression_is_not_an_array_declaration(self):
        self.assertEqual(self.dimensions(
            '#define READ A[k][i]\n'
            '#if TRANSPOSE\n#define READ B[i][k]\n#else\n#define READ B[k][i]\n#endif\n'
            '#define PARAMS scalar_t A[][ld]\n'
            'void f(int ld, PARAMS);'), {'ld': ('A', 1)})

    def test_reference_path_comes_from_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'manifest').mkdir()
            (root / 'ref_c').mkdir()
            (root / 'ref_c/shared.c').write_text('void f(scalar_t A[][ld]);')
            (root / 'ref_c/variant.c').write_text('void f(scalar_t B[][pitch]);')
            (root / 'manifest/catalog.json').write_text(json.dumps(dict(
                routine='different_name', ref_c='shared.c', specs=[
                    dict(name='first'), dict(name='second', ref_c='variant.c')])))
            with patch('region_match.matcher._load_one', return_value=None) as load:
                _load_specs_uncached(str(root))
            self.assertEqual(load.call_args_list[0].args[3], {'ld': ('A', 1)})
            self.assertEqual(load.call_args_list[1].args[3], {'pitch': ('B', 1)})


if __name__ == '__main__':
    unittest.main()
