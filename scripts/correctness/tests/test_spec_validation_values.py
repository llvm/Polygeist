"""The imported-source validator must distinguish nonfinite result classes."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / 'lib_v2'))
import validate_specs as V


class FloatingValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        source = V.harness('probe', [], 'f64').replace('int main(void)', 'int compare(void)')
        source += '''
#include <stdlib.h>
static double left, right;
double probe_ref(void) { return left; }
double probe_impl(void) { return right; }
int main(int argc, char **argv) {
  if (argc != 3) return 99;
  left = strtod(argv[1], 0); right = strtod(argv[2], 0);
  return compare();
}
'''
        (cls.root / 'compare.c').write_text(source)
        subprocess.run(['cc', '-O0', str(cls.root / 'compare.c'), '-lm',
                        '-o', str(cls.root / 'compare')], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_matching_classes_and_finite_tolerance(self):
        for a, b in [('nan', 'nan'), ('inf', 'inf'), ('-inf', '-inf'),
                     ('1', '1.0000000000001')]:
            with self.subTest(a=a, b=b):
                result = subprocess.run([str(self.root / 'compare'), a, b], capture_output=True)
                self.assertEqual(result.returncode, 0, result.stdout)

    def test_different_results_are_rejected(self):
        for a, b in [('nan', '1'), ('1', 'nan'), ('nan', 'inf'),
                     ('inf', '-inf'), ('inf', '1'), ('1', 'inf'), ('1', '1.001')]:
            with self.subTest(a=a, b=b):
                result = subprocess.run([str(self.root / 'compare'), a, b], capture_output=True)
                self.assertEqual(result.returncode, 1, result.stdout)

    def test_abs_extraction_preserves_upstream_fabs(self):
        spec = importlib.util.spec_from_file_location('extract', HERE / 'generate_aten_scalar_extractions.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        expr, _, fn = module.MATH_UNARY['abs']
        text = module.body('abs', 'float x[N], float out[N]', f'out[i] = {expr};', extern=fn)
        root = HERE.parents[1]
        self.assertEqual(text, (root / 'issues/aten_c_kernels/aten_abs.c').read_text())


if __name__ == '__main__':
    unittest.main()
