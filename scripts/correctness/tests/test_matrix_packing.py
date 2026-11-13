"""Physical matrix views must reach BLAS without losing their logical layout."""
import importlib.util
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve()
while not (ROOT / 'scripts/correctness/lib_v2/lower_v2.py').exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / 'scripts/correctness/lib_v2'))
source = os.environ.get('TEST_MATRIX_LOWER', str(ROOT / 'scripts/correctness/lib_v2/lower_v2.py'))
spec = importlib.util.spec_from_file_location('matrix_lower_tested', source)
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)


def program(inner):
    ty = f'memref<5x5xf64, strided<[20, {inner}], offset: ?>>'
    types = f'i32, i32, f64, i32, {ty}, memref<5xf64>, memref<5xf64>'
    return f'''module {{
  func.func @apply(%m: i32, %n: i32, %alpha: f64, %lda: i32,
                   %a: {ty}, %x: memref<5xf64>, %y: memref<5xf64>) {{
    kernel.launch @openblas_dgemv_t_beta1(%m, %n, %alpha, %lda, %a, %x, %y) : ({types}) -> ()
    return
  }}
}}
'''


class MatrixPacking(unittest.TestCase):
    def test_strided_matrix_adapts_and_verifies(self):
        for inner in ('4', '?'):
            text = L.lower(program(inner), do_bufferize=False, backend='cpu')
            self.assertIn('func.call @polygeist_v2_matrix_pack(', text)
            self.assertIn('func.call @polygeist_v2_matrix_release(', text)
            self.assertLess(text.index('func.call @polygeist_v2_matrix_require_disjoint('),
                            text.index('func.call @polygeist_v2_matrix_pack('))
            self.assertIn('func.call @polygeist_v2_cublasDgemv(', text)
            p = subprocess.run([str(ROOT / 'build/bin/polygeist-opt')], input=text,
                               text=True, capture_output=True)
            self.assertEqual(p.returncode, 0, p.stderr)

    def test_contiguous_matrix_does_not_pack(self):
        text = program('1')
        output = L.lower(text, do_bufferize=False, backend='cpu')
        self.assertNotIn('matrix_pack', output)

    def test_actual_blas_and_capacity_controls(self):
        here = Path(__file__).resolve().parent
        helper = Path(os.environ.get('TEST_MATRIX_RUNTIME', str(ROOT / 'runtime')))
        if not (helper / 'polygeist_v2_matrix_layout.inc').exists():
            helper = here
        source = here / 'test_matrix_layout.c'
        with tempfile.TemporaryDirectory() as folder:
            binary = Path(folder) / 'check'
            p = subprocess.run(['gcc', '-O2', '-I' + str(helper), str(source),
                                '-lopenblas', '-lm', '-o', str(binary)], capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            for arg, success in [('', True), ('capacity', False), ('overflow', False), ('overlap', False)]:
                p = subprocess.run([str(binary)] + ([arg] if arg else []), capture_output=True,
                                   preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_CORE, (0, 0)))
                self.assertEqual(p.returncode == 0, success, p.stderr)


if __name__ == '__main__':
    unittest.main()
