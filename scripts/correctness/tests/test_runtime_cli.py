"""Runtime generation works from an unrelated directory without PYTHONPATH."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]

class StandaloneRuntimeGeneration(unittest.TestCase):
    def test_generates_from_clean_import_environment(self):
        env = dict(os.environ)
        env.pop('PYTHONPATH', None)
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(ROOT / 'scripts/correctness/lib_v2/gen_v2_runtime.py'),
                 '--output-dir', directory], cwd=directory, env=env,
                text=True, capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr[-3000:])
            output = Path(directory) / 'polygeist_v2_rt_cuda.c'
            self.assertTrue(output.is_file())
            self.assertIn('polygeist_cudnn_pointwise_graph_f32(', output.read_text())

if __name__ == '__main__':
    unittest.main()
