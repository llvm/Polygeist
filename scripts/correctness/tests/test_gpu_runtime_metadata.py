"""CUDA pointer-residency is an explicit backend ABI contract."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib_v2'))
import abi_v2 as A
import lower_v2 as L


class GpuRuntimeMetadata(unittest.TestCase):
    def test_only_cuda_lowering_marks_imported_runtime(self):
        calls, specs, errors = A.build_table(['aten_softmax_f32'])
        self.assertFalse(errors)
        text = '''module {
  func.func @apply(%n: i32, %x: memref<?xf32>, %y: memref<?xf32>) {
    kernel.launch @aten_softmax_f32(%n, %x, %y) : (i32, memref<?xf32>, memref<?xf32>) -> ()
    return
  }
}'''
        for backend in ('cpu', 'cuda'):
            out = L.rewrite(text, {}, specs, calls, [], backend)
            self.assertEqual('attributes {polygeist.gpu_runtime}' in out,
                             backend == 'cuda')
            self.assertNotIn('polygeist.cuda_graph_safe', out)


if __name__ == '__main__':
    unittest.main()
