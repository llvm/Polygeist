"""A reference's dynamic dimensions must not erase known application extents."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from region_match.matcher import Binding, _Emit, _LaunchSpec, _launch

class LaunchShapeTests(unittest.TestCase):
    def emit(self, actual, expected, rank=2):
        program = SimpleNamespace(func=SimpleNamespace(args=[('%array', actual)]))
        emitter = _Emit(program, 'test')
        spec = _LaunchSpec('reference', [('%x', expected)], {'%x': 'x'}, {}, {})
        binding = Binding({'%x': ('%array', None, tuple(range(rank)))}, {}, {})
        _launch(emitter, spec, binding)
        return '\n'.join(emitter.lines)

    def test_static_dimensions_are_kept_for_dynamic_reference(self):
        for actual in ['memref<3x7xf32>', 'memref<?x7xf32>']:
            with self.subTest(actual=actual):
                text = self.emit(actual, 'memref<?x?xf32>')
                self.assertNotIn('memref.cast', text)
                self.assertIn(f'kernel.launch @reference(%array) : ({actual})', text)

    def test_existing_static_requirement_still_needs_checked_cast(self):
        text = self.emit('memref<?x7xf32>', 'memref<3x7xf32>')
        self.assertIn('memref.cast %array : memref<?x7xf32> to memref<3x7xf32>', text)

    def test_strided_layout_is_preserved(self):
        actual = 'memref<3x7xf32, strided<[9, 1], offset: 2>>'
        expected = 'memref<?x?xf32, strided<[9, 1], offset: 2>>'
        text = self.emit(actual, expected)
        self.assertNotIn('memref.cast', text)
        self.assertIn(actual, text)

if __name__ == '__main__':
    unittest.main()
