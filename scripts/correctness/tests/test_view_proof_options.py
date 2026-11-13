"""Proof verification accepts precisely the optional view passes we produce."""
import copy
from contextlib import ExitStack
import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve()
while not (ROOT / 'scripts/correctness/spec_match.py').exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / 'scripts/correctness'))
sys.path.insert(0, str(ROOT / 'scripts/correctness/lib_v2'))
source = os.environ.get('TEST_SPEC_MATCH', str(ROOT / 'scripts/correctness/spec_match.py'))
spec = importlib.util.spec_from_file_location('tested_spec_match', source)
SM = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SM)


class ViewProofOptions(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name, value in [('semantic_configuration', {}), ('semantic_inputs', {}),
                            ('abi_digest', 'abi'), ('registry', {}),
                            ('capabilities', (set(), {})), ('digest', 'hash')]:
            self.stack.enter_context(patch.object(SM, name, return_value=value))
        self.stack.enter_context(patch.object(SM.PROVENANCE, 'file_hash', return_value='compiler'))
        self.proof = dict(schema=2, pipeline='imported-specifications-only',
                          output_sha256='hash', backend='cuda', configuration={},
                          semantic_inputs={}, abi_sha256='abi', library_sha256='hash',
                          matcher_sha256='hash', emitted=[], matches=[],
                          view_normalization=dict(pass_name='--normalize-linalg-matching-views',
                                                  cleanup=['--cse'], compiler_sha256='compiler'))

    def test_emitted_options_verify(self):
        for suffix in ('', '=max-parallel-dims=2', '=max-operand-rank=2',
                       '=max-parallel-dims=2 max-operand-rank=2',
                       '=max-operand-rank=2 fuse-initializers=true',
                       '=max-parallel-dims=2 max-operand-rank=2 fuse-initializers=true'):
            with self.subTest(suffix=suffix):
                self.proof['view_normalization']['pass_name'] = '--normalize-linalg-matching-views' + suffix
                SM.verify('module {}', self.proof, backend='cuda')

    def test_initializer_fusion_requires_an_explicit_rank_limit(self):
        with self.assertRaisesRegex(SM.SpecificationError, 'rank limit'):
            SM.match('module {}', fuse_initializers=True)
        with self.assertRaisesRegex(SM.SpecificationError, 'boolean'):
            SM.match('module {}', max_operand_rank=2, fuse_initializers='yes')

    def test_unsupported_options_and_changed_compiler_fail(self):
        for suffix in ('=max-parallel-dims=-1', '=max-operand-rank=0',
                       '=max-parallel-dims=2 max-parallel-dims=3',
                       '=max-operand-rank=2 max-parallel-dims=2',
                       '=unknown=2', '=max-parallel-dims=2 --canonicalize',
                       '=fuse-initializers=true', '=max-operand-rank=2 fuse-initializers=false',
                       '=fuse-initializers=true max-operand-rank=2'):
            proof = copy.deepcopy(self.proof)
            proof['view_normalization']['pass_name'] += suffix
            with self.subTest(suffix=suffix), self.assertRaises(SM.SpecificationError):
                SM.verify('module {}', proof, backend='cuda')
        for key, value in [('compiler_sha256', 'other'), ('cleanup', ['--canonicalize'])]:
            proof = copy.deepcopy(self.proof)
            proof['view_normalization'][key] = value
            with self.subTest(key=key), self.assertRaises(SM.SpecificationError):
                SM.verify('module {}', proof, backend='cuda')


if __name__ == '__main__':
    unittest.main()
