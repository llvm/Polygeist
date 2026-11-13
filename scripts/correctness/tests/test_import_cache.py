"""Validation reuse must reject changed inputs/artifacts and incomplete evidence."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib_v2'))
from import_cache import ValidatedImports, digest


class ImportReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ref = self.root / 'reference.c'
        self.ref.write_text('#include "nested.h"\n')
        self.header = self.root / 'nested.h'
        self.header.write_text('/* first header */')
        self.fragment = self.root / 'manifest.json'
        self.fragment.write_text('{}')
        self.spec = SimpleNamespace(name='example', ref_c=self.ref,
                                    fragment=self.fragment, defines={}, raw={})
        self.cache = ValidatedImports.__new__(ValidatedImports)
        self.cache.B = SimpleNamespace(REPO=self.root, RAISED=self.root,
                                       R=SimpleNamespace(RESOURCE_DIR='/resource'))
        self.cache.E = SimpleNamespace(CLANG='clang')
        self.cache.directory = self.root / 'cache'
        self.cache.common = {'compiler': 'first', 'recipe': 'production'}
        for ext in ('.c', '.mlir', '_linalg.mlir', '_debuf.mlir', '.steps.json'):
            (self.root / ('example' + ext)).write_text('original ' + ext)
        self.result = dict(spec='example', status='RAISED', defn='kernel.defn @example',
                           meta={'args': ['input']}, validation={
                               stage: 'EXACT' for stage in ('affine', 'linalg', 'defn')})
        # Model preprocessing with a real header file; the integration experiment
        # also exercises actual clang and the complete import/validation path.
        self.addCleanup(patch.stopall)
        patch('import_cache.command', side_effect=lambda *a, **k: self.header.read_text()).start()

    def key(self):
        return self.cache.inputs(self.spec)

    def save(self):
        self.cache.save(self.spec, self.key(), self.result)

    def test_unchanged_success_reuses_exact_result(self):
        self.save()
        self.assertEqual(self.cache.load(self.spec, self.key()), self.result)

    def test_each_artifact_change_or_deletion_forces_rebuild(self):
        for ext in ('.c', '.mlir', '_linalg.mlir', '_debuf.mlir', '.steps.json'):
            with self.subTest(artifact=ext):
                self.save()
                path = self.root / ('example' + ext)
                original = path.read_text()
                path.write_text('changed ' + original)
                self.assertIsNone(self.cache.load(self.spec, self.key()))
                path.unlink()
                self.assertIsNone(self.cache.load(self.spec, self.key()))
                path.write_text(original)

    def test_source_manifest_header_and_compiler_changes_invalidate(self):
        for kind in ('source', 'manifest', 'header', 'compiler'):
            with self.subTest(input=kind):
                self.save()
                old = self.key()
                if kind == 'compiler':
                    self.cache.common['compiler'] += '-changed'
                else:
                    path = {'source': self.ref, 'manifest': self.fragment,
                            'header': self.header}[kind]
                    path.write_text(path.read_text() + '\n')
                self.assertNotEqual(old, self.key())
                self.assertIsNone(self.cache.load(self.spec, self.key()))

    def test_unvalidated_or_failed_stage_is_never_reused(self):
        for stage in ('affine', 'linalg', 'defn'):
            for verdict in ('UNSOUND', 'LOWER_FAIL', 'RUN_TIMEOUT', None):
                with self.subTest(stage=stage, verdict=verdict):
                    self.save()
                    path = self.cache.directory / 'example.json'
                    saved = json.loads(path.read_text())
                    saved['payload']['result']['validation'][stage] = verdict
                    saved['sha256'] = digest(saved['payload'])
                    path.write_text(json.dumps(saved))
                    self.assertIsNone(self.cache.load(self.spec, self.key()))

    def test_corrupt_cache_is_a_miss(self):
        self.save()
        path = self.cache.directory / 'example.json'
        saved = json.loads(path.read_text())
        saved['payload']['result']['defn'] += ' altered'
        path.write_text(json.dumps(saved))
        self.assertIsNone(self.cache.load(self.spec, self.key()))
        path.write_text('not json')
        self.assertIsNone(self.cache.load(self.spec, self.key()))

    def test_upstream_hash_failure_cannot_be_reused(self):
        self.fragment.write_text(json.dumps({'upstream_files': [
            {'path': 'nested.h', 'sha256': 'wrong'}]}))
        with self.assertRaisesRegex(ValueError, 'upstream reference changed'):
            self.key()

    def test_changes_during_build_block_publication(self):
        keys = {self.spec.name: self.key()}
        with patch.object(self.cache, 'common_inputs', return_value=self.cache.common.copy()):
            self.header.write_text('changed during validation')
            with self.assertRaisesRegex(RuntimeError, 'inputs changed during build'):
                self.cache.verify_unchanged([self.spec], keys)
        with patch.object(self.cache, 'common_inputs', return_value={'compiler': 'new'}):
            with self.assertRaisesRegex(RuntimeError, 'toolchain changed'):
                self.cache.verify_unchanged([self.spec], keys)


if __name__ == '__main__':
    unittest.main()
