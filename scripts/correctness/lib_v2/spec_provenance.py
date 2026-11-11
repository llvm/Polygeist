"""Content-addressed record of the reference-to-MLIR import.

This establishes which inputs produced a specification, not equivalence of a
reference implementation to a proprietary vendor implementation.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


@lru_cache(maxsize=128)
def _hash(path, size, mtime_ns):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def file_hash(path):
    path = Path(path).resolve()
    st = path.stat()
    return _hash(str(path), st.st_size, st.st_mtime_ns)


def record(paths):
    return {str(Path(p).resolve().relative_to(ROOT)): file_hash(p) for p in paths}


def check(files):
    for name, expected in files.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or file_hash(path) != expected:
            return f'import input or artifact changed: {name}'
    return None


def write(path, specs, results, recipe, library, compiler_paths):
    here = Path(__file__).resolve().parent
    common = record([here / name for name in (
        'recipe.py', 'build_lib_v2.py', 'validate_specs.py', 'spec_provenance.py')]
                    + list(compiler_paths))
    rows = {}
    by_name = {r['spec']: r for r in results}
    for spec in specs:
        result = by_name[spec.name]
        prefix = path.parent / 'raised' / spec.name
        artifacts = [Path(str(prefix) + ext) for ext in ('.mlir', '_linalg.mlir', '_debuf.mlir')]
        rows[spec.name] = dict(
            status=result['status'], inputs=record([spec.fragment, spec.ref_c]),
            artifacts=record([p for p in artifacts if p.is_file()]),
            commands=result.get('commands', []),
            recoveries=result.get('recoveries', {}),
            numerical_validation=result.get('validation'),
        )
    path.write_text(json.dumps(dict(schema=1, recipe=recipe.name,
                                   common_inputs=common, library=record([library]),
                                   specifications=rows), indent=2) + '\n')


def validated_stages(row):
    validation = row.get('numerical_validation') or {}
    good = {'EXACT', 'WITHIN_1e-12'}
    return [stage for stage, key in [('cgeist-affine', 'affine'), ('debufferized', 'defn')]
            if validation.get(key) in good]


def validate(path):
    if not path.is_file():
        return {}, 'missing import manifest; rebuild the specification library'
    data = json.loads(path.read_text())
    if data.get('schema') != 1:
        return {}, 'unsupported import manifest schema'
    common_error = check(data['common_inputs']) or check(data['library'])
    errors = {}
    for name, row in data['specifications'].items():
        error = common_error or check(row['inputs']) or check(row['artifacts'])
        if not validated_stages(row):
            error = f"reference import has no validated semantic stage: {row['status']}"
        if not row['artifacts']:
            error = 'reference import has no generated MLIR'
        if error:
            errors[name] = error
    return errors, common_error
