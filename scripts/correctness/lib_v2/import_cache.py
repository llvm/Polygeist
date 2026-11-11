"""Reuse successful reference validation only when its full inputs still agree.

This is a local build cache, not a replacement for the import provenance record.
Failed or incomplete validation is always retried. Cached MLIR is checked before
reuse, and the ordinary full catalog/provenance writers still run afterwards.
"""
from __future__ import annotations

from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile

import spec_provenance as P


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def files(paths):
    # Keep the original path too: a changed symlink must invalidate its user.
    return {str(Path(p).absolute()): P.file_hash(p) for p in sorted(set(map(str, paths)))}


def command(args, **kwargs):
    result = subprocess.run(list(map(str, args)), capture_output=True, text=True,
                            timeout=60, **kwargs)
    if result.returncode:
        raise ValueError(f'cache dependency probe failed: {args[0]}: {result.stderr[-300:]}')
    return result.stdout


def shared_libraries(path):
    result = subprocess.run(['ldd', str(path)], capture_output=True, text=True, timeout=60)
    if result.returncode and not any(s in result.stdout + result.stderr for s in
                                     ('not a dynamic executable', 'statically linked')):
        raise ValueError(f'cannot identify runtime dependencies of {path}')
    if 'not found' in result.stdout:
        raise ValueError(f'missing runtime dependency of {path}')
    return re.findall(r'(?:=>\s+|^\s*)(/\S+)\s+\(', result.stdout, re.M)


class ValidatedImports:
    def __init__(self, builder, recipe):
        import e2e_v2 as E
        self.B, self.recipe, self.E = builder, recipe, E
        self.common = self.common_inputs()
        location = digest(str(builder.LIB.resolve()))[:16]
        self.directory = builder.REPO / 'build/reference-import-cache' / location

    def common_inputs(self):
        B, E = self.B, self.E
        here = Path(__file__).resolve().parent
        paths = [here / name for name in (
            'import_cache.py', 'build_lib_v2.py', 'spec_provenance.py', 'recipe.py',
            'validate_specs.py', 'e2e_v2.py', 'mlir_lite.py', 'legalize_tensor_submaps.py')]
        paths.append(here.parent / 'generate_polybench_harness.py')
        tools = [B.R.CGEIST, B.R.POLYGEIST_OPT, E.CLANG, E.MLIR_OPT,
                 E.MLIR_TRANSLATE, sys.executable,
                 E.MLIR_LIBDIR / 'libmlir_c_runner_utils.so']
        linker = command([E.CLANG, '-print-prog-name=ld']).strip()
        tools.append(shutil.which(linker) or linker)
        for name in ('crt1.o', 'crti.o', 'crtn.o', 'crtbegin.o', 'crtend.o',
                     'libm.so', 'libc.so', 'libgcc.a', 'libgcc_s.so'):
            path = command([E.CLANG, '-print-file-name=' + name]).strip()
            if not Path(path).is_file():
                raise ValueError(f'cannot identify validation link input {name}')
            paths.append(path)
        for tool in tools:
            paths.append(tool)
            paths.extend(shared_libraries(tool))
        # libm may be a linker script, so include its actual runtime library too.
        libm = command([E.CLANG, '-print-file-name=libm.so.6']).strip()
        paths.extend(shared_libraries(libm))
        paths.append(libm)
        env_keys = ('PATH', 'CPATH', 'C_INCLUDE_PATH', 'CPLUS_INCLUDE_PATH',
                    'LIBRARY_PATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD', 'COMPILER_PATH',
                    'GCC_EXEC_PREFIX', 'LANG', 'LC_ALL', 'SOURCE_DATE_EPOCH')
        return dict(schema=1, files=files(paths), recipe=asdict(self.recipe),
                    flags=B.R.COMMON_CFLAGS, python=sys.version,
                    platform=platform.platform(),
                    environment={k: os.environ.get(k) for k in env_keys})

    def inputs(self, spec):
        manifest = json.loads(spec.fragment.read_text())
        sources = [spec.fragment, spec.ref_c]
        upstreams = list(manifest.get('upstream_files', []))
        if manifest.get('upstream'):
            upstreams.append(manifest['upstream'])
        for upstream in upstreams:
            path = (self.B.REPO / upstream['path']).resolve()
            if not path.is_relative_to(self.B.REPO) or P.file_hash(path) != upstream['sha256']:
                raise ValueError(f'upstream reference changed: {spec.name}')
            sources.append(path)
        wrapper = f'#define NETLIB_SPEC {spec.name}\n'
        wrapper += ''.join(f'#define {k} {v}\n' for k, v in spec.defines.items())
        wrapper += f'#include "{spec.ref_c}"\n'
        # Preprocessing includes transitive headers, system headers, conditional
        # includes and their current search paths. Match frontend and validator
        # environments, including the frontend's static-exposure retry.
        variants = [[], ['-O3', '-fPIC', '-resource-dir=' + self.B.R.RESOURCE_DIR],
                    ['-O3', '-fPIC', '-Dstatic=', '-resource-dir=' + self.B.R.RESOURCE_DIR]]
        preprocessed = [command([self.E.CLANG, '-E', '-P', '-x', 'c', *flags, '-'],
                                input=wrapper) for flags in variants]
        return digest(dict(common=self.common, sources=files(sources),
                           preprocessing=preprocessed, spec=spec.raw))

    def artifacts(self, spec):
        prefix = self.B.RAISED / spec.name
        return files([Path(str(prefix) + suffix) for suffix in
                      ('.c', '.mlir', '_linalg.mlir', '_debuf.mlir', '.steps.json')])

    @staticmethod
    def reusable(result):
        return (isinstance(result, dict) and result.get('status') == 'RAISED'
                and bool(result.get('defn'))
                and isinstance(result.get('meta'), dict)
                and isinstance(result.get('validation'), dict)
                and all(result.get('validation', {}).get(stage) in ('EXACT', 'WITHIN_1e-12')
                        for stage in ('affine', 'linalg', 'defn')))

    def load(self, spec, key):
        try:
            saved = json.loads((self.directory / (spec.name + '.json')).read_text())
            payload = saved['payload']
            if saved['sha256'] != digest(payload) or payload['key'] != key:
                return None
            result = payload['result']
            if (not self.reusable(result) or result.get('spec') != spec.name
                    or payload['artifacts'] != self.artifacts(spec)):
                return None
            return result
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def save(self, spec, key, result):
        if not self.reusable(result):
            return
        payload = dict(key=key, artifacts=self.artifacts(spec), result=result)
        self.directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile('w', dir=self.directory, delete=False) as out:
            json.dump(dict(payload=payload, sha256=digest(payload)), out)
            temporary = out.name
        os.replace(temporary, self.directory / (spec.name + '.json'))

    def verify_unchanged(self, specs, keys, jobs=8):
        if self.common != self.common_inputs():
            raise RuntimeError('import toolchain changed during build; rerun before publishing')
        def verify(spec):
            if keys.get(spec.name) is not None and keys[spec.name] != self.inputs(spec):
                raise RuntimeError(f'import inputs changed during build: {spec.name}')
        with ThreadPoolExecutor(jobs) as pool:
            list(pool.map(verify, specs))
