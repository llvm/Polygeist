#!/usr/bin/env python3
"""Specification-only production matching and provenance.

Recognition is exclusively delegated to the imported-library Linalg and
region matchers. Backend capability filtering precedes matching. No operation
recognizer, benchmark identity, or ABI-only definition can authorize a launch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / 'lib_v2'))
import abi_v2 as ABI
import match_v2 as MATCH
import spec_provenance as PROVENANCE


class SpecificationError(ValueError):
    pass


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def registry():
    rows = {}
    import_path = MATCH.SPEC_DIR / 'import_manifest.json'
    import_errors, common_error = PROVENANCE.validate(import_path)
    imported = (json.loads(import_path.read_text()).get('specifications', {})
                if import_path.is_file() else {})
    for path in sorted(MATCH.SPEC_DIR.glob('manifest/*.json')):
        fragment = json.loads(path.read_text())
        for entry in fragment.get('specs', []):
            source = MATCH.SPEC_DIR / 'ref_c' / entry.get(
                'ref_c', fragment.get('ref_c', fragment.get('routine', path.stem) + '.c'))
            if entry['name'] in rows:
                raise SpecificationError(f"duplicate specification {entry['name']}")
            if not source.is_file() or not fragment.get('source'):
                raise SpecificationError(f"missing reference provenance for {entry['name']}")
            rows[entry['name']] = dict(
                entry=entry, manifest=str(path.relative_to(ROOT)),
                manifest_sha256=digest(path.read_text()),
                reference=str(source.relative_to(ROOT)),
                reference_sha256=digest(source.read_text()),
                origin=fragment['source'],
                import_error=(common_error or import_errors.get(entry['name']) or
                              ('no recorded import for this specification'
                               if entry['name'] not in imported else None)))
    return rows


def capabilities(rows, backend, disabled=(), only=None):
    _calls, bound, errors = ABI.build_table()
    allowed = set()
    rejected = {}
    for name, row in rows.items():
        target = row['entry'].get('target', {})
        call = target.get('call')
        library = target.get('library')
        if row.get('import_error'):
            rejected[name] = row['import_error']
        elif (semantic_configuration()['numeric_model'] == 'strict' and
              row['entry'].get('deviations')):
            rejected[name] = 'reference has documented deviations; strict matching requires an exact import'
        elif target.get('recognition_only'):
            rejected[name] = 'reference marked recognition-only'
        elif backend == 'pva' and library != 'PVA':
            rejected[name] = 'target library unavailable for requested backend'
        elif name in disabled or call in disabled:
            rejected[name] = 'disabled by explicit API/specification filter'
        elif only is not None and name not in only and call not in only:
            rejected[name] = 'outside explicit API/specification filter'
        elif name not in bound:
            rejected[name] = errors.get(name, 'no validated target ABI binding')
        else:
            allowed.add(name)
    return allowed, rejected


def semantic_inputs():
    paths = [HERE / name for name in (
        'spec_match.py', 'kernel_match.py', 'eqsat_ablation.py')]
    paths += sorted((HERE / 'lib_v2').glob('*.py'))
    paths += sorted((HERE / 'region_match').glob('*.py'))
    paths += [MATCH.SPEC_DIR / 'import_manifest.json']
    paths += [ROOT / 'runtime' / name for name in (
        'polygeist_v2_rt.h', 'polygeist_v2_rt_cpu.c', 'polygeist_v2_rt_cuda.c')]
    return PROVENANCE.record([path for path in paths if path.is_file()])


def abi_digest():
    # This includes the resolved vendor header types and Fortran prototypes,
    # even when their source headers live outside the repository.
    return digest(json.dumps(ABI.table_json(*ABI.build_table()), sort_keys=True))


def semantic_configuration():
    return dict(numeric_model=os.environ.get('POLYGEIST_NUMERIC_MODEL', 'strict'),
                eqsat_ablation=os.environ.get('EQSAT_ABLATION', 'full'),
                spec_suite=MATCH.SUITE)


def match(text, *, backend='cuda', disabled=(), only=None, max_launches=None,
          region_budget=60, region=True):
    if semantic_configuration()['numeric_model'] not in {'strict', 'relaxed'}:
        raise SpecificationError('unknown numerical model')
    if re.search(r'\bkernel\.launch\s+@', text):
        raise SpecificationError('input already contains library replacements; supply pre-match IR')
    if max_launches is not None and max_launches != 0:
        raise SpecificationError('partial launch limits are unsupported; filter specifications instead')
    rows = registry()
    allowed, rejected = capabilities(rows, backend, disabled, only)
    if max_launches == 0:
        allowed = set()
    started = time.monotonic()
    if allowed:
        output, matches, diagnostics = MATCH.run(
            text, MATCH.DEFAULT_LIB, region=region,
            region_budget=region_budget, allowed_specs=allowed)
    else:
        output, matches, diagnostics = text, [], {}
    emitted = re.findall(r'\bkernel\.launch\s+@([\w.$-]+)', output)
    if set(emitted) - allowed:
        raise SpecificationError('matcher emitted a call outside the validated specification registry')
    evidence = []
    for item in matches:
        name = item.get('spec') or item.get('region_match')
        if not name or item.get('recognition_only'):
            continue
        names = name.split('+')
        if any(n not in rows for n in names):
            continue  # diagnostic such as a timeout, not a match
        evidence.append(dict(
            engine='linalg' if 'spec' in item else 'region',
            specifications=names, binding=item.get('binding', {}),
            details=item,
            references=[{k: v for k, v in rows[n].items() if k != 'entry'} for n in names],
            targets=[rows[n]['entry']['target'] for n in names]))
    proved = {name for item in evidence for name in item['specifications']}
    if set(emitted) - proved:
        raise SpecificationError('emitted call lacks specification-match evidence')
    proof = dict(
        schema=2, pipeline='imported-specifications-only', backend=backend,
        semantic_inputs=semantic_inputs(), configuration=semantic_configuration(),
        abi_sha256=abi_digest(),
        input_sha256=digest(text), output_sha256=digest(output),
        library=str(MATCH.DEFAULT_LIB.relative_to(ROOT)),
        library_sha256=digest(MATCH.DEFAULT_LIB.read_text()),
        matcher_sha256=digest(Path(MATCH.__file__).read_text()),
        elapsed_ms=round((time.monotonic() - started) * 1000, 3),
        emitted=emitted, matches=evidence, rejected_specifications=rejected,
        diagnostics=diagnostics,
        validation_stage='recognition-and-ABI-binding; vendor execution not yet checked')
    return output, proof


def verify(text, proof, *, backend=None):
    if (proof.get('schema') != 2 or
            proof.get('pipeline') != 'imported-specifications-only' or
            proof.get('output_sha256') != digest(text)):
        raise SpecificationError('stored match evidence does not describe this IR')
    if backend is not None and proof.get('backend') != backend:
        raise SpecificationError('stored evidence uses a different target backend')
    if proof.get('configuration') != semantic_configuration():
        raise SpecificationError('stored evidence uses a different numerical model or ablation')
    if proof.get('semantic_inputs') != semantic_inputs():
        raise SpecificationError('stored evidence uses different matcher or import inputs')
    if proof.get('abi_sha256') != abi_digest():
        raise SpecificationError('stored evidence uses different vendor ABI declarations')
    if proof.get('library_sha256') != digest(MATCH.DEFAULT_LIB.read_text()):
        raise SpecificationError('stored match evidence uses a different specification library')
    if proof.get('matcher_sha256') != digest(Path(MATCH.__file__).read_text()):
        raise SpecificationError('stored match evidence uses a different matcher revision')
    for item in proof.get('matches', []):
        for row in item['references']:
            for key in ('manifest', 'reference'):
                if digest((ROOT / row[key]).read_text()) != row[key + '_sha256']:
                    raise SpecificationError('stored reference inputs have changed')
    emitted = re.findall(r'\bkernel\.launch\s+@([\w.$-]+)', text)
    allowed, _ = capabilities(registry(), proof['backend'])
    if set(emitted) - allowed:
        raise SpecificationError('stored launch no longer has a valid reference import and ABI')
    proved = {name for item in proof.get('matches', []) for name in item['specifications']}
    if emitted != proof.get('emitted') or set(emitted) - proved:
        raise SpecificationError('stored launch list is not justified by match evidence')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('-o', '--output', type=Path)
    p.add_argument('--report', type=Path)
    p.add_argument('--verify-report', type=Path)
    p.add_argument('--backend', choices=['cpu', 'cuda', 'pva'],
                   default=os.environ.get('POLYGEIST_ABI_BACKEND', 'cuda'))
    p.add_argument('--no-region-match', action='store_true')
    p.add_argument('--numeric-model', choices=['strict', 'relaxed'],
                   default=os.environ.get('POLYGEIST_NUMERIC_MODEL', 'strict'))
    p.add_argument('--region-budget', type=int, default=60)
    p.add_argument('--disable-kernel', action='append', default=[])
    p.add_argument('--only-kernel', action='append')
    p.add_argument('--max-launches', type=int)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)
    os.environ['POLYGEIST_NUMERIC_MODEL'] = args.numeric_model
    try:
        text = args.input.read_text()
        if args.verify_report:
            verify(text, json.loads(args.verify_report.read_text()), backend=args.backend)
            return 0
        output, proof = match(text, backend=args.backend,
                              disabled=args.disable_kernel, only=args.only_kernel,
                              max_launches=args.max_launches,
                              region_budget=args.region_budget,
                              region=not args.no_region_match)
        if args.report:
            args.report.write_text(json.dumps(proof, indent=2) + '\n')
        if not args.dry_run:
            if args.output:
                args.output.write_text(output)
            else:
                sys.stdout.write(output)
        print(f"spec_match: {len(proof['emitted'])} proven library launch(es)", file=sys.stderr)
        return 0
    except (SpecificationError, OSError, ValueError) as error:
        print(f'spec_match: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
