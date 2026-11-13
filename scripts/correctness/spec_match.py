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
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / 'lib_v2'))
import abi_v2 as ABI
import match_v2 as MATCH
import spec_provenance as PROVENANCE
import recipe as RECIPE


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
    calls, bound, errors = ABI.build_table()
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
        elif backend not in calls[bound[name].call].backends:
            rejected[name] = 'validated API adapter unavailable for requested backend'
        else:
            allowed.add(name)
    return allowed, rejected


def semantic_inputs():
    paths = [HERE / name for name in (
        'spec_match.py', 'kernel_match.py', 'eqsat_ablation.py',
        'polygeist_build.sh', 'build_jetson.sh')]
    paths += sorted((HERE / 'lib_v2').glob('*.py'))
    paths += sorted((HERE / 'region_match').glob('*.py'))
    paths += [MATCH.SPEC_DIR / 'import_manifest.json']
    paths += [ROOT / 'runtime' / name for name in (
        'polygeist_v2_rt.h', 'polygeist_v2_rt_cpu.c', 'polygeist_v2_rt_cuda.c',
        'polygeist_cublas_rt.h', 'polygeist_cublas_rt_cuda.c',
        'polygeist_v2_rt_pva.c', 'polygeist_pva_image_rt.c',
        'polygeist_cub_scan_rt.cu', 'polygeist_cuda_graph_rt.h',
        'polygeist_v2_matrix_layout.inc', 'polygeist_v2_memref_copy.inc',
        'polygeist_pointwise_partition.h', 'polygeist_pointwise_rounding.inc',
        'polygeist_pointwise_division.inc', 'polygeist_thrust_rt.cu')]
    return PROVENANCE.record([path for path in paths if path.is_file()])


def abi_digest():
    # This includes the resolved vendor header types and Fortran prototypes,
    # even when their source headers live outside the repository.
    return digest(json.dumps(ABI.table_json(*ABI.build_table()), sort_keys=True))


def semantic_configuration():
    return dict(numeric_model=os.environ.get('POLYGEIST_NUMERIC_MODEL', 'strict'),
                eqsat_ablation=os.environ.get('EQSAT_ABLATION', 'full'),
                spec_suite=MATCH.SUITE)


def restore_library_metadata(original, normalized):
    """Preserve the imported ABI metadata across an MLIR printer round-trip.

    SSA spelling is not semantic, but the imported metadata deliberately uses
    the C parameter names. Restore argument spellings by position and give all
    other SSA values fresh names so they cannot shadow those parameters.
    """
    before = {f.name: f for f in MATCH.ML.Module(original).funcs}
    edits = []
    for function in MATCH.ML.Module(normalized).funcs:
        old = before[function.name]
        if len(old.args) != len(function.args):
            raise SpecificationError('normalization changed an imported library ABI')
        rename = {new[0]: prior[0] for new, prior in zip(function.args, old.args)}
        prefix = '%normalized_'
        while any(name.startswith(prefix) for name, _ in old.args):
            prefix += '_'
        body = normalized[function.start:function.end]
        body = re.sub(r'%[\w$.\-]+',
                      lambda m: rename.get(m[0], prefix + m[0][1:]), body)
        edits.append((function.start, function.end, body))
    for start, end, body in reversed(edits):
        normalized = normalized[:start] + body + normalized[end:]
    metadata = re.findall(r'^\s*//\s*lib_v2-meta:.*$', original, re.MULTILINE)
    return '\n'.join(metadata) + '\n' + normalized


def normalize_matched_application(text, library, normalize):
    """Preserve already-proven calls as opaque typed calls during normalization.

    Region matches use memref ABIs and Linalg matches use tensor ABIs. Preserve
    each exact signature, without inventing a new semantic library definition.
    Opaque func.call has conservative effects, so normalization cannot delete
    or combine it. The existing proof, not these declarations, authorizes it.
    """
    prefix = '__polygeist_preserved_launch_'
    while '@' + prefix in text:
        prefix += '_'
    edits, declarations, names = [], [], {}
    for op in MATCH.ML.Module(text).ops:
        if op.name != 'kernel.launch':
            continue
        symbol = re.search(r'kernel\.launch\s+@([\w.$-]+)', op.text)[1]
        bridge = prefix + str(len(names))
        names[bridge] = symbol
        start = op.text.index('(', op.text.index('kernel.launch'))
        end = MATCH.ML.matching_paren(op.text, start)
        signature = op.text[end + 1:].strip().removeprefix(':').strip()
        declarations.append(f'  func.func private @{bridge}{signature}')
        changed = op.text.replace('kernel.launch @' + symbol, 'func.call @' + bridge, 1)
        edits.append((op.start, op.end, changed))
    for start, end, changed in reversed(edits):
        text = text[:start] + changed + text[end:]
    end = text.rfind('}')
    if end < 0:
        raise SpecificationError('application must be an MLIR module')
    text = text[:end] + '\n' + '\n'.join(declarations) + '\n' + text[end:]
    normalized = normalize(text)
    for bridge, symbol in names.items():
        call = r'(?<![\w.])(?:func\.)?call @' + re.escape(bridge) + r'\('
        if len(re.findall(call, normalized)) != 1:
            raise SpecificationError('normalization changed an existing library call')
        normalized = re.sub(call, 'kernel.launch @' + symbol + '(', normalized)
        normalized = re.sub(r'^\s*func\.func private @' + re.escape(bridge) +
                            r'\([^\n]*\n', '', normalized, flags=re.MULTILINE)
    return normalized


def match(text, *, backend='cuda', disabled=(), only=None, max_launches=None,
          region_budget=60, region=True, normalize_parallel=False, max_parallel_dims=0, max_operand_rank=0,
          fuse_initializers=False):
    if not isinstance(max_parallel_dims, int) or max_parallel_dims < 0:
        raise SpecificationError('parallel rank limit must be a nonnegative integer')
    if not isinstance(max_operand_rank, int) or max_operand_rank < 0:
        raise SpecificationError('operand rank limit must be a nonnegative integer')
    if not isinstance(fuse_initializers, bool):
        raise SpecificationError('initializer fusion must be a boolean')
    if fuse_initializers and not (max_parallel_dims or max_operand_rank):
        raise SpecificationError('initializer fusion requires a parallel tiling rank limit')
    normalize_parallel = normalize_parallel or max_parallel_dims > 0 or max_operand_rank > 0
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
    normalization = None
    if allowed:
        if normalize_parallel:
            # Both sides undergo the identical generic IR transformation.
            # This produces a candidate, not a claim of profitability. The
            # ordinary path remains available for measured candidate selection.
            option = '--normalize-linalg-matching-views'
            rank_options = []
            if max_parallel_dims:
                rank_options.append('max-parallel-dims=' + str(max_parallel_dims))
            if max_operand_rank:
                rank_options.append('max-operand-rank=' + str(max_operand_rank))
            if fuse_initializers:
                rank_options.append('fuse-initializers=true')
            if rank_options:
                option += '=' + ' '.join(rank_options)
            baseline, baseline_matches, baseline_diagnostics = MATCH.run(
                text, MATCH.DEFAULT_LIB, region=region,
                region_budget=region_budget, allowed_specs=allowed)
            def normalize(source):
                result = subprocess.run([RECIPE.POLYGEIST_OPT, option, '--cse'],
                                        input=source, text=True, capture_output=True,
                                        timeout=120)
                if result.returncode:
                    raise SpecificationError('view normalization failed: ' + result.stderr[-2000:])
                return result.stdout
            # Preserve matches already established by either shared matcher.
            # Normalize only their residual computations; a representation
            # change must not make an existing region match disappear.
            original_library = MATCH.DEFAULT_LIB.read_text()
            application = normalize_matched_application(baseline, original_library, normalize)
            library = restore_library_metadata(original_library, normalize(original_library))
            normalization = dict(
                pass_name=option,
                cleanup=['--cse'],
                compiler_sha256=PROVENANCE.file_hash(Path(RECIPE.POLYGEIST_OPT)),
                matched_baseline_sha256=digest(baseline),
                application_sha256=digest(application),
                library_sha256=digest(library))
            with tempfile.TemporaryDirectory(prefix='polygeist-matching-views-') as directory:
                path = Path(directory) / 'library.mlir'
                path.write_text(library)
                output, matches, diagnostics = MATCH.run(
                    application, path, region=False, region_budget=region_budget,
                    allowed_specs=allowed)
            matches = baseline_matches + matches
            diagnostics['baseline'] = baseline_diagnostics
        else:
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
    if normalization:
        proof['view_normalization'] = normalization
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
    normalization = proof.get('view_normalization')
    if normalization and (not re.fullmatch(
                          r'--normalize-linalg-matching-views(?:=(?:'
                          r'max-parallel-dims=[1-9][0-9]*(?: max-operand-rank=[1-9][0-9]*)?'
                          r'|max-operand-rank=[1-9][0-9]*)(?: fuse-initializers=true)?)?',
                          str(normalization.get('pass_name', ''))) or
                          normalization.get('cleanup') != ['--cse'] or
                          normalization.get('compiler_sha256') !=
                          PROVENANCE.file_hash(Path(RECIPE.POLYGEIST_OPT))):
        raise SpecificationError('stored evidence uses a different view normalization compiler')
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
    p.add_argument('--max-parallel-dims', type=int, default=0,
                   help='optional upstream tiling candidate with this parallel rank limit')
    p.add_argument('--max-operand-rank', type=int, default=0,
                   help='optional upstream slicing candidate with this operand rank limit')
    p.add_argument('--fuse-initializers', action='store_true',
                   help='use upstream scalar-initializer fusion inside parallel tiles')
    p.add_argument('--normalize-parallel-dims', action='store_true',
                   help='produce a dimension-collapse candidate using the same pass on both sides')
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
                              region=not args.no_region_match,
                              normalize_parallel=args.normalize_parallel_dims,
                              max_parallel_dims=args.max_parallel_dims,
                              max_operand_rank=args.max_operand_rank,
                              fuse_initializers=args.fuse_initializers)
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
