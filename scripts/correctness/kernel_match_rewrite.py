#!/usr/bin/env python3
"""Compatibility entry point for the specification-only matcher.

All recognition is performed by spec_match. The former handwritten region
recognizers are deliberately absent; unsupported computations remain in IR.
"""
import argparse
import json
import os
from pathlib import Path
import sys

from spec_match import SpecificationError, match


def rewrite_mlir(text, dry_run=False, roundtrip_markers=False,
                 show_candidates=False, show_semantic_only=False,
                 max_launches=None, disable_pointwise_matching=False,
                 disable_semantic_fallback=False, show_structured_regions=False,
                 enable_structured_rewrite=False, stencil_backend='cudnn',
                 pva_approximation_budgets=None, pva_histogram_output_type='u32',
                 disabled_kernels=None, only_kernels=None):
    if roundtrip_markers:
        raise SpecificationError('roundtrip markers retired; retain pre-match IR and the proof report')
    if pva_approximation_budgets:
        raise SpecificationError('a tolerance is not a specification-equivalence proof')
    output, proof = match(text, max_launches=max_launches,
                          backend=os.environ.get('POLYGEIST_ABI_BACKEND', 'cuda'),
                          disabled=disabled_kernels or (), only=only_kernels)
    report = [('match', [], name) for name in proof['emitted']]
    return output, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--disable-kernel', action='append', default=[])
    p.add_argument('--only-kernel', action='append')
    p.add_argument('--max-launches', type=int)
    p.add_argument('--telemetry-json', type=Path)
    p.add_argument('--report', type=Path)
    p.add_argument('--matcher-mode', choices=['egglog', 'syntactic'], default='egglog')
    p.add_argument('--pva-approximation-budget', action='append')
    p.add_argument('--pva-histogram-output-type', choices=['u32', 's32'], default='u32')
    p.add_argument('--stencil-backend', choices=['cudnn', 'custen'], default='cudnn')
    p.add_argument('--with-roundtrip-markers', action='store_true')
    for flag in ['enable-structured-rewrite', 'disable-pointwise-matching',
                 'disable-semantic-fallback', 'show-candidates', 'show-semantic-only',
                 'show-structured-regions', 'collect-egraph-sizes']:
        p.add_argument('--' + flag, action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    try:
        if args.pva_approximation_budget or args.with_roundtrip_markers:
            raise SpecificationError('legacy approximation/roundtrip overrides are unsupported')
        if args.matcher_mode == 'syntactic':
            os.environ['EQSAT_ABLATION'] = 'none'
        output, proof = match(args.input.read_text(),
                              backend=os.environ.get('POLYGEIST_ABI_BACKEND', 'cuda'),
                              disabled=args.disable_kernel, only=args.only_kernel,
                              max_launches=args.max_launches)
        for path in [args.telemetry_json, args.report]:
            if path:
                path.write_text(json.dumps(proof, indent=2) + '\n')
        for name in proof['emitted']:
            print(f'  match  imported specification {name}', file=sys.stderr)
        print(f"  total: {len(proof['emitted'])} proven library launch(es)", file=sys.stderr)
        if not args.dry_run:
            sys.stdout.write(output)
        return 0
    except (SpecificationError, ValueError, OSError) as error:
        print(f'kernel_match_rewrite: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
