#!/usr/bin/env python3
"""Select a validated IR alternative using measured end-to-end runtime cost.

Profiles must describe the same input, hardware, numerical model, residency,
warmup and timing protocol in `context`. Each contains `samples_ms`,
`correctness_passed`, and hashes of its IR, proof and executable. No benchmark
name, tensor shape or library identity participates in the decision. Originals
are retained; unknown, incorrect or unprofitable alternatives never replace them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics


def choose(baseline, candidate, minimum_improvement=0.02):
    if not 0 <= minimum_improvement < 1:
        raise ValueError('minimum improvement must be in [0, 1)')
    if baseline.get('correctness_passed') is not True:
        raise ValueError('baseline has not passed correctness')
    if not baseline.get('context') or baseline['context'] != candidate.get('context'):
        raise ValueError('candidate and baseline measurement contexts differ')
    summaries = {}
    for label, profile in [('baseline', baseline), ('candidate', candidate)]:
        samples = profile.get('samples_ms', [])
        if len(samples) < 5 or any(type(x) not in (int, float) or
                                  not math.isfinite(x) or x <= 0 for x in samples):
            raise ValueError(f'{label} needs at least five finite positive runtime samples')
        summaries[label] = dict(minimum_ms=min(samples),
                                median_ms=statistics.median(samples), samples=len(samples))
    ratio = summaries['candidate']['median_ms'] / summaries['baseline']['median_ms']
    accepted = candidate.get('correctness_passed') is True and ratio < 1 - minimum_improvement
    reason = ('candidate failed correctness' if candidate.get('correctness_passed') is not True else
              'measured improvement exceeds margin' if accepted else
              'measured improvement does not exceed margin')
    return dict(selected='candidate' if accepted else 'baseline', reason=reason,
                median_ratio=ratio, minimum_improvement=minimum_improvement,
                measurements=summaries, scope='this recorded input and measurement context')


def checked_profile(path):
    profile = json.loads(path.read_text())
    for kind in ['ir', 'proof', 'executable']:
        artifact = Path(profile[kind])
        if not artifact.is_absolute():
            artifact = path.parent / artifact
        actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if actual != profile[kind + '_sha256']:
            raise ValueError(f'{kind} changed after measurement: {artifact}')
        profile[kind] = str(artifact.resolve())
    proof = json.loads(Path(profile['proof']).read_text())
    if proof.get('output_sha256') != profile['ir_sha256']:
        raise ValueError('match proof does not describe measured IR')
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--minimum-improvement', type=float, default=0.02)
    args = parser.parse_args()
    baseline, candidate = map(checked_profile, [args.baseline, args.candidate])
    decision = choose(baseline, candidate, args.minimum_improvement)
    selected = candidate if decision['selected'] == 'candidate' else baseline
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(selected['ir'], args.output / 'selected.mlir')
    shutil.copyfile(selected['proof'], args.output / 'selected-proof.json')
    decision.update(baseline_profile=str(args.baseline.resolve()),
                    candidate_profile=str(args.candidate.resolve()))
    (args.output / 'selection.json').write_text(json.dumps(decision, indent=2) + '\n')
    print(json.dumps(decision))


if __name__ == '__main__':
    main()
