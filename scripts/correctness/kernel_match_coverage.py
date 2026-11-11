#!/usr/bin/env python3
"""Measure application coverage using the same imported-specification pipeline."""
import argparse
import json
from pathlib import Path
import spec_match


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--backend', choices=['cpu', 'cuda', 'pva'], default='cpu')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in args.inputs:
        try:
            _, proof = spec_match.match(path.read_text(), backend=args.backend)
            rows.append(dict(input=str(path), status='matched' if proof['emitted'] else 'unmatched',
                             proof=proof))
        except (ValueError, OSError) as error:
            rows.append(dict(input=str(path), status='error', reason=str(error)))
    args.report.write_text(json.dumps(rows, indent=2) + '\n')
    print(f"{sum(row['status'] == 'matched' for row in rows)}/{len(rows)} applications matched; "
          f"{sum(row['status'] == 'error' for row in rows)} errors")
    return int(any(row['status'] == 'error' for row in rows))


if __name__ == '__main__':
    raise SystemExit(main())
