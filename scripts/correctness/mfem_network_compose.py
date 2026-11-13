#!/usr/bin/env python3
"""Compose matcher-labelled MFEM contractions into cuTensorNet networks.

The matcher output intentionally contains only `kernel.launch` references.
`polygeist-opt` needs matching `kernel.defn` symbols to verify those launches,
so this helper synthesizes temporary exact-signature definitions before running
the general network-composition pass and removes them from the result.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OPT = ROOT / "build" / "bin" / "polygeist-opt"
LAUNCH_RE = re.compile(r"\bkernel\.launch\s+@([A-Za-z0-9_.$-]+)")
DEFN_RE = re.compile(r"\bkernel\.defn(?:\s+private)?\s+@([A-Za-z0-9_.$-]+)")
NETWORK_RE = re.compile(r"\bkernel\.launch\s+@(cutensornetNetwork_[A-Za-z0-9_.$-]+)")
LAUNCH_SIGNATURE_RE = re.compile(
    r"kernel\.launch\s+@(?P<symbol>[A-Za-z0-9_.$-]+)"
    r"(?P<middle>\([^\n]*\)\s*(?:\{[^\n]*\})?\s*:\s*)"
    r"\((?P<inputs>[^\n]*)\)\s*->\s*(?P<result>[^\s]+)"
)


def _definition_blocks(text: str) -> dict[str, str]:
    """Return complete top-level kernel.defn blocks keyed by symbol."""
    blocks: dict[str, str] = {}
    for match in DEFN_RE.finditer(text):
        line_start = text.rfind("\n", 0, match.start()) + 1
        opening = text.find("{", match.end())
        if opening < 0:
            continue
        depth = 0
        in_string = False
        escaped = False
        closing = -1
        for index in range(opening, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    closing = index + 1
                    break
        if closing > 0:
            blocks[match.group(1)] = text[line_start:closing]
    return blocks


def _split_types(types: str) -> list[str]:
    result, start, depth = [], 0, 0
    for index, char in enumerate(types):
        if char in "<([{" :
            depth += 1
        elif char in ">)]}" :
            depth -= 1
        elif char == "," and depth == 0:
            result.append(types[start:index].strip())
            start = index + 1
    tail = types[start:].strip()
    if tail:
        result.append(tail)
    return result


def inject_referenced_definitions(matched_text: str) -> tuple[str, dict[str, str]]:
    """Add minimal exact-signature definitions required by kernel.launch.

    The canonical library bodies are not needed by the composition pass and
    can themselves carry a more specialized ABI than an old matcher artifact.
    Synthesizing identity bodies from the concrete launch signatures makes the
    temporary module verifiable without changing the published launch ABI.
    """
    signatures: dict[str, dict[tuple[str, str], str]] = {}
    aliases: dict[str, str] = {}
    existing = set(DEFN_RE.findall(matched_text))

    def rewrite(match: re.Match[str]) -> str:
        symbol = match.group("symbol")
        if symbol in existing:
            return match.group(0)
        signature = (match.group("inputs"), match.group("result"))
        variants = signatures.setdefault(symbol, {})
        if signature not in variants:
            suffix = "" if not variants else f"__compose_sig{len(variants)}"
            variants[signature] = symbol + suffix
        alias = variants[signature]
        aliases[alias] = symbol
        return ("kernel.launch @" + alias + match.group("middle") + "(" +
                match.group("inputs") + ") -> " + match.group("result"))

    rewritten = LAUNCH_SIGNATURE_RE.sub(rewrite, matched_text)
    if not signatures:
        return rewritten, aliases

    definitions = []
    for variants in signatures.values():
        for (inputs_text, result_type), alias in variants.items():
            input_types = _split_types(inputs_text)
            arguments = ", ".join(
                f"%arg{index}: {typ}" for index, typ in enumerate(input_types)
            )
            if result_type == "()":
                body = "    kernel.yield\n"
            else:
                try:
                    output_index = max(
                        index for index, typ in enumerate(input_types)
                        if typ == result_type
                    )
                except ValueError as error:
                    raise RuntimeError(
                        f"launch @{alias} has no operand matching result "
                        f"type {result_type}"
                    ) from error
                body = f"    kernel.yield %arg{output_index} : {result_type}\n"
            definitions.append(
                f"  kernel.defn @{alias}({arguments}) -> {result_type} {{\n"
                f"{body}  }}\n"
            )
    module = re.search(r"^module(?:\s+attributes\s+.*)?\s*\{\s*$", matched_text,
                       re.MULTILINE)
    if not module:
        raise RuntimeError("could not find the top-level MLIR module body")
    insertion = module.end()
    definition_text = "\n" + "".join(definitions) + "\n"
    return rewritten[:insertion] + definition_text + rewritten[insertion:], aliases


def strip_temporary_definitions(text: str, aliases: dict[str, str]) -> str:
    """Remove verifier-only defns and restore the original launch names."""
    blocks = _definition_blocks(text)
    for alias in aliases:
        block = blocks.get(alias)
        if block:
            text = text.replace(block, "")
    for alias, original in sorted(aliases.items(), key=lambda item: -len(item[0])):
        text = text.replace("@" + alias, "@" + original)
    return text


def compose_file(matched: Path, composed: Path, log: Path,
                 timeout: int = 300) -> tuple[int, str]:
    injected = composed.with_suffix(".with_defns.mlir")
    try:
        injected_text, aliases = inject_referenced_definitions(matched.read_text())
        injected.write_text(injected_text)
    except Exception as error:  # Preserve a useful per-kernel sweep result.
        message = f"definition injection failed: {error}\n"
        log.write_text(message)
        return 1, message
    try:
        proc = subprocess.run(
            [str(OPT), "--compose-cutensornet-networks",
             str(injected), "-o", str(composed)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=timeout,
        )
        message = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired as error:
        message = (error.stdout or "") + (error.stderr or "") + "\ntimeout\n"
        log.write_text(message)
        injected.unlink(missing_ok=True)
        return 124, message
    log.write_text(message)
    injected.unlink(missing_ok=True)
    if proc.returncode == 0:
        composed.write_text(strip_temporary_definitions(composed.read_text(), aliases))
    return proc.returncode, message


def launch_symbols(path: Path) -> list[str]:
    if not path.exists():
        return []
    return LAUNCH_RE.findall(path.read_text())


def network_symbols(path: Path) -> list[str]:
    if not path.exists():
        return []
    return NETWORK_RE.findall(path.read_text())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("matched", type=Path)
    parser.add_argument("composed", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument(
        "--inject-only", action="store_true",
        help="write verifier-ready exact-signature definitions without composing",
    )
    args = parser.parse_args()
    if args.inject_only:
        try:
            prepared, _ = inject_referenced_definitions(args.matched.read_text())
            args.composed.write_text(prepared)
        except Exception as error:
            raise SystemExit(f"definition injection failed: {error}")
        raise SystemExit(0)
    output_log = args.log or args.composed.with_suffix(".compose.log")
    raise SystemExit(compose_file(args.matched, args.composed, output_log)[0])
