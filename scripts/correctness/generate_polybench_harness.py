#!/usr/bin/env python3
"""Create a PolyBench driver that declares, but does not define, a kernel.

The generated file preserves the source's allocation, initialization, timing,
dump, and cleanup code.  Only the selected function body is replaced by an
external declaration, so a compiler-produced kernel object can be linked
without symbol interposition or weak-symbol replacement.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def _masked(text: str) -> str:
    """Mask comments and literals while retaining byte offsets and newlines."""
    chars = list(text)
    index = 0
    state = "code"
    while index < len(chars):
        char = chars[index]
        following = chars[index + 1] if index + 1 < len(chars) else ""
        if state == "code":
            if char == "/" and following == "/":
                chars[index] = chars[index + 1] = " "
                index += 2
                state = "line_comment"
                continue
            if char == "/" and following == "*":
                chars[index] = chars[index + 1] = " "
                index += 2
                state = "block_comment"
                continue
            if char == '"':
                chars[index] = " "
                state = "string"
            elif char == "'":
                chars[index] = " "
                state = "char"
        elif state == "line_comment":
            if char == "\n":
                state = "code"
            else:
                chars[index] = " "
        elif state == "block_comment":
            if char == "*" and following == "/":
                chars[index] = chars[index + 1] = " "
                index += 2
                state = "code"
                continue
            if char != "\n":
                chars[index] = " "
        else:
            if char == "\\" and following:
                if char != "\n":
                    chars[index] = " "
                if following != "\n":
                    chars[index + 1] = " "
                index += 2
                continue
            if (state == "string" and char == '"') or (
                    state == "char" and char == "'"):
                chars[index] = " "
                state = "code"
            elif char != "\n":
                chars[index] = " "
        index += 1
    return "".join(chars)


def _matching_delimiter(masked: str, opening: int,
                        open_char: str, close_char: str) -> int:
    depth = 0
    for index in range(opening, len(masked)):
        if masked[index] == open_char:
            depth += 1
        elif masked[index] == close_char:
            depth -= 1
            if depth == 0:
                return index
    raise ValueError(f"unterminated {open_char}{close_char} group")


def generate_harness(text: str, function: str) -> str:
    masked = _masked(text)
    pattern = re.compile(
        rf"(?m)^(?P<indent>[ \t]*)(?:(?:static)[ \t\r\n]+)?"
        rf"(?P<return>void[ \t\r\n]+{re.escape(function)}[ \t]*\()"
    )
    matches = list(pattern.finditer(masked))
    definitions: list[tuple[re.Match[str], int, int]] = []
    for match in matches:
        opening_paren = masked.find("(", match.start("return"), match.end())
        closing_paren = _matching_delimiter(masked, opening_paren, "(", ")")
        body = closing_paren + 1
        while body < len(masked) and masked[body].isspace():
            body += 1
        if body < len(masked) and masked[body] == "{":
            definitions.append((
                match, closing_paren,
                _matching_delimiter(masked, body, "{", "}"),
            ))
    if len(definitions) != 1:
        raise ValueError(
            f"expected one definition of {function}, found {len(definitions)}")
    match, closing_paren, body_end = definitions[0]
    declaration = text[match.start("return"):closing_paren + 1].rstrip() + ";"
    return text[:match.start()] + match.group("indent") + declaration + text[body_end + 1:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--function", required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generate_harness(
        args.source.read_text(), args.function))


if __name__ == "__main__":
    main()
