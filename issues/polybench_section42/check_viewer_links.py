#!/usr/bin/env python3
"""Reject broken local href targets in the generated viewer."""

from __future__ import annotations

import html.parser
import sys
import urllib.parse
from pathlib import Path


class Links(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, _tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for key, value in attrs:
            if key == "href" and value:
                self.hrefs.append(value)


root = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ir_viewer").resolve()
broken: list[tuple[Path, str]] = []
checked = 0
for page in root.rglob("*.html"):
    parser = Links()
    parser.feed(page.read_text(errors="replace"))
    for href in parser.hrefs:
        parsed = urllib.parse.urlparse(href)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        target = (root / parsed.path.lstrip("/")) if parsed.path.startswith("/") else (page.parent / parsed.path)
        target = Path(urllib.parse.unquote(str(target)))
        checked += 1
        if not target.exists():
            broken.append((page.relative_to(root), href))
if broken:
    for page, href in broken:
        print(f"BROKEN {page}: {href}")
    raise SystemExit(1)
print(f"PASS: {checked} local links across {sum(1 for _ in root.rglob('*.html'))} HTML pages")
