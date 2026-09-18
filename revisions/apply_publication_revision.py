"""Apply the reviewed September 18 text revision once, without overwriting concurrent edits.

The three payload files encode an LZMA-compressed UTF-8 JSON line-edit manifest.
They contain manuscript/source text only, not executable serialization. Each file has
before/after SHA-256 values, and the complete manifest has an independent digest.
All paths and hashes are validated before any file is changed. Generated result tables
and the PDF are rebuilt separately, after tests. The one-time transport files are removed
from the final tree by the build job; the Git history retains this exact input.
"""
from __future__ import annotations

import base64
import hashlib
import json
import lzma
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SHA256 = "9866c918923edcba37fca6e0d8af7f7affb00d4f6ed45d22c619fa4b99facf15"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    encoded = "".join((ROOT / f"revisions/20260918-payload-{i}.b64").read_text().strip()
                      for i in (1, 2, 3))
    raw = lzma.decompress(base64.b64decode(encoded, validate=True), memlimit=256 * 1024**2)
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Revision manifest digest mismatch; no files changed")
    manifest = json.loads(raw)
    staged = []
    for name, change in manifest.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not (
            name.startswith(("paper/", "code/", "docs/"))
            or name in ("README.md", "requirements-revision.txt")
        ):
            raise ValueError(f"Unexpected target: {name}")
        old = path.read_text(encoding="utf-8") if path.exists() else ""
        current = sha(old) if path.exists() else None
        if current == change["after"]:
            print(f"Already applied: {name}")
            continue
        if current != change["before"]:
            raise ValueError(f"Concurrent edit or unexpected base: {name}; no files changed")
        lines = old.splitlines(keepends=True)
        last_end = 0
        for start, end, replacement in change["edits"]:
            if not (last_end <= start <= end <= len(lines)):
                raise ValueError(f"Invalid edit range: {name}")
            last_end = end
        for start, end, replacement in reversed(change["edits"]):
            lines[start:end] = replacement.splitlines(keepends=True)
        new = "".join(lines)
        if sha(new) != change["after"]:
            raise ValueError(f"Result digest mismatch: {name}; no files changed")
        staged.append((path, new))
    for path, text in staged:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"Applied: {path.relative_to(ROOT)}")
    print(f"Applied {len(staged)} text files; original result records were not changed.")


if __name__ == "__main__":
    main()
