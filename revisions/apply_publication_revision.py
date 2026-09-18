"""Apply the requested, hash-checked manuscript edits atomically after validation."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import zlib

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = "f319b00dd9fb55419db557dcf59492d33cb8fc284d8784ecd979e63e96747e08"
ALLOWED = {
    "README.md", "code/bench_data.py", "docs/REPRODUCE.md", "paper/abstract.txt",
    "paper/emit_kernel_dnn.tex", "paper/paired_and_tail_results.tex",
    "paper/reflectance_stability.tex", "paper/residual_analysis.tex",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    encoded = "".join(
        "".join((ROOT / f"revisions/20260918-payload-{i}.b64").read_text().split())
        for i in (1, 2, 3)
    )
    # Normalize one identified transport insertion; the complete decoded payload
    # must still match the independently computed digest before it is accepted.
    encoded = encoded.replace("Fq4t2sa82bca82Ltqc", "Fq4t2sa82Ltqc")
    raw = zlib.decompress(base64.b64decode(encoded, validate=True))
    if sha(raw) != EXPECTED:
        raise RuntimeError("Staged payload digest mismatch; no files changed")
    records = json.loads(raw)
    if {r["path"] for r in records} != ALLOWED or len(records) != len(ALLOWED):
        raise RuntimeError("Unexpected or duplicate source paths")
    writes: list[tuple[Path, bytes]] = []
    for record in records:
        path = ROOT / record["path"]
        before = path.read_bytes()
        if sha(before) != record["before"]:
            raise RuntimeError(f"Concurrent source change in {record['path']}; no files changed")
        lines = before.decode("utf-8").splitlines(keepends=True)
        last_end = 0
        for start, end, replacement in record["edits"]:
            if not (last_end <= start <= end <= len(lines)):
                raise RuntimeError("Invalid edit offsets")
            if not isinstance(replacement, str):
                raise RuntimeError("Invalid replacement text")
            last_end = end
        for start, end, replacement in reversed(record["edits"]):
            lines[start:end] = replacement.splitlines(keepends=True)
        after = "".join(lines).encode("utf-8")
        if sha(after) != record["after"]:
            raise RuntimeError(f"Result digest mismatch in {record['path']}; no files changed")
        writes.append((path, after))
    # Validate every base and result before writing any of the requested files.
    for path, data in writes:
        path.write_bytes(data)
        print(f"Applied {path.relative_to(ROOT)} {sha(data)}")


if __name__ == "__main__":
    main()
