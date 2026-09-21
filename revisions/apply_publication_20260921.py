"""Apply the reviewed source edits only to their exact original versions."""
from pathlib import Path
import base64
import hashlib
import json
import lzma

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = "61c20665bc309c1d69328918227731021479e98684c37d289650becaf7afc79f"
encoded = "".join((ROOT / f"revisions/publication-20260921-{i}.b64").read_text().strip() for i in range(1, 5))
raw = lzma.decompress(base64.b64decode(encoded, validate=True))
if hashlib.sha256(raw).hexdigest() != EXPECTED:
    raise SystemExit("Revision payload checksum mismatch")
payload = json.loads(raw)
pending = {}
for name, record in payload.items():
    path = ROOT / name
    if path.resolve().is_relative_to(ROOT) is False:
        raise SystemExit(f"Unsafe destination: {name}")
    if "new" in record:
        if path.exists():
            raise SystemExit(f"New destination already exists: {name}")
        result = record["new"].encode("utf-8")
    else:
        before = path.read_bytes()
        if hashlib.sha256(before).hexdigest() != record["old_sha256"]:
            raise SystemExit(f"Source changed since review; refusing overwrite: {name}")
        lines = before.decode("utf-8").splitlines(keepends=True)
        for start, stop, replacement in reversed(record["ops"]):
            lines[start:stop] = replacement.splitlines(keepends=True)
        result = "".join(lines).encode("utf-8")
    if hashlib.sha256(result).hexdigest() != record["sha256"]:
        raise SystemExit(f"Reconstructed file checksum mismatch: {name}")
    pending[path] = result
for path, result in pending.items():
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(result)
    print("Applied", path.relative_to(ROOT))
print(f"Applied {len(pending)} hash-checked source files.")
