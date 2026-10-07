"""Content Fingerprints and Exclusive Local Research Artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def fingerprint(value: object) -> str:
    """Hash canonical JSON so Windows line endings do not change scientific settings."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_fingerprint() -> str:
    """Binding calibration and definitive runs to the same package source snapshot."""
    root = Path(__file__).resolve().parents[1]
    return fingerprint(
        {
            path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
            for path in sorted(root.rglob("*.py"))
        }
    )


def write_json(path: Path, payload: object) -> None:
    """Exclusive creation: existing output files are never replaced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}.")
    return value


def seal(payload: dict[str, Any]) -> dict[str, Any]:
    return {**payload, "content_sha256": fingerprint(payload)}


def read_sealed(path: Path) -> dict[str, Any]:
    value = read_json(path)
    digest = value.pop("content_sha256", None)
    if digest != fingerprint(value):
        raise ValueError(f"Integrity failure in {path.name}: content fingerprint changed.")
    return value


def local_path(root: Path, relative: str) -> Path:
    """Reject absolute paths, traversal, and symlink escape from a candidate directory."""
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("Candidate paths must remain within their private candidate directory.")
    return path
