"""Append-only manifest for image generations.

One JSON object per line at `pipeline/manifest.jsonl`. Every successful
generation appends one record; QC retries append additional records
keyed by the same urlHash so the full retry history is auditable.

Required fields per spec:
  urlHash, prompt_hash, backend_id, model_revision, seed,
  timestamp, image_sha256, incidental_details, retry_count
Phase-A additions:
  qc_status (ok|manual_review|skipped),
  qc_clip_text_cosine,
  qc_vision_check
Phase-B additions (defaulted so old records still parse):
  mode ("text2image"|"edit"),
  source_image_sha256 (sha256 of the input image for edit mode, "" otherwise),
  source_provider (e.g. "existing-file", "backend:gemini")
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO / "pipeline" / "manifest.jsonl"


@dataclass
class ManifestRecord:
    urlHash: str
    prompt_hash: str
    backend_id: str
    model_revision: str
    seed: int | None
    timestamp: str
    image_sha256: str
    incidental_details: list[str]
    retry_count: int
    qc_status: str = "ok"
    qc_clip_text_cosine: float | None = None
    qc_vision_check: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    mode: str = "text2image"
    source_image_sha256: str = ""
    source_provider: str = ""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class ManifestWriter:
    """Append-only writer with one record per call.

    Not thread-safe. The driver is single-threaded by design.
    """

    def __init__(self, path: Path = MANIFEST_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: ManifestRecord) -> None:
        line = json.dumps(asdict(record), ensure_ascii=False)
        # Open per-call so an interrupted run still leaves the file
        # consistent up to the last successful flush. fsync isn't
        # strictly necessary (manifest is rebuildable from images),
        # but we want each record durable on disk before we count it.
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

    def latest_by_url_hash(self) -> dict[str, ManifestRecord]:
        """Read the manifest and return the most-recent record per
        urlHash. Used by `--resume` so we skip already-generated images."""
        if not self.path.exists():
            return {}
        out: dict[str, ManifestRecord] = {}
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                out[d["urlHash"]] = ManifestRecord(**d)
        return out
