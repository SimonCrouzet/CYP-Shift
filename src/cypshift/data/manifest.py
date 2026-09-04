"""Provenance records for downloaded data.

Every file fetched from a remote source is accompanied by a ``.manifest.json``
recording where it came from, when, and what it contained. Curation can then change
without re-querying the source, and a result can always be traced back to the exact
bytes it was computed from.
"""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

MANIFEST_SUFFIX = ".manifest.json"


@dataclass(frozen=True)
class Manifest:
    """What a downloaded file is, and where it came from."""

    filename: str
    source_url: str
    sha256: str
    n_bytes: int
    retrieved_at: str
    n_rows: int | None = None
    n_columns: int | None = None

    @property
    def path_for(self) -> str:
        return self.filename + MANIFEST_SUFFIX

    def write(self, directory: Path) -> Path:
        """Write the manifest beside its file and return the manifest path."""
        path = directory / self.path_for
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")
        return path

    @classmethod
    def read(cls, path: Path) -> "Manifest":
        return cls(**json.loads(path.read_text()))

    @classmethod
    def describe(
        cls,
        path: Path,
        source_url: str,
        n_rows: int | None = None,
        n_columns: int | None = None,
    ) -> "Manifest":
        """Build a manifest by hashing a file already on disk."""
        return cls(
            filename=path.name,
            source_url=source_url,
            sha256=sha256_of(path),
            n_bytes=path.stat().st_size,
            retrieved_at=datetime.now(UTC).isoformat(timespec="seconds"),
            n_rows=n_rows,
            n_columns=n_columns,
        )


def sha256_of(path: Path, chunk_bytes: int = 1 << 20) -> str:
    """Hex digest of a file, read in chunks so large files do not need to fit in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path) -> bool:
    """Whether a file still matches the digest recorded in its manifest."""
    manifest_path = path.with_name(path.name + MANIFEST_SUFFIX)
    if not manifest_path.exists():
        logger.warning("no manifest beside %s", path.name)
        return False
    recorded = Manifest.read(manifest_path).sha256
    actual = sha256_of(path)
    if recorded != actual:
        logger.error("%s has changed since download", path.name)
    return recorded == actual
