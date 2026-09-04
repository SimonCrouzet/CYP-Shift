"""Optional Weights & Biases tracking.

Kept behind a flag and behind a lazy import so the package runs, and the tests run,
without wandb installed or a network connection. A run logs its full configuration as
flat dotted keys, which is the shape a sweep expects, and its metrics under a prefix
naming the split they were measured on.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from cypshift.experiment import ExperimentConfig

logger = logging.getLogger(__name__)


class Tracker:
    """Logs to Weights & Biases when enabled, and does nothing when not."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self._run: Any = None

    def start(self) -> "Tracker":
        if not self.config.tracking.enabled:
            logger.info("tracking disabled")
            return self
        try:
            import wandb
        except ImportError:
            logger.warning("tracking requested but wandb is not installed; continuing without it")
            return self
        self._run = wandb.init(
            project=self.config.tracking.project,
            entity=self.config.tracking.entity,
            tags=self.config.tracking.tags or None,
            config=self.config.to_flat_dict(),
        )
        logger.info("tracking run %s", getattr(self._run, "name", "?"))
        return self

    def log(self, metrics: dict[str, float], prefix: str = "") -> None:
        """Record metrics, optionally namespaced by the split they came from."""
        named = {f"{prefix}/{k}" if prefix else k: v for k, v in metrics.items()}
        logger.info("%s", " ".join(f"{k}={v:.4f}" for k, v in named.items()))
        if self._run is not None:
            self._run.log(named)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None


@contextmanager
def track(config: ExperimentConfig) -> Iterator[Tracker]:
    """Scope a tracked run, closing it even if the experiment raises."""
    tracker = Tracker(config).start()
    try:
        yield tracker
    finally:
        tracker.finish()
