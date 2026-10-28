"""Per-machine settings and tokens, kept out of version control.

``local.machine.env`` at the repo root holds ``NAME=value`` lines (an optional ``export`` prefix,
``#`` comments and quoted values are accepted). :func:`load_local_env` copies them into
``os.environ`` for the entry points that need a token (the launcher, the sweep, the CLI), so the
libraries that read ``WANDB_API_KEY``, ``TABPFN_TOKEN``, ``HF_TOKEN`` or the Lightning SDK's
``LIGHTNING_*`` variables find them without an interactive login.

Rules that keep it safe to leave in a working tree:

* The file is gitignored, so ``git archive`` does not ship it; the launcher forwards chosen
  variables to a remote sweep explicitly.
* A variable already in the environment wins, so an explicit ``export`` overrides the file.
* An empty value is skipped, so a blank template entry never shadows a real one.
* Only variable *names* are returned or logged; a value never appears in a message, including the
  warning for a malformed line (which cites the line number, not its text).
"""

import logging
import os
import re
from collections.abc import MutableMapping
from pathlib import Path

logger = logging.getLogger(__name__)

LOCAL_ENV_FILE = "local.machine.env"
#: Overrides the file location, e.g. to keep tokens outside the checkout.
PATH_VARIABLE = "CYP_LOCAL_ENV"

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def default_path() -> Path:
    """``$CYP_LOCAL_ENV`` if set, else ``local.machine.env`` at the repo root."""
    override = os.environ.get(PATH_VARIABLE)
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parents[2] / LOCAL_ENV_FILE


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    # Unquoted values end at a ``#`` comment; a quoted one may contain it.
    return value.split(" #", 1)[0].rstrip()


def parse_local_env(text: str, source: str = LOCAL_ENV_FILE) -> dict[str, str]:
    """Parse ``NAME=value`` lines; blanks, comments and empty values are dropped.

    A line that is not ``NAME=value`` is skipped with a warning naming only its number.
    """
    values: dict[str, str] = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator or not _NAME.fullmatch(name):
            logger.warning("%s:%d is not NAME=value; ignored", source, number)
            continue
        value = _unquote(value)
        if value:
            values[name] = value
    return values


def load_local_env(
    path: Path | None = None, environ: MutableMapping[str, str] | None = None
) -> list[str]:
    """Load the file into ``environ`` (default ``os.environ``) without overriding what is set.

    Returns the names newly set, in file order. A missing file is not an error: every token is
    optional, and a machine that authenticates another way simply has no file.
    """
    target = os.environ if environ is None else environ
    path = default_path() if path is None else path
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    loaded = []
    for name, value in parse_local_env(text, source=path.name).items():
        if name not in target:
            target[name] = value
            loaded.append(name)
    return loaded
