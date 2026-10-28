"""The nested postprocessor-validation entrypoint's data-root resolution."""

import sys
from pathlib import Path

from cypshift.experiment import ExperimentConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from postprocess_validation import resolve_directory  # noqa: E402


def test_resolve_directory_uses_the_config_when_no_cli_override():
    config = ExperimentConfig.load()
    config.override("data.directory", "/tmp/cfg-root")
    assert str(resolve_directory(config, None)) == "/tmp/cfg-root"


def test_cli_directory_overrides_and_syncs_the_config():
    """Review #5: --directory must write through to config.data.directory, so the labels/test
    loads and assemble_training_data's screen/TDI/auxiliary reads share one root."""
    config = ExperimentConfig.load()
    config.override("data.directory", "/tmp/cfg-root")
    resolved = resolve_directory(config, Path("/tmp/cli-root"))
    assert str(resolved) == "/tmp/cli-root"
    assert config.data.directory == "/tmp/cli-root"  # assemble() reads from here
