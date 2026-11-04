"""The experiment runner must accept a W&B sweep's native argv, and the sweep configs
must invoke it correctly (F5)."""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, "scripts")
import run_experiment as rx  # noqa: E402


def test_parse_value_handles_booleans_numbers_and_strings():
    assert rx.parse_value("true") is True
    assert rx.parse_value("True") is True
    assert rx.parse_value("False") is False
    assert rx.parse_value("127") == 127
    assert rx.parse_value("0.03") == 0.03
    assert rx.parse_value("coarse_to_fine") == "coarse_to_fine"


def test_wandb_style_flags_build_the_config():
    """W&B passes --section.field=value, not --set section.field=value; both must work."""
    argv = [
        "--track",
        "--model.kind=boosted",
        "--sources.octant=true",
        "--model.boosted.num_leaves=127",
        "--training.preset=coarse_to_fine",
    ]
    parser = rx.argparse.ArgumentParser()
    parser.add_argument("--config", default=str(rx.DEFAULT_CONFIG_PATH))
    parser.add_argument("--set", nargs="*")
    parser.add_argument("--track", action="store_true")
    args, extras = parser.parse_known_args(argv)
    config = rx.build_config(args, extras)
    assert config.model.kind == "boosted"
    assert config.sources.octant is True
    assert config.model.boosted.num_leaves == 127
    assert config.training.preset == "coarse_to_fine"
    assert config.tracking.enabled is True


def test_every_sweep_config_invokes_the_runner_with_track():
    for path in sorted(Path("configs").glob("sweep*.yaml")):
        spec = yaml.safe_load(path.read_text())
        assert spec["program"] == "scripts/run_experiment.py"
        assert "--track" in spec["command"]
        assert "${args}" in spec["command"]
