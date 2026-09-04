"""Tests that the sweep files address real configuration.

A sweep is a YAML file the code never imports, so nothing normally checks that its keys
exist or its values are legal. Both failure modes are silent: a misplaced block nests
under ``early_terminate`` and is simply never swept, and a mistyped key sets an
attribute nobody reads. Each wastes a whole sweep before anyone notices.
"""

from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest
import yaml

from cypshift.experiment import ExperimentConfig

SWEEPS = sorted(Path("configs").glob("sweep*.yaml"))


def resolve(config: object, dotted: str) -> object:
    """Walk a dotted path through the config dataclasses, or raise."""
    current = config
    for part in dotted.split("."):
        if not is_dataclass(current) or part not in {f.name for f in fields(current)}:
            raise AttributeError(f"{dotted} stops at {part!r}")
        current = getattr(current, part)
    return current


def test_there_is_at_least_one_sweep():
    assert SWEEPS


@pytest.mark.parametrize("path", SWEEPS, ids=lambda p: p.name)
class TestSweepFile:
    def test_is_valid_yaml_with_the_required_blocks(self, path: Path):
        spec = yaml.safe_load(path.read_text())
        assert {"program", "method", "metric", "parameters"} <= set(spec)

    def test_early_termination_is_a_sibling_not_a_parameter(self, path: Path):
        """The bug this file exists for: an appended block nested under
        early_terminate is accepted by YAML and never swept."""
        spec = yaml.safe_load(path.read_text())
        assert set(spec.get("early_terminate", {})) <= {"type", "min_iter", "eta", "max_iter"}

    def test_every_parameter_names_a_real_config_field(self, path: Path):
        spec = yaml.safe_load(path.read_text())
        config = ExperimentConfig()
        for dotted in spec["parameters"]:
            resolve(config, dotted)

    def test_every_parameter_declares_how_it_varies(self, path: Path):
        spec = yaml.safe_load(path.read_text())
        for dotted, body in spec["parameters"].items():
            assert isinstance(body, dict), dotted
            assert {"values", "value", "distribution"} & set(body), dotted

    def test_the_optimised_metric_is_the_ranked_one(self, path: Path):
        """ST-RAE decides the leaderboard; optimising anything else is a different
        experiment."""
        spec = yaml.safe_load(path.read_text())
        assert "ST-RAE" in spec["metric"]["name"]
        assert spec["metric"]["goal"] == "minimize"

    def test_swept_values_pass_config_validation(self, path: Path):
        """Catches a legal-looking value the config would reject at run time."""
        spec = yaml.safe_load(path.read_text())
        for dotted, body in spec["parameters"].items():
            for value in body.get("values", [body["value"]] if "value" in body else []):
                config = ExperimentConfig()
                owner, _, leaf = dotted.rpartition(".")
                setattr(resolve(config, owner) if owner else config, leaf, value)
                config.validate()

    def test_the_grid_stays_legible(self, path: Path):
        """Staged sweeps exist to keep this small; one joint file reached 46,656."""
        spec = yaml.safe_load(path.read_text())
        grid = 1
        for body in spec["parameters"].values():
            grid *= len(body.get("values", [1]))
        assert grid <= 200, f"{path.name} has {grid} grid points; split it"
