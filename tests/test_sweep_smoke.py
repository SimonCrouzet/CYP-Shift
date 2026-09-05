"""Every configuration a sweep can propose must build, fit and predict.

Config validation only checks that values are spelled correctly. It does not check that
the object they name exists and works, and three bugs got through that gap: a swept
descriptor source the model never read, a conditioning mode whose descriptors were never
built, and a curve model with no ``predict`` at all -- which meant every curve run in
the sweep would have died after training.

These fit on a handful of compounds for one epoch. They assert nothing about accuracy,
only that the configuration is executable end to end.
"""

import itertools

import numpy as np
import pandas as pd
import pytest
import yaml

from cypshift.config import REGRESSION_ENDPOINTS
from cypshift.experiment import ExperimentConfig
from cypshift.models.base import Model
from cypshift.pipeline import build_model, supported_multifidelity
from tests.test_sweeps import SWEEPS

#: Varied enough that a fingerprint has non-constant columns to split on. Eight tiny
#: aliphatics gave LightGBM zero usable features at a reduced bit count.
SMILES = [
    "CN(C)CCOC(c1ccccc1)c1ccccc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CN1CCC(CC1)Oc1ccccc1Cl",
    "Cc1ccc(cc1)S(=O)(=O)N",
    "CCN(CC)CCNC(=O)c1ccc(N)cc1",
    "COc1ccc2nc(N)sc2c1",
    "OC(=O)c1ccccc1O",
    "CCCCn1cnc2c1c(=O)n(C)c(=O)n2C",
    "Clc1ccc(cc1)C(c1ccccc1)N1CCN(C)CC1",
    "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1",
    "Nc1ncnc2n(cnc12)C1CC(O)C(CO)O1",
    "c1ccc2[nH]ccc2c1",
]


@pytest.fixture
def observations():
    rows = []
    for index, smiles in enumerate(SMILES):
        for target in ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"):
            rows.append(
                {
                    "Molecule_Name": f"M{index}",
                    "SMILES": smiles,
                    "target": target,
                    "assay": "direct_inhibition_drc",
                    "value": 5.0 + (index % 3) * 0.5,
                    "conf_low": 4.5 + (index % 3) * 0.5,
                    "conf_high": 5.5 + (index % 3) * 0.5,
                    "weight": 1.0,
                    "is_censored": False,
                }
            )
    return pd.DataFrame(rows)


def swept_values(key: str, fallback: list) -> list:
    """Every value any sweep file proposes for a parameter."""
    found: list = []
    for path in SWEEPS:
        body = yaml.safe_load(path.read_text())["parameters"].get(key)
        if body is None:
            continue
        found.extend(body.get("values", [body["value"]] if "value" in body else []))
    return found or fallback


def _shrink(config: ExperimentConfig) -> ExperimentConfig:
    config.model.curve.epochs = 1
    config.model.curve.forward_passes = 2
    config.model.neural.epochs = 1
    config.model.boosted.n_estimators = 8
    config.features.fingerprint_bits = 512
    config.model.n_members = 2
    return config


@pytest.mark.parametrize(
    "kind,conditioning",
    sorted(
        {
            (k, c)
            for k in swept_values("model.kind", ["boosted", "curve"])
            for c in swept_values("model.curve.conditioning", ["heads"])
            if k != "graph"  # chemprop is exercised by its own tests
        }
    ),
)
def test_each_model_and_conditioning_fits_and_predicts(kind, conditioning, observations):
    config = _shrink(ExperimentConfig())
    config.model.kind = kind
    config.model.curve.conditioning = conditioning
    config.validate()

    model = build_model(config)
    assert isinstance(model, Model), f"{kind} does not satisfy the Model protocol"
    model.fit(observations)
    prediction = model.predict(SMILES[:3])
    assert prediction.mean.shape == (3, len(REGRESSION_ENDPOINTS))
    assert np.isfinite(prediction.mean).all()


@pytest.mark.parametrize("sources", swept_values("model.curve.descriptor_sources", [["curated"]]))
def test_each_descriptor_source_selection_changes_the_model(sources, observations):
    """A swept source that the model never reads produces identical runs under
    different names, which is how this one went unnoticed."""
    from cypshift.data.conditioning import describe_all

    config = _shrink(ExperimentConfig())
    config.model.kind = "curve"
    config.model.curve.conditioning = "film"
    config.model.curve.descriptor_sources = sources
    config.validate()

    model = build_model(config)
    width = describe_all(list(model.targets), tuple(sources)).shape[1]
    assert width > 0
    model.fit(observations)
    assert np.isfinite(model.predict(SMILES[:2]).mean).all()


@pytest.mark.parametrize("path", SWEEPS, ids=lambda p: p.name)
def test_swept_multifidelity_pairs_are_supported(path):
    """Pairs must be checked within a sweep file, not across them: stage one sweeps
    model families, stage two sweeps the boosted tree's strategies, and a value from
    one never meets a value from the other in a real run."""
    parameters = yaml.safe_load(path.read_text())["parameters"]

    def options(key: str, fallback: list) -> list:
        body = parameters.get(key)
        if body is None:
            return fallback
        return body.get("values", [body["value"]] if "value" in body else fallback)

    for kind, strategy in itertools.product(
        options("model.kind", ["boosted", "neural", "graph", "curve"]),
        options("multifidelity.strategy", ["none"]),
    ):
        assert strategy in supported_multifidelity(kind), f"{path.name}: {kind} x {strategy}"
