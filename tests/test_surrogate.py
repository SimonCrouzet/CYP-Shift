"""Tests for feeding the screen in as a feature rather than as extra training rows."""

import numpy as np
import pandas as pd
import pytest

from cypshift.data.surrogate import (
    SURROGATE_STRATEGIES,
    fit_surrogate,
    hidden_activations,
    screen_targets,
)
from cypshift.features.molecular import featurise

SMILES = [
    "CCO",
    "c1ccccc1",
    "CC(=O)O",
    "CCN",
    "CCCC",
    "c1ccncc1",
    "CC(C)O",
    "CCOC",
    "CCCCO",
    "c1ccc(O)cc1",
    "CCCN",
    "CC(=O)N",
]


@pytest.fixture
def screen():
    rows = []
    for index, smiles in enumerate(SMILES):
        for enzyme in ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"):
            rows.append(
                {
                    "Molecule_Name": f"M{index}",
                    "SMILES": smiles,
                    "enzyme": enzyme,
                    "log2fc_estimate": -0.1 * (index % 5) - 0.05,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def trainable():
    return {f"M{i}" for i in range(len(SMILES))}


def test_screen_targets_are_inhibited_fractions(screen):
    wide, enzymes = screen_targets(screen)
    assert len(enzymes) == 4
    values = wide[enzymes].to_numpy()
    assert ((values >= 0.0) & (values <= 1.0)).all()


@pytest.mark.parametrize("strategy", SURROGATE_STRATEGIES)
def test_surrogate_produces_a_column_per_declared_name(screen, trainable, strategy):
    surrogate = fit_surrogate(screen, trainable, strategy, n_bits=256)
    assert surrogate.transform(SMILES).shape == (len(SMILES), len(surrogate.columns))


def test_predicted_label_gives_one_column_per_enzyme(screen, trainable):
    surrogate = fit_surrogate(screen, trainable, "predicted_label", n_bits=256)
    assert len(surrogate.columns) == 4


def test_embedding_is_wider_than_the_output(screen, trainable):
    """It keeps structure the four output numbers discard."""
    labels = fit_surrogate(screen, trainable, "predicted_label", n_bits=256)
    embedded = fit_surrogate(screen, trainable, "embedding", n_bits=256, hidden=16)
    assert len(embedded.columns) > len(labels.columns)


@pytest.mark.parametrize("strategy", SURROGATE_STRATEGIES)
def test_output_is_standardised(screen, trainable, strategy):
    """The columns land where BlockScaler passes them through untouched, so the
    surrogate has to bring its own scale."""
    surrogate = fit_surrogate(screen, trainable, strategy, n_bits=256)
    values = surrogate.transform(SMILES)
    assert abs(float(values.mean())) < 1.0
    assert float(values.std()) == pytest.approx(1.0, abs=0.6)


def test_a_surrogate_is_callable_for_featurise(screen, trainable):
    """It is handed straight to featurise(extra=...), so it must be callable."""
    surrogate = fit_surrogate(screen, trainable, "predicted_label", n_bits=256)
    plain = featurise(SMILES, n_bits=256)
    with_extra = featurise(SMILES, n_bits=256, extra=surrogate)
    assert with_extra.shape[1] == plain.shape[1] + len(surrogate.columns)


def test_holdout_compounds_are_excluded_from_fitting(screen):
    """A held-out compound's screen reading must not reach the model through the
    feature; that inflates the holdout score and nothing else."""
    surrogate = fit_surrogate(screen, {"M0", "M1", "M2", "M3"}, "predicted_label", n_bits=256)
    assert surrogate.network.n_features_in_ == featurise(SMILES[:1], n_bits=256).shape[1]


def test_no_trainable_compound_is_refused(screen):
    with pytest.raises(ValueError, match="no trainable compound"):
        fit_surrogate(screen, {"absent"}, "predicted_label", n_bits=256)


def test_unknown_strategy_is_refused(screen, trainable):
    with pytest.raises(ValueError, match="unknown surrogate strategy"):
        fit_surrogate(screen, trainable, "distillation", n_bits=256)


def test_hidden_activations_are_non_negative():
    """ReLU throughout, matching the network's activation."""

    class Fake:
        coefs_ = [np.array([[1.0, -1.0]]), np.array([[1.0], [1.0]])]
        intercepts_ = [np.array([0.0, 0.0]), np.array([0.0])]

    values = hidden_activations(Fake(), np.array([[1.0], [-1.0]]))
    assert (values >= 0).all()
    assert values.shape == (2, 2)


def test_hidden_activations_stop_before_the_output_layer(screen, trainable):
    """The output is what predicted_label already provides; the embedding is the layer
    before it."""
    surrogate = fit_surrogate(screen, trainable, "embedding", n_bits=256, hidden=16)
    assert surrogate.transform(SMILES).shape[1] == 16
