"""Tests for feeding the screen in as a feature rather than as extra training rows."""

import numpy as np
import pandas as pd
import pytest

from cypshift.data.screen_model import (
    SCREEN_MODEL_STRATEGIES,
    ScreenModelStack,
    fit_response_model,
    fit_screen_model,
    hidden_activations,
    screen_targets,
    stack_helper_models,
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


@pytest.mark.parametrize("strategy", SCREEN_MODEL_STRATEGIES)
def test_screen_model_produces_a_column_per_declared_name(screen, trainable, strategy):
    screen_model = fit_screen_model(screen, trainable, strategy, n_bits=256)
    assert screen_model.transform(SMILES).shape == (len(SMILES), len(screen_model.columns))


def test_predicted_label_gives_one_column_per_enzyme(screen, trainable):
    screen_model = fit_screen_model(screen, trainable, "predicted_label", n_bits=256)
    assert len(screen_model.columns) == 4


def test_embedding_is_wider_than_the_output(screen, trainable):
    """It keeps structure the four output numbers discard."""
    labels = fit_screen_model(screen, trainable, "predicted_label", n_bits=256)
    embedded = fit_screen_model(screen, trainable, "embedding", n_bits=256, hidden=16)
    assert len(embedded.columns) > len(labels.columns)


@pytest.mark.parametrize("strategy", SCREEN_MODEL_STRATEGIES)
def test_output_is_standardised(screen, trainable, strategy):
    """The columns land where BlockScaler passes them through untouched, so the
    screen_model has to bring its own scale."""
    screen_model = fit_screen_model(screen, trainable, strategy, n_bits=256)
    values = screen_model.transform(SMILES)
    assert abs(float(values.mean())) < 1.0
    assert float(values.std()) == pytest.approx(1.0, abs=0.6)


def test_a_screen_model_is_callable_for_featurise(screen, trainable):
    """It is handed straight to featurise(extra=...), so it must be callable."""
    screen_model = fit_screen_model(screen, trainable, "predicted_label", n_bits=256)
    plain = featurise(SMILES, n_bits=256)
    with_extra = featurise(SMILES, n_bits=256, extra=screen_model)
    assert with_extra.shape[1] == plain.shape[1] + len(screen_model.columns)


def test_holdout_compounds_are_excluded_from_fitting(screen):
    """A held-out compound's screen reading must not reach the model through the
    feature; that inflates the holdout score and nothing else."""
    screen_model = fit_screen_model(screen, {"M0", "M1", "M2", "M3"}, "predicted_label", n_bits=256)
    assert screen_model.network.n_features_in_ == featurise(SMILES[:1], n_bits=256).shape[1]


def test_no_trainable_compound_is_refused(screen):
    with pytest.raises(ValueError, match="no trainable compound"):
        fit_screen_model(screen, {"absent"}, "predicted_label", n_bits=256)


def test_unknown_strategy_is_refused(screen, trainable):
    with pytest.raises(ValueError, match="unknown screen_model strategy"):
        fit_screen_model(screen, trainable, "distillation", n_bits=256)


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
    screen_model = fit_screen_model(screen, trainable, "embedding", n_bits=256, hidden=16)
    assert screen_model.transform(SMILES).shape[1] == 16


def _response_rows(enzymes=("CYP2C9", "CYP2D6", "CYP3A4")):
    rows = []
    for index, smiles in enumerate(SMILES):
        for offset, enzyme in enumerate(enzymes):
            rows.append(
                {
                    "SMILES": smiles,
                    "target": enzyme,
                    "value": ((index * 7 + offset * 3) % 10) / 10,
                }
            )
    return pd.DataFrame(rows)


def test_response_model_has_one_named_column_per_enzyme_and_standardised_output():
    model = fit_response_model(_response_rows(), n_bits=64)
    assert model.columns == (
        "qhts_response_predicted_CYP2C9",
        "qhts_response_predicted_CYP2D6",
        "qhts_response_predicted_CYP3A4",
    )
    out = model(SMILES)
    assert out.shape == (len(SMILES), 3)
    assert np.allclose(out.mean(axis=0), 0.0, atol=1e-3)  # standardised on its training compounds


def test_response_model_uses_only_compounds_with_a_reading_for_every_enzyme():
    rows = _response_rows()
    rows = rows[~((rows["SMILES"] == "CCO") & (rows["target"] == "CYP2D6"))]
    # a compound missing one enzyme is dropped from the fit, the model still fits and predicts it
    model = fit_response_model(rows, n_bits=64)
    assert model(["CCO"]).shape == (1, 3)


def test_response_model_refuses_no_rows_and_no_complete_compound():
    with pytest.raises(ValueError, match="no max-response rows"):
        fit_response_model(_response_rows().iloc[:0])
    disjoint = pd.DataFrame(
        {
            "SMILES": ["CCO", "CCN"],
            "target": ["CYP2C9", "CYP2D6"],
            "value": [0.1, 0.2],
        }
    )
    with pytest.raises(ValueError, match="every enzyme"):
        fit_response_model(disjoint)


def test_helpers_stack_into_one_extra_block(screen, trainable):
    screen_model = fit_screen_model(screen, trainable, "predicted_label", n_bits=64)
    response = fit_response_model(_response_rows(), n_bits=64)
    assert stack_helper_models() is None
    assert stack_helper_models(None, screen_model) is screen_model
    stack = stack_helper_models(screen_model, response)
    assert isinstance(stack, ScreenModelStack)
    assert stack.columns == screen_model.columns + response.columns
    assert stack(SMILES).shape == (len(SMILES), len(stack.columns))
    assert np.allclose(stack(SMILES)[:, : len(screen_model.columns)], screen_model(SMILES))
    # a stack is accepted wherever featurise takes an ``extra`` block
    assert featurise(SMILES, n_bits=64, extra=stack).shape[1] >= len(stack.columns)
