"""Tests for auxiliary enzyme selection and distance weighting."""

import numpy as np
import pandas as pd
import pytest

from cypshift.data.auxiliary import (
    AUXILIARY_WEIGHTING,
    Auxiliary,
    contribution,
    distance_weight,
    select_auxiliaries,
    weight_observations,
)


@pytest.fixture
def distances():
    return pd.DataFrame(
        {
            "nearest": ["CYP2C9", "CYP3A4", "CYP1A2", "CYP2C9"],
            "distance": [0.5, 0.9, 1.3, 4.0],
        },
        index=pd.Index(["CYP2C19", "CYP3A5", "CYP1A1", "CYP2A6"], name="enzyme"),
    )


@pytest.mark.parametrize("scheme", AUXILIARY_WEIGHTING)
def test_weight_is_one_at_zero_distance(scheme):
    """An enzyme identical to the target should count fully, whatever the scheme."""
    assert distance_weight(0.0, scheme) == pytest.approx(1.0)


@pytest.mark.parametrize("scheme", ["inverse", "exponential"])
def test_weight_falls_with_distance(scheme):
    assert distance_weight(0.5, scheme) > distance_weight(2.0, scheme)


def test_uniform_ignores_distance():
    """The baseline that says distance weighting is doing nothing."""
    assert distance_weight(0.1, "uniform") == distance_weight(9.0, "uniform") == 1.0


def test_exponential_falls_faster_than_inverse():
    assert distance_weight(3.0, "exponential") < distance_weight(3.0, "inverse")


def test_exponential_reaches_one_over_e_at_the_scale():
    assert distance_weight(2.0, "exponential", scale=2.0) == pytest.approx(1 / np.e)


def test_inverse_halves_at_the_scale():
    assert distance_weight(2.0, "inverse", scale=2.0) == pytest.approx(0.5)


def test_weights_stay_positive():
    """A zero weight would silently drop an admitted enzyme."""
    assert all(distance_weight(9.0, s) > 0 for s in ("inverse", "exponential"))


def test_unknown_scheme_is_refused():
    with pytest.raises(ValueError, match="unknown weighting"):
        distance_weight(1.0, "gaussian")


def test_non_positive_scale_is_refused():
    with pytest.raises(ValueError, match="scale must be positive"):
        distance_weight(1.0, "exponential", scale=0.0)


def test_selection_admits_only_within_the_threshold(distances):
    admitted = select_auxiliaries(distances, max_distance=1.0)
    assert [a.name for a in admitted] == ["CYP2C19", "CYP3A5"]


def test_selection_can_admit_nothing(distances):
    """The correct answer for CYP2D6, and one that must surface rather than error."""
    assert select_auxiliaries(distances, max_distance=0.1) == []


def test_selection_is_ordered_closest_first(distances):
    admitted = select_auxiliaries(distances, max_distance=5.0)
    assert [a.distance for a in admitted] == sorted(a.distance for a in admitted)


def test_a_closer_enzyme_gets_more_weight(distances):
    admitted = select_auxiliaries(distances, max_distance=5.0)
    assert admitted[0].weight > admitted[-1].weight


def test_observations_take_their_enzyme_weight():
    observations = pd.DataFrame({"target": ["CYP2C9", "CYP2C19", "CYP2C19"]})
    auxiliaries = [Auxiliary("CYP2C19", "CYP2C9", 0.5, 0.6)]
    weights = weight_observations(observations, auxiliaries, scored=("CYP2C9",))
    assert list(weights) == [1.0, 0.6, 0.6]


def test_an_unadmitted_enzyme_gets_zero_weight():
    """A table may carry more than was selected; the selection decides what trains."""
    observations = pd.DataFrame({"target": ["CYP2C9", "CYP2A6"]})
    weights = weight_observations(
        observations, [Auxiliary("CYP2C19", "CYP2C9", 0.5, 0.6)], scored=("CYP2C9",)
    )
    assert list(weights) == [1.0, 0.0]


def test_target_weight_is_configurable():
    observations = pd.DataFrame({"target": ["CYP2C9", "CYP2C19"]})
    auxiliaries = [Auxiliary("CYP2C19", "CYP2C9", 0.5, 0.6)]
    weights = weight_observations(observations, auxiliaries, ("CYP2C9",), target_weight=3.0)
    assert list(weights) == [3.0, 0.6]


def test_an_enzyme_that_is_both_scored_and_auxiliary_is_refused():
    observations = pd.DataFrame({"target": ["CYP2C9"]})
    with pytest.raises(ValueError, match="both scored and auxiliary"):
        weight_observations(
            observations, [Auxiliary("CYP2C9", "CYP2C9", 0.0, 1.0)], scored=("CYP2C9",)
        )


def test_contribution_shows_an_auxiliary_outweighing_the_target():
    """Volume is the risk: downweighted rows can still dominate if there are enough."""
    observations = pd.DataFrame({"target": ["CYP2C9"] * 10 + ["CYP2C19"] * 200})
    auxiliaries = [Auxiliary("CYP2C19", "CYP2C9", 0.5, 0.6)]
    weights = weight_observations(observations, auxiliaries, scored=("CYP2C9",))
    shares = contribution(observations, weights)
    assert shares.loc["CYP2C19", "share"] > shares.loc["CYP2C9", "share"]
    assert shares["share"].sum() == pytest.approx(1.0)
