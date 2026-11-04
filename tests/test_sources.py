"""The source registry: which loaders supply which quantity."""

import pytest

from cypshift.data.sources import LOADER_SOURCES, QUANTITY_NAMES, quantities, supplying
from cypshift.experiment import ExperimentConfig


@pytest.mark.parametrize("source", LOADER_SOURCES)
def test_every_loader_declares_known_quantities_that_include_the_potency(source):
    declared = quantities(source)
    assert "pic50" in declared and set(declared) <= set(QUANTITY_NAMES)


def test_the_suppliers_of_a_max_response_are_the_loaders_that_declare_it():
    assert set(supplying("max_response")) == {
        name for name in LOADER_SOURCES if "max_response" in quantities(name)
    }
    assert set(supplying("pic50")) == set(LOADER_SOURCES)


def test_every_source_field_of_the_config_is_either_a_loader_or_a_named_non_loader():
    flags = {
        name for name, value in vars(ExperimentConfig().sources).items() if isinstance(value, bool)
    }
    assert set(LOADER_SOURCES) <= flags
