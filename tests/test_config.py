"""The challenge specification constants must not drift from the official evaluation code."""

from cypshift import config


def test_regression_endpoints_match_official_names():
    assert config.REGRESSION_ENDPOINTS == (
        "CYP1A2_pIC50_direct_inhibition",
        "CYP2C9_pIC50_direct_inhibition",
        "CYP2D6_pIC50_direct_inhibition",
        "CYP3A4_pIC50_direct_inhibition",
    )


def test_classification_endpoints_cover_only_the_two_scored_isoforms():
    assert config.CLASSIFICATION_ENDPOINTS == ("CYP2D6_is_TDI", "CYP3A4_is_TDI")


def test_bootstrap_is_seeded_and_matches_the_official_repeat_count():
    assert config.BOOTSTRAP_SEED == 0
    assert config.BOOTSTRAP_SAMPLES == 1000
