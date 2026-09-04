"""Constants mirroring the challenge specification.

Values here must match the official evaluation code in
https://github.com/OpenADMET/CYP-Challenge-Tutorial (``evaluation/config.py``),
which the organisers state is ported from the live scoring backend.
"""

from typing import Final

ISOFORMS: Final = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")

REGRESSION_ENDPOINTS: Final = tuple(f"{iso}_pIC50_direct_inhibition" for iso in ISOFORMS)
CLASSIFICATION_ENDPOINTS: Final = ("CYP2D6_is_TDI", "CYP3A4_is_TDI")

CREDIBLE_UPPER_SUFFIX: Final = "_conf_high"
CREDIBLE_LOWER_SUFFIX: Final = "_conf_low"

IDENTIFIER_COLUMNS: Final = ("SMILES", "Molecule_Name")
TEST_SET_SIZE: Final = 750

MACRO_LABEL: Final = "MA"
BOOTSTRAP_SAMPLES: Final = 1000
BOOTSTRAP_SEED: Final = 0

#: Below this pIC50 the assay is outside its reliable dynamic range; the fitted
#: credible interval widens roughly tenfold and the metric stops penalising error.
ASSAY_FLOOR_PIC50: Final = 4.0

#: log10(2), the fold-shift threshold defining a time-dependent inhibitor.
TDI_SHIFT_THRESHOLD: Final = 0.301
