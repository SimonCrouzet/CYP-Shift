"""The single-concentration primary screen, calibrated into usable pseudo-labels.

The dose-response labels are sparse and, worse, not missing at random: a compound was
promoted to a full curve only if it hit that isoform in the primary screen. Coverage
runs 26-48% per isoform and 3,596 of 4,905 compounds carry exactly one label. Train on
those alone and the model learns "given this compound is a hit, how big a hit", which
is not the question the blinded test set asks.

The screen is the correction. Every one of 4,376 compounds was measured against all
four enzymes at 49.5 uM, so the matrix is dense and unbiased, and it contains exactly
the inactives the curve set lacks. Its log2 fold-change ranks pIC50 at Spearman -0.83
to -0.94, so it carries real potency information despite being a single point.

What it cannot do is give a potency. A single concentration establishes a direction,
not an IC50, so the calibration here is deliberately monotone and non-parametric --
isotonic regression from fold-change to pIC50, fitted only on compounds that have
both. The team placing second in the preceding PXR challenge called this class of
transfer "the single most important data lever" in their solution.

**Extrapolated readings are returned as censored bounds, not as point estimates**,
and on one isoform that is the whole story rather than an edge case. A curve was only
fitted where the screen showed a hit, so the calibration is fitted over the *active*
part of the fold-change range and un-promoted compounds can sit outside it. Measured
here, the fraction of un-promoted compounds above the fitted range is 0.0-0.1% on
CYP1A2, CYP2C9 and CYP3A4 -- and **99% on CYP2D6**, whose curves only ever covered
fold-change below -0.59 while the un-promoted median is +0.38. Every CYP2D6
pseudo-label is therefore an extrapolation clipped to one endpoint. Treating those
2,883 clipped values as measurements would inject the same invented number 2,883
times; treated as one-sided bounds they say the one true thing the screen supports,
that the compound is inactive.

Bounds are emitted in the same ``_conf_low`` / ``_conf_high`` form the challenge uses,
so a pseudo-label drops straight into the soft-threshold loss the metric is built on.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from cypshift.config import ISOFORMS

logger = logging.getLogger(__name__)

SCREEN_CONCENTRATION_MOLAR = 4.95e-5
FOLD_CHANGE_COLUMN = "log2fc_estimate"


def to_wide(screen: pd.DataFrame, value: str = FOLD_CHANGE_COLUMN) -> pd.DataFrame:
    """Pivot the long screen table to one row per compound, one column per enzyme."""
    wide = screen.pivot_table(
        index=["Molecule_Name", "SMILES"], columns="enzyme", values=value, aggfunc="mean"
    ).reset_index()
    wide.columns.name = None
    return wide


class ScreenCalibrator:
    """Monotone map from screen fold-change to pIC50, fitted per isoform."""

    def __init__(self) -> None:
        self._models: dict[str, IsotonicRegression] = {}
        self._residual_std: dict[str, float] = {}
        self._fitted_range: dict[str, tuple[float, float]] = {}

    def fit(self, screen_wide: pd.DataFrame, inhibition: pd.DataFrame) -> "ScreenCalibrator":
        """Fit on compounds that have both a screen reading and a fitted curve.

        Args:
            screen_wide: Output of :func:`to_wide`.
            inhibition: The dose-response training set.

        Raises:
            ValueError: If an isoform has too few paired observations to fit.
        """
        merged = screen_wide.merge(
            inhibition.drop(columns=["SMILES"]), on="Molecule_Name", how="inner"
        )
        for isoform in ISOFORMS:
            paired = merged[[isoform, f"{isoform}_pIC50_direct_inhibition"]].dropna()
            if len(paired) < 50:
                raise ValueError(f"{isoform}: only {len(paired)} paired observations")
            fold_change = paired[isoform].to_numpy(float)
            pic50 = paired[f"{isoform}_pIC50_direct_inhibition"].to_numpy(float)
            # Inhibition drives fold-change down, so the map is decreasing.
            model = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(
                fold_change, pic50
            )
            self._models[isoform] = model
            self._residual_std[isoform] = float(np.std(pic50 - model.predict(fold_change)))
            self._fitted_range[isoform] = (float(fold_change.min()), float(fold_change.max()))
            logger.info(
                "%s: calibrated on %d compounds, residual sd %.2f",
                isoform,
                len(paired),
                self._residual_std[isoform],
            )
        return self

    def residual_std(self, isoform: str) -> float:
        """Spread of the calibration residual, the honest noise on a pseudo-label."""
        return self._residual_std[isoform]

    def fitted_range(self, isoform: str) -> tuple[float, float]:
        """Fold-change range the calibration was fitted over."""
        return self._fitted_range[isoform]

    def predict(self, screen_wide: pd.DataFrame, censored_floor: float = -2.0) -> pd.DataFrame:
        """Pseudo-pIC50 with credible bounds, from the screen alone.

        Args:
            screen_wide: Output of :func:`to_wide`.
            censored_floor: Lower bound assigned to a censored label, in pIC50. It
                only has to sit below anything the assay can resolve.

        Returns:
            Point estimates, ``_conf_low`` / ``_conf_high`` bounds, and an
            ``{isoform}_is_censored`` flag. A compound whose fold-change falls above
            the fitted range gets a one-sided bound running down from the calibration
            endpoint, because the screen establishes only that it is inactive.
        """
        if not self._models:
            raise RuntimeError("calibrator is not fitted")
        out = screen_wide[["Molecule_Name", "SMILES"]].copy()
        for isoform in ISOFORMS:
            measured = screen_wide[isoform].to_numpy(float)
            present = np.isfinite(measured)
            point = np.full(len(screen_wide), np.nan)
            point[present] = self._models[isoform].predict(measured[present])

            # Above the fitted fold-change range means less active than anything a
            # curve was ever fitted for, so the reading bounds the value rather than
            # estimating it.
            _, upper_fold_change = self._fitted_range[isoform]
            censored = present & (measured > upper_fold_change)
            spread = 1.96 * self._residual_std[isoform]

            column = f"{isoform}_pIC50_direct_inhibition"
            out[column] = point
            out[f"{column}_conf_low"] = np.where(censored, censored_floor, point - spread)
            out[f"{column}_conf_high"] = point + np.where(censored, 0.0, spread)
            out[f"{isoform}_is_censored"] = censored
            if censored.sum():
                logger.info(
                    "%s: %d of %d pseudo-labels are censored bounds, not estimates",
                    isoform,
                    int(censored.sum()),
                    int(present.sum()),
                )
        return out


def pseudo_labels(
    screen: pd.DataFrame,
    inhibition: pd.DataFrame,
    exclude_measured: bool = True,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Turn the screen into pseudo-labels for compounds lacking a real curve.

    Args:
        screen: The long single-concentration table.
        inhibition: The dose-response training set.
        exclude_measured: Drop a pseudo-label wherever a real measurement exists, so
            the two never compete for the same compound and isoform.

    Returns:
        The pseudo-label frame, and the calibration residual standard deviation per
        isoform -- which is what a pseudo-label should be weighted by, since it is
        several times noisier than a fitted curve.
    """
    wide = to_wide(screen)
    calibrator = ScreenCalibrator().fit(wide, inhibition)
    labels = calibrator.predict(wide)

    if exclude_measured:
        real = inhibition.set_index("Molecule_Name")
        for isoform in ISOFORMS:
            column = f"{isoform}_pIC50_direct_inhibition"
            has_real = labels["Molecule_Name"].map(real[column]).notna().to_numpy()
            labels.loc[has_real, column] = np.nan
        logger.info(
            "kept %d pseudo-labels not covered by a real curve",
            int(labels[[f"{i}_pIC50_direct_inhibition" for i in ISOFORMS]].notna().sum().sum()),
        )

    return labels, {isoform: calibrator.residual_std(isoform) for isoform in ISOFORMS}


def inhibited_fraction(log2_fold_change: np.ndarray) -> np.ndarray:
    """Convert the screen's fold-change readout into an inhibited fraction.

    The screen reports log2 of remaining activity against control, so a fold change of
    ``f`` leaves ``2**f`` of the signal and the inhibited fraction is ``1 - 2**f``.
    Values above zero mean the readout rose, which inhibition cannot explain, so they
    clip to zero rather than becoming a negative fraction.

    No calibration is involved. This is the measured quantity in the units a
    dose-response curve predicts at that concentration, which is what lets a screen
    reading constrain the curve directly instead of via a fitted pIC50 screen_model.
    """
    return np.clip(1.0 - np.power(2.0, log2_fold_change), 0.0, 1.0)


def response_observations(
    screen: pd.DataFrame,
    measurement_noise: float = 0.1,
    assay: str = "direct_inhibition_screen",
) -> pd.DataFrame:
    """Screen readings as constraints on the curve, one row per compound and enzyme.

    Args:
        screen: The long single-concentration table.
        measurement_noise: Half-width of the interval placed around each reading, as an
            inhibited fraction. A single well with no replicate does not support a
            tight bound.
        assay: The assay tag stamped on each row. The default feeds the curve model, which
            reads the screen as a point on the very curve it predicts; passing
            :data:`cypshift.data.assembly.SCREEN_RESPONSE_ASSAY` instead routes the same
            fractions to the graph's dedicated response head.

    Returns:
        Rows carrying the inhibited fraction and an interval around it, ready to be
        compared against the curve evaluated at the screen concentration.
    """
    rows = []
    for isoform in ISOFORMS:
        block = screen[screen["enzyme"] == isoform]
        if block.empty:
            continue
        fraction = inhibited_fraction(block[FOLD_CHANGE_COLUMN].to_numpy(float))
        rows.append(
            pd.DataFrame(
                {
                    "Molecule_Name": block["Molecule_Name"].to_numpy(),
                    "SMILES": block["SMILES"].to_numpy(),
                    "target": isoform,
                    "assay": assay,
                    "value": fraction,
                    "conf_low": np.clip(fraction - measurement_noise, 0.0, 1.0),
                    "conf_high": np.clip(fraction + measurement_noise, 0.0, 1.0),
                    "is_censored": False,
                    "data_space": "challenge",
                    "source": "screen",
                }
            )
        )
    columns = [
        "Molecule_Name",
        "SMILES",
        "target",
        "assay",
        "value",
        "conf_low",
        "conf_high",
        "is_censored",
        "data_space",
    ]
    if not rows:
        # No compound in this subset has a screen reading. An empty result is the
        # correct answer; concatenating nothing raises instead.
        logger.info("no screen readings for these compounds")
        return pd.DataFrame(columns=columns)
    observations = pd.concat(rows, ignore_index=True)
    logger.info(
        "screen as curve constraints: %d rows, median inhibited fraction %.2f",
        len(observations),
        float(observations["value"].median()),
    )
    return observations
