"""The is_TDI decision heuristic shared by models that predict two arms.

Important: this is **our** heuristic, not the organisers' specification. The organisers
provide ``is_TDI`` labels directly and score predictions against them by MCC -- their
published ``evaluation/config.py`` defines only the ``CYP2D6_is_TDI`` / ``CYP3A4_is_TDI``
endpoints and names no shift threshold or assay floor. So these values are **not** a
challenge constant and do not belong in :mod:`cypshift.config`.

To *emit* an is_TDI call, a model that predicts a direct and a preincubated dose-response
arm compares them: we call a compound time-dependent on a >=2-fold IC50 shift between the
arms -- ``log10(2)``, the common literature definition of TDI -- provided its direct-arm
potency clears a minimum. Both are swept (``model.<family>.tdi_shift_threshold`` /
``tdi_assay_floor``), because the threshold that best recovers the provided labels from a
model's noisy predicted shifts need not equal whatever rule generated them.
"""

import math

#: Default arm shift to call is_TDI: a 2-fold IC50 shift, log10(2). Swept, not a spec value.
TDI_SHIFT_DEFAULT = math.log10(2.0)
#: Default minimum direct-arm pIC50 to consider an is_TDI call -- a weak-activity cutoff we
#: chose (IC50 ~100 uM), not an organiser constant. Swept.
TDI_FLOOR_DEFAULT = 4.0
