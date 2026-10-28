"""Which quantities each auxiliary source can supply.

Every loader module declares ``QUANTITIES``. Validation, the pipeline and the sweep ask this
registry which sources supply a quantity, so adding a quantity to a loader needs no change
anywhere else, and no caller names a source to decide whether a quantity applies to it.
"""

import importlib

#: The sources with a loader, as :class:`cypshift.experiment.SourcesConfig` field names.
LOADER_SOURCES = ("chembl", "sair", "octant", "veith", "tox21", "novartis")

#: The quantities a source can be read as: the fitted potency, and the 0-1 inhibition at the
#: maximal curve response.
QUANTITY_NAMES = ("pic50", "max_response")


def quantities(source: str) -> tuple[str, ...]:
    """The quantities ``source``'s loader can return."""
    return importlib.import_module(f"cypshift.data.{source}").QUANTITIES


def supplying(quantity: str) -> tuple[str, ...]:
    """The sources whose loader can return ``quantity``."""
    return tuple(name for name in LOADER_SOURCES if quantity in quantities(name))
