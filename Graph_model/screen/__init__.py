"""
Graph_model.screen
==================
Virtual screening pipeline for galloyl analogue discovery.

Workflow
--------
1. **Library generation** — enumerate 200-500 galloyl analogues via RDKit
   combinatorial chemistry (OH substitution, linker variation, ester patterns).
2. **Dual prediction** — predict ΔG_collagen and Selectivity Index (SI) for
   each candidate using a trained deep ensemble.
3. **Pareto optimisation** — identify the Pareto front in (ΔG, SI) space,
   apply an uncertainty filter (ensemble std > threshold → discard), and
   shortlist the top 5-10 novel structures.

Public API
----------
>>> from Graph_model.screen import (
...     GalloylLibrary,
...     screen_candidates,
...     pareto_front,
...     filter_by_uncertainty,
...     shortlist_candidates,
... )
"""

from .library_gen import GalloylLibrary
from .predict import screen_candidates
from .pareto import (
    pareto_front,
    filter_by_uncertainty,
    shortlist_candidates,
)

__all__ = [
    "GalloylLibrary",
    "screen_candidates",
    "pareto_front",
    "filter_by_uncertainty",
    "shortlist_candidates",
]
