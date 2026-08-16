"""
Graph_model.data — data loading, feature engineering, and dataset assembly.

Submodules are exposed lazily (PEP 562) for two reasons:

1. Circular imports. Graph_model.data.features.conditions needs
   Graph_model.data.config; eagerly importing .dataset here made that a cycle
   (data -> dataset -> features -> conditions -> data), which raised
   ImportError depending purely on which package was imported first.

2. Cost. `from Graph_model.data.config import LIGAND_CATALOGUE` should not
   require torch and torch_geometric to be installed. Config is plain data.

The public names below still work exactly as before -- they are just resolved
on first attribute access instead of at import time.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

# Config is dependency-free; import it eagerly so `Graph_model.data.config`
# is always available without pulling the ML stack.
from . import config  # noqa: F401

_LAZY = {
    "CollagenDockingDataset": ".dataset",
    "HeteroDockingDataset":   ".hetero_dataset",
    "StratifiedSplitter":     ".splitter",
    "AnchorLoader":           ".anchor",
}

__all__ = ["config", *_LAZY]


def __getattr__(name: str):
    """Resolve heavy submodule attributes on first access."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    module = importlib.import_module(target, __name__)
    value = getattr(module, name)
    globals()[name] = value          # cache, so this runs once
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:  # static analysers still see the real names
    from .dataset import CollagenDockingDataset
    from .hetero_dataset import HeteroDockingDataset
    from .splitter import StratifiedSplitter
    from .anchor import AnchorLoader
