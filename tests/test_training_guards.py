"""
Guards against silently-NaN training runs.

Before these fixes, `train_lolo_cv` wrapped every train and val batch in a bare
`except Exception: continue` with no logging. A ModuleNotFoundError inside
Option A's forward() therefore killed every batch, leaving `val_preds` empty;
`regression_metrics([], [])` returns NaN; and the run completed all 9 folds
reporting NaN as though it were a result.
"""
from __future__ import annotations

import inspect
import re

import pytest

pytest.importorskip("torch")


def test_regression_metrics_on_empty_input_is_nan_not_zero():
    """
    Documents the hazard. NaN is the correct output for no data — the bug was
    ever reaching this call with an empty list.
    """
    from Graph_model.train.metrics import regression_metrics

    m = regression_metrics([], [])
    assert m["rmse"] != m["rmse"], "empty metrics must be NaN, never 0.0"


def test_lolo_loop_does_not_swallow_batch_errors_silently():
    """Every `except` in train_lolo_cv must log or raise — never bare-continue."""
    from Graph_model.train import run_training

    src = inspect.getsource(run_training.train_lolo_cv)
    # a bare handler whose entire body is `continue`
    bare = re.findall(r"except\s+Exception\s*:\s*\n\s*continue\b", src)
    assert not bare, (
        f"train_lolo_cv still contains {len(bare)} bare "
        f"`except Exception: continue` handler(s) — batch failures would be "
        f"invisible and the fold would report NaN.")


def test_lolo_loop_refuses_to_report_when_all_batches_fail():
    """The loop must raise if nothing trained, rather than emit NaN."""
    from Graph_model.train import run_training

    src = inspect.getsource(run_training.train_lolo_cv)
    assert "if nb == 0:" in src and "RuntimeError" in src, (
        "train_lolo_cv must raise when every training batch fails")
    assert "if not val_preds:" in src, (
        "train_lolo_cv must raise when every validation batch fails")


def test_option_a_condition_encoder_import_resolves():
    """
    _encode_raw imports ConditionEncoder lazily inside forward(). A wrong path
    there is invisible until runtime — and was, for options A/B/C/D.
    """
    from Graph_model.model.option_a import _encode_raw

    # only inspect real import statements — comments legitimately mention the
    # bad path when explaining the historical bug
    bad = [ln.strip() for ln in inspect.getsource(_encode_raw).splitlines()
           if ln.lstrip().startswith(("import ", "from "))
           and "Graph_model.data.features" in ln]
    assert not bad, (
        f"Graph_model.data.features does not exist — use Graph_model.features: {bad}")

    # prove it actually imports
    from Graph_model.features.conditions import ConditionEncoder
    assert ConditionEncoder is not None


def test_no_module_references_the_nonexistent_data_features_package():
    """`Graph_model.data.features` has never existed. Catch new ones early."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "Graph_model"
    offenders = []
    for py in root.rglob("*.py"):
        for i, line in enumerate(py.read_text().splitlines(), 1):
            if "Graph_model.data.features" in line or "from .features" in line:
                # docstring header lines are cosmetic; imports are not
                if line.lstrip().startswith(("import ", "from ")):
                    offenders.append(f"{py.relative_to(root)}:{i}: {line.strip()}")
    assert not offenders, "broken imports:\n" + "\n".join(offenders)
