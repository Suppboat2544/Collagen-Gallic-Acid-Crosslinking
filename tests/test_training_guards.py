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

    # The features package lives at Graph_model/data/features. Only inspect
    # real import statements — comments legitimately mention the wrong path
    # when explaining the historical bug.
    bad = [ln.strip() for ln in inspect.getsource(_encode_raw).splitlines()
           if ln.lstrip().startswith(("import ", "from "))
           and "Graph_model.features" in ln]
    assert not bad, (
        f"Graph_model.features does not exist — the package is "
        f"Graph_model.data.features: {bad}")

    # prove it actually imports
    from Graph_model.data.features.conditions import ConditionEncoder
    assert ConditionEncoder is not None


def test_every_feature_import_targets_the_real_package():
    """
    The features package is Graph_model/data/features. On the north-review
    branch the directory sat at Graph_model/features, so imports written for
    the correct layout all failed. Upstream 69a6c14 restored the directory;
    this test stops the tree and the imports drifting apart again.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "Graph_model"
    assert (root / "data" / "features" / "__init__.py").is_file(), (
        "Graph_model/data/features/ is missing — the features package moved")
    assert not (root / "features").is_dir() or not list(
        (root / "features").glob("*.py")), (
        "a second features package exists at Graph_model/features/ — "
        "it will shadow or duplicate Graph_model/data/features/")

    offenders = []
    for py in root.rglob("*.py"):
        for i, line in enumerate(py.read_text().splitlines(), 1):
            if not line.lstrip().startswith(("import ", "from ")):
                continue
            if "Graph_model.features" in line:
                offenders.append(f"{py.relative_to(root)}:{i}: {line.strip()}")
    assert not offenders, (
        "imports pointing at the nonexistent Graph_model.features:\n"
        + "\n".join(offenders))
