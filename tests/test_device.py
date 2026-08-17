"""
Device-selection tests (CPU / CUDA / MPS).

These run on any machine: the availability-dependent assertions branch on what
is actually present, so the suite is meaningful on a Mac, a CUDA box, or a
CPU-only CI runner.

Regression context: the package previously carried five separate device
resolution blocks that disagreed. `run_model_d_interpret.py` had no CUDA branch
at all and silently used CPU on NVIDIA hardware, and `train_main.resolve_device`
accepted any string, so `--device cuda` on a Mac returned a device that failed
later inside the training loop with an unrelated error.
"""
from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")

from Graph_model.train.device import (  # noqa: E402
    VALID_DEVICES, available_devices, cuda_available, describe, empty_cache,
    loader_kwargs, memory_report, mps_available, resolve_device,
    seed_everything, synchronize,
)


# ── Resolution ────────────────────────────────────────────────────────────────

def test_cpu_is_always_available():
    assert "cpu" in available_devices()
    assert resolve_device("cpu").type == "cpu"


def test_auto_never_raises_and_returns_an_available_backend():
    for spec in (None, "auto"):
        d = resolve_device(spec)
        assert d.type in available_devices()


def test_auto_prefers_an_accelerator_when_present():
    d = resolve_device("auto")
    if cuda_available():
        assert d.type == "cuda"
    elif mps_available():
        assert d.type == "mps"
    else:
        assert d.type == "cpu"


def test_unavailable_device_raises_rather_than_silently_using_cpu():
    """
    A silent downgrade turns a 10x slowdown into an invisible one. An explicit
    request for a missing backend must fail loudly.
    """
    missing = [d for d in ("cuda", "mps")
               if d not in available_devices()]
    if not missing:
        pytest.skip("every accelerator is available on this machine")
    with pytest.raises(RuntimeError, match="not available"):
        resolve_device(missing[0])


def test_unavailable_device_falls_back_when_not_strict():
    missing = [d for d in ("cuda", "mps") if d not in available_devices()]
    if not missing:
        pytest.skip("every accelerator is available on this machine")
    d = resolve_device(missing[0], strict=False)
    assert d.type in available_devices()


def test_unknown_device_name_is_a_value_error():
    with pytest.raises(ValueError, match="Unknown device"):
        resolve_device("tpu")


def test_accepts_a_torch_device_object_and_indexed_form():
    assert resolve_device(torch.device("cpu")).type == "cpu"
    assert resolve_device("cpu:0").type == "cpu"


def test_valid_devices_advertises_auto():
    assert set(VALID_DEVICES) == {"auto", "cpu", "cuda", "mps"}


# ── Backend helpers ───────────────────────────────────────────────────────────

def test_pin_memory_only_for_cuda():
    assert loader_kwargs(torch.device("cuda")).get("pin_memory") is True
    for d in ("cpu", "mps"):
        assert "pin_memory" not in loader_kwargs(torch.device(d))


def test_sync_and_empty_cache_are_safe_on_every_backend():
    """
    Called on cleanup paths each epoch; must never raise -- including for a
    backend this machine does not have, where torch.cuda.synchronize() would
    otherwise throw "Torch not compiled with CUDA enabled".
    """
    for name in ("cpu", "cuda", "mps"):
        for d in (name, torch.device(name)):
            synchronize(d)
            empty_cache(d)


def test_memory_report_never_raises_and_returns_text():
    for name in ("cpu", "cuda", "mps"):
        s = memory_report(torch.device(name))
        assert isinstance(s, str) and s.strip()


def test_describe_mentions_torch_version():
    assert torch.__version__ in describe()


# ── Reproducibility ───────────────────────────────────────────────────────────

def test_seed_everything_covers_random_numpy_and_torch():
    import random
    np = pytest.importorskip("numpy")

    def draw():
        seed_everything(1234)
        return (random.random(), float(np.random.rand()), float(torch.rand(1)))

    assert draw() == draw()


def test_seed_everything_accepts_deterministic_flag():
    seed_everything(0, torch.device("cpu"), deterministic=True)
    # restore non-deterministic mode so later tests are unaffected
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


# ── No duplicated resolvers ───────────────────────────────────────────────────

def test_all_entry_points_delegate_to_the_shared_resolver():
    """
    Guards the regression: every device decision must come from one place.
    A bare `torch.backends.mps.is_available()` branch outside device.py means
    a code path can disagree with the rest of the package again.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "Graph_model"
    offenders = []
    for py in root.rglob("*.py"):
        if py.name == "device.py":
            continue
        text = py.read_text()
        if "backends.mps.is_available" in text or "torch.cuda.is_available()" in text:
            for i, line in enumerate(text.splitlines(), 1):
                if ("backends.mps.is_available" in line
                        or "torch.cuda.is_available()" in line):
                    offenders.append(f"{py.relative_to(root)}:{i}: {line.strip()}")
    assert not offenders, (
        "device detection outside Graph_model/train/device.py:\n"
        + "\n".join(offenders))


def test_run_training_auto_device_delegates():
    from Graph_model.train import run_training

    src = inspect.getsource(run_training._auto_device)
    assert "resolve_device" in src, "_auto_device must delegate to device.py"
