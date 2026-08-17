"""
Graph_model.train.device
========================
Single source of truth for compute-device selection across CPU, CUDA and MPS.

Before this module there were five separate device-resolution blocks
(`run_training._auto_device`, `train_main.resolve_device`, `pretrain`,
`finetune`, `run_model_d_interpret`) which disagreed with each other — the
interpretation entry point had no CUDA branch at all and silently fell back to
CPU on an NVIDIA machine. All of them now delegate here.

Usage
-----
    from Graph_model.train.device import resolve_device, seed_everything

    device = resolve_device("auto")     # or "cpu" / "cuda" / "mps" / None
    seed_everything(0, device)

Backend notes
-------------
* **MPS** (Apple Silicon) does not support float64. Keep tensors float32.
  `torch.use_deterministic_algorithms(True)` is not fully supported either, so
  `seed_everything(..., deterministic=True)` degrades to warn-only there.
* **CUDA** benefits from `pin_memory=True` in DataLoaders; MPS and CPU do not.
  Use `loader_kwargs(device)`.
"""
from __future__ import annotations

import logging
import os
import random
from typing import Any

import torch

logger = logging.getLogger(__name__)

VALID_DEVICES = ("auto", "cpu", "cuda", "mps")


# ── Availability probes ───────────────────────────────────────────────────────

def cuda_available() -> bool:
    """True if a usable CUDA device is present."""
    try:
        return bool(torch.cuda.is_available())
    except Exception:          # pragma: no cover - defensive
        return False


def mps_available() -> bool:
    """
    True if Apple MPS is usable.

    Guarded with getattr because `torch.backends.mps` is absent on some older
    builds, where attribute access raises rather than returning False.
    """
    try:
        backend = getattr(torch.backends, "mps", None)
        if backend is None:
            return False
        return bool(backend.is_available())
    except Exception:          # pragma: no cover - defensive
        return False


def available_devices() -> list[str]:
    """Device strings usable on this machine, best first."""
    out = []
    if cuda_available():
        out.append("cuda")
    if mps_available():
        out.append("mps")
    out.append("cpu")
    return out


# ── Resolution ────────────────────────────────────────────────────────────────

def resolve_device(prefer: str | torch.device | None = None,
                   *, strict: bool = True) -> torch.device:
    """
    Resolve a compute device.

    Parameters
    ----------
    prefer : "auto" | "cpu" | "cuda" | "mps" | torch.device | None
        None and "auto" both auto-detect: CUDA -> MPS -> CPU.
    strict : bool
        If True (default) an explicit request for an unavailable backend
        raises RuntimeError. Silently downgrading a `--device cuda` request to
        CPU turns a 12x slowdown into an invisible one, so this is deliberate.
        Pass strict=False to fall back with a warning instead.

    Returns
    -------
    torch.device
    """
    if isinstance(prefer, torch.device):
        prefer = prefer.type

    if prefer is None or prefer == "auto":
        for name in ("cuda", "mps"):
            if (name == "cuda" and cuda_available()) or (name == "mps" and mps_available()):
                logger.info("Auto-selected device: %s", name)
                return torch.device(name)
        logger.info("Auto-selected device: cpu")
        return torch.device("cpu")

    prefer = str(prefer).lower().split(":")[0]
    if prefer not in VALID_DEVICES:
        raise ValueError(
            f"Unknown device {prefer!r}. Choose from {', '.join(VALID_DEVICES)}."
        )

    ok = {"cpu": True, "cuda": cuda_available(), "mps": mps_available()}[prefer]
    if ok:
        return torch.device(prefer)

    msg = (f"Requested device {prefer!r} is not available on this machine. "
           f"Available: {', '.join(available_devices())}.")
    if prefer == "cuda":
        msg += " (No CUDA build/GPU detected.)"
    if prefer == "mps":
        msg += " (MPS requires Apple Silicon and a macOS-capable torch build.)"

    if strict:
        raise RuntimeError(msg + " Pass --device auto to fall back automatically.")
    logger.warning("%s Falling back to auto-detection.", msg)
    return resolve_device("auto")


# ── Backend-specific helpers ──────────────────────────────────────────────────

def _as_device(device: str | torch.device) -> torch.device:
    """Accept either a torch.device or a plain string ('cpu', 'cuda:0', ...)."""
    return device if isinstance(device, torch.device) else torch.device(str(device))


def synchronize(device: str | torch.device) -> None:
    """
    Block until queued async work on `device` completes.

    A no-op for a backend that is not present: these run on cleanup paths, so
    naming an absent backend must not raise (torch.cuda.synchronize() otherwise
    throws "Torch not compiled with CUDA enabled").
    """
    device = _as_device(device)
    if device.type == "cuda" and cuda_available():
        torch.cuda.synchronize()
    elif device.type == "mps" and mps_available():
        torch.mps.synchronize()


def empty_cache(device: str | torch.device) -> None:
    """Release cached allocator blocks for `device`; no-op if unavailable."""
    device = _as_device(device)
    if device.type == "cuda" and cuda_available():
        torch.cuda.empty_cache()
    elif device.type == "mps" and mps_available():
        torch.mps.empty_cache()


def memory_report(device: str | torch.device) -> str:
    """
    Short human-readable memory string for progress lines.

    Previously only MPS reported device memory; CUDA runs showed host RSS only.
    """
    device = _as_device(device)
    try:
        import resource
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    except Exception:
        rss_mb = float("nan")

    try:
        if device.type == "mps":
            cur = torch.mps.current_allocated_memory() / (1024 ** 2)
            drv = torch.mps.driver_allocated_memory() / (1024 ** 2)
            return f"  rss={rss_mb:.0f}MB  mps={cur:.0f}/{drv:.0f}MB"
        if device.type == "cuda":
            cur = torch.cuda.memory_allocated() / (1024 ** 2)
            res = torch.cuda.memory_reserved() / (1024 ** 2)
            return f"  rss={rss_mb:.0f}MB  cuda={cur:.0f}/{res:.0f}MB"
    except Exception:          # pragma: no cover - reporting must never crash
        pass
    return f"  mem={rss_mb:.0f}MB"


def loader_kwargs(device: str | torch.device, num_workers: int = 0) -> dict[str, Any]:
    """
    DataLoader kwargs appropriate for `device`.

    pin_memory only helps host->CUDA copies; on MPS it is ignored or harmful.
    """
    device = _as_device(device)
    kw: dict[str, Any] = {"num_workers": num_workers}
    if device.type == "cuda":
        kw["pin_memory"] = True
        if num_workers > 0:
            kw["persistent_workers"] = True
    return kw


# ── Reproducibility ───────────────────────────────────────────────────────────

def seed_everything(seed: int,
                    device: torch.device | None = None,
                    *, deterministic: bool = False) -> None:
    """
    Seed every RNG that affects a run, on every backend.

    The library previously seeded only `torch`, leaving `random` and `numpy`
    free-running, so two runs with the same seed were not the same run.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    torch.manual_seed(seed)
    if cuda_available():
        torch.cuda.manual_seed_all(seed)
    if mps_available():
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass               # older torch: covered by torch.manual_seed

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as exc:
            logger.warning("Deterministic algorithms unavailable: %s", exc)
        if device is not None and device.type == "cuda":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def describe() -> str:
    """One-line environment summary for logs and result metadata."""
    parts = [f"torch={torch.__version__}"]
    if cuda_available():
        try:
            parts.append(f"cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")
        except Exception:
            parts.append("cuda=yes")
    parts.append(f"mps={'yes' if mps_available() else 'no'}")
    parts.append(f"available={'/'.join(available_devices())}")
    return "  ".join(parts)
