"""
Graph_model.screen.pareto
==========================
Pareto front identification, uncertainty filtering, and top-candidate
shortlisting for multi-objective virtual screening.

Objectives (minimise both):
    1. ΔG_collagen  — more negative = stronger collagen binding  (minimise)
    2. −SI          — higher SI = more selective for collagen     (minimise −SI)

Workflow
--------
1. ``pareto_front``          — identify non-dominated solutions
2. ``filter_by_uncertainty`` — discard compounds with ensemble std > threshold
3. ``shortlist_candidates``  — combine the above + diversity filter → top N

Usage
-----
>>> from Graph_model.screen.pareto import (
...     pareto_front, filter_by_uncertainty, shortlist_candidates)
>>> results = screen_candidates(ensemble, smiles_list)
>>> front = pareto_front(results)
>>> filtered = filter_by_uncertainty(results, max_std=0.5)
>>> top = shortlist_candidates(results, top_k=10, max_std=0.5)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Pareto front identification ──────────────────────────────────────────────

def pareto_front(
    results: list[dict],
    obj1_key: str = "delta_g_mean",
    obj2_key: str = "si_mean",
    minimise_obj1: bool = True,
    minimise_obj2: bool = False,
) -> list[dict]:
    """
    Extract the Pareto-optimal set from screening results.

    By default:
        - obj1 (ΔG_collagen): minimise (more negative = stronger binding)
        - obj2 (SI):          maximise  (higher = more selective for collagen)

    A solution *dominates* another if it is ≤ in both objectives and < in at
    least one (after flipping sign for maximisation objectives).

    Parameters
    ----------
    results : list[dict]
        Screening results from ``screen_candidates``.
    obj1_key : str
        Dict key for objective 1 (default: 'delta_g_mean').
    obj2_key : str
        Dict key for objective 2 (default: 'si_mean').
    minimise_obj1 : bool
        True → lower is better for obj1.
    minimise_obj2 : bool
        True → lower is better for obj2.  False → higher is better (SI).

    Returns
    -------
    list[dict]
        Subset of results on the Pareto front, sorted by obj1.
    """
    if not results:
        return []

    # Build objective matrix (both objectives to minimise)
    n = len(results)
    obj = np.zeros((n, 2))
    for i, r in enumerate(results):
        obj[i, 0] = r.get(obj1_key, 0.0) * (1.0 if minimise_obj1 else -1.0)
        obj[i, 1] = r.get(obj2_key, 0.0) * (1.0 if minimise_obj2 else -1.0)

    # Find non-dominated set
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            # j dominates i  if  j ≤ i in both  AND  j < i in at least one
            if (obj[j] <= obj[i]).all() and (obj[j] < obj[i]).any():
                is_pareto[i] = False
                break

    front = [results[i] for i in range(n) if is_pareto[i]]
    # Sort by obj1
    front.sort(key=lambda r: r.get(obj1_key, 0.0),
               reverse=(not minimise_obj1))
    return front


# ── Uncertainty filter ───────────────────────────────────────────────────────

def filter_by_uncertainty(
    results: list[dict],
    max_std: float = 0.5,
    std_key: str = "delta_g_std",
) -> list[dict]:
    """
    Discard compounds whose ensemble uncertainty exceeds a threshold.

    Compounds with std > ``max_std`` (kcal/mol) are considered unreliable
    predictions and removed from consideration.

    Parameters
    ----------
    results : list[dict]
        Screening results.
    max_std : float
        Maximum allowed standard deviation (kcal/mol).  Default 0.5.
    std_key : str
        Dict key for the uncertainty value.

    Returns
    -------
    list[dict]
        Filtered results with std ≤ max_std.
    """
    filtered = [r for r in results if r.get(std_key, 0.0) <= max_std]
    n_removed = len(results) - len(filtered)
    if n_removed > 0:
        logger.info(
            f"Uncertainty filter: removed {n_removed}/{len(results)} "
            f"compounds (std > {max_std})"
        )
    return filtered


# ── Diversity filter via Tanimoto ────────────────────────────────────────────

def _tanimoto_diverse(
    smiles_list: list[str],
    max_n: int,
    min_sim: float = 0.7,
) -> list[int]:
    """
    Greedy diversity selection: keep up to max_n compounds with pairwise
    Tanimoto similarity < min_sim.

    Returns indices into smiles_list.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit import DataStructs
    except ImportError:
        # Fallback: just take first max_n
        return list(range(min(len(smiles_list), max_n)))

    fps = []
    valid_idx = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fps.append(fp)
            valid_idx.append(i)

    if not fps:
        return []

    selected: list[int] = [valid_idx[0]]
    selected_fps = [fps[0]]

    for k in range(1, len(fps)):
        if len(selected) >= max_n:
            break
        # Check max similarity to any already-selected compound
        max_sim = max(
            DataStructs.TanimotoSimilarity(fps[k], sfp)
            for sfp in selected_fps
        )
        if max_sim < min_sim:
            selected.append(valid_idx[k])
            selected_fps.append(fps[k])

    return selected


# ── Top-candidate shortlisting ───────────────────────────────────────────────

def shortlist_candidates(
    results: list[dict],
    top_k: int = 10,
    max_std: float = 0.5,
    std_key: str = "delta_g_std",
    obj_key: str = "delta_g_mean",
    use_pareto: bool = True,
    si_key: Optional[str] = "si_mean",
    diversity_filter: bool = True,
    min_tanimoto: float = 0.7,
    minimise_obj: bool = True,
) -> list[dict]:
    """
    End-to-end candidate shortlisting pipeline.

    Steps:
        1. Filter by uncertainty (std > max_std → discard)
        2. Optionally extract Pareto front (if si_key present in results)
        3. Optionally apply diversity filter (Tanimoto)
        4. Rank by primary objective and return top_k

    Parameters
    ----------
    results : list[dict]
        Raw screening results.
    top_k : int
        Number of candidates to return.
    max_std : float
        Uncertainty threshold (kcal/mol).
    std_key : str
        Key for uncertainty value.
    obj_key : str
        Primary ranking objective key.
    use_pareto : bool
        If True and si_key is in results, use Pareto front.
    si_key : str | None
        Key for selectivity objective.
    diversity_filter : bool
        Apply Tanimoto diversity filter.
    min_tanimoto : float
        Minimum dissimilarity threshold for diversity.
    minimise_obj : bool
        True → smaller obj_key is better (for ΔG).

    Returns
    -------
    list[dict]
        Top-k candidates with additional 'rank' field.
    """
    # Step 1: Uncertainty filter
    reliable = filter_by_uncertainty(results, max_std=max_std, std_key=std_key)

    if not reliable:
        logger.warning("No compounds passed the uncertainty filter!")
        return []

    # Step 2: Pareto front (if SI data available)
    has_si = any(si_key in r for r in reliable) if si_key else False
    if use_pareto and has_si:
        pool = pareto_front(
            reliable,
            obj1_key=obj_key,
            obj2_key=si_key,
            minimise_obj1=minimise_obj,
            minimise_obj2=False,  # maximise SI
        )
        # If Pareto front is too small, fall back to full reliable set
        if len(pool) < top_k:
            pool = reliable
    else:
        pool = reliable

    # Step 3: Sort by primary objective
    pool.sort(key=lambda r: r.get(obj_key, 0.0),
              reverse=(not minimise_obj))

    # Step 4: Diversity filter
    if diversity_filter:
        smiles_pool = [r["smiles"] for r in pool]
        diverse_idx = _tanimoto_diverse(
            smiles_pool, max_n=top_k * 3, min_sim=min_tanimoto
        )
        pool = [pool[i] for i in diverse_idx]

    # Step 5: Take top_k
    top = pool[:top_k]

    # Add rank
    for rank, entry in enumerate(top, 1):
        entry["rank"] = rank

    logger.info(f"Shortlisted {len(top)} candidates from {len(results)} screened")
    return top
