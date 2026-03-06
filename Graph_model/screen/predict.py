"""
Graph_model.screen.predict
============================
Batch inference with an ensemble for virtual screening.

Given a library of SMILES, builds ligand-only HeteroData graphs (Level-1)
and runs each through the ensemble to obtain:
    - ΔG_collagen  (mean ± std)
    - SI (selectivity index, if OptionD ensemble)
    - Per-compound uncertainty

The pipeline uses ligand-only graphs (no protein/bipartite) for rapid
screening.  For final re-scoring of top hits, a full ThreeLevelGraphBuilder
should be used.

Usage
-----
>>> from Graph_model.screen.predict import screen_candidates
>>> results = screen_candidates(ensemble, smiles_list, ph=5.0, temp_c=25)
>>> results[0]
{'smiles': 'OC(=O)c1cc(O)c(O)c(O)c1',
 'delta_g_mean': -6.12,
 'delta_g_std':  0.34,
 'si_mean':      2.1,
 'si_std':       0.15, ...}
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

try:
    from torch_geometric.data import HeteroData, Batch
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False

try:
    from Graph_model.graph.level1_ligand import mol_to_ligand_graph
except ImportError:
    mol_to_ligand_graph = None  # type: ignore


# ── Condition defaults ────────────────────────────────────────────────────────

# PropKa protonation encoding from data/config.py
_PROPKA: dict[float, float] = {5.0: 0.85, 5.5: 0.15, 7.0: 0.02}
_TEMP_MIN, _TEMP_RANGE = 4.0, 33.0   # normalisation: (T - 4) / 33
_DEFAULT_BOX_IDX = 7                  # global_blind
_DEFAULT_RECEPTOR_FLAG = 0.0          # collagen


def _encode_ph(ph: float) -> float:
    """Map pH to PropKa protonation fraction."""
    if ph in _PROPKA:
        return _PROPKA[ph]
    return min(_PROPKA, key=lambda x: abs(x - ph))


def _encode_temp(temp_c: float) -> float:
    """Normalise temperature to [0, 1]."""
    return (temp_c - _TEMP_MIN) / _TEMP_RANGE


# ── Graph construction for screening ─────────────────────────────────────────

def _smiles_to_screen_graph(
    smiles: str,
    ph: float = 5.0,
    temp_c: float = 25.0,
    box_idx: int = _DEFAULT_BOX_IDX,
    receptor_flag: float = _DEFAULT_RECEPTOR_FLAG,
) -> Optional[HeteroData]:
    """
    Build a ligand-only HeteroData graph for rapid screening.

    This creates a Level-1 ligand graph with dummy protein/bipartite nodes
    to satisfy the model's expected input schema.  Condition metadata is
    attached identically to the training data format.

    Parameters
    ----------
    smiles : str
        Molecule SMILES string.
    ph : float
        Environmental pH (5.0, 5.5, or 7.0).
    temp_c : float
        Temperature in °C.
    box_idx : int
        Docking box type index (0-7; 7 = global_blind for screening).
    receptor_flag : float
        0.0 = collagen, 1.0 = MMP-1.

    Returns
    -------
    HeteroData or None if SMILES cannot be parsed.
    """
    assert mol_to_ligand_graph is not None, \
        "Graph_model.graph.level1_ligand not available"

    try:
        node_feat, edge_index, edge_feat = mol_to_ligand_graph(smiles)
    except ValueError:
        return None

    n_lig = node_feat.shape[0]

    data = HeteroData()
    data['ligand'].x = torch.tensor(node_feat, dtype=torch.float32)
    data['ligand', 'bond', 'ligand'].edge_index = torch.tensor(
        edge_index, dtype=torch.long)
    data['ligand', 'bond', 'ligand'].edge_attr = torch.tensor(
        edge_feat, dtype=torch.float32)

    # Dummy protein graph (single residue node, no edges)
    from Graph_model.graph import PROTEIN_NODE_DIM, PROTEIN_EDGE_DIM, BIPARTITE_EDGE_DIM
    data['residue'].x = torch.zeros(1, PROTEIN_NODE_DIM, dtype=torch.float32)
    data['residue', 'contact', 'residue'].edge_index = torch.zeros(
        2, 0, dtype=torch.long)
    data['residue', 'contact', 'residue'].edge_attr = torch.zeros(
        0, PROTEIN_EDGE_DIM, dtype=torch.float32)

    # Dummy bipartite edges
    data['ligand', 'interacts', 'residue'].edge_index = torch.zeros(
        2, 0, dtype=torch.long)
    data['ligand', 'interacts', 'residue'].edge_attr = torch.zeros(
        0, BIPARTITE_EDGE_DIM, dtype=torch.float32)

    # Condition encoding (matches model's expected attributes)
    data.ph_enc = torch.tensor(_encode_ph(ph), dtype=torch.float32)
    data.temp_enc = torch.tensor(_encode_temp(temp_c), dtype=torch.float32)
    data.box_idx = torch.tensor(box_idx, dtype=torch.long)
    data.receptor_flag = torch.tensor(receptor_flag, dtype=torch.float32)

    # Metadata
    data.y = torch.tensor([[float('nan')]], dtype=torch.float32)
    data.smiles = smiles
    data.tier = 0
    data.ligand_name = "screen"

    return data


# ── Main screening function ──────────────────────────────────────────────────

@torch.no_grad()
def screen_candidates(
    ensemble,
    smiles_list: list[str],
    ph: float = 5.0,
    temp_c: float = 25.0,
    box_idx: int = _DEFAULT_BOX_IDX,
    receptor_flag: float = _DEFAULT_RECEPTOR_FLAG,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    include_si: bool = False,
) -> list[dict]:
    """
    Screen a list of SMILES through a trained ensemble.

    Parameters
    ----------
    ensemble : DeepEnsemble or any nn.Module
        Trained model (ensemble preferred for uncertainty).
    smiles_list : list[str]
        Candidate SMILES to screen.
    ph : float
        Environmental pH for condition encoding.
    temp_c : float
        Temperature in °C.
    box_idx : int
        Docking box type index.
    receptor_flag : float
        0.0 for collagen, 1.0 for MMP-1.
    batch_size : int
        Inference mini-batch size.
    device : torch.device | None
        Device for inference.
    include_si : bool
        If True, also extract selectivity index from OptionD ensemble.

    Returns
    -------
    list[dict]
        Each dict contains:
            smiles       : str
            delta_g_mean : float    — predicted ΔG (kcal/mol)
            delta_g_std  : float    — uncertainty (ensemble std)
            si_mean      : float    — selectivity index (if include_si)
            si_std       : float    — SI uncertainty (if include_si)
            total_std    : float    — total uncertainty  √(σ²_epi + σ²_alea)
    """
    assert _HAS_PYG, "torch_geometric is required for screening"

    if device is None:
        device = next(ensemble.parameters()).device

    # Build graphs
    graphs: list[tuple[str, HeteroData]] = []
    failed: list[str] = []
    for smi in smiles_list:
        g = _smiles_to_screen_graph(smi, ph, temp_c, box_idx, receptor_flag)
        if g is not None:
            graphs.append((smi, g))
        else:
            failed.append(smi)

    if failed:
        logger.warning(f"Skipped {len(failed)} unparseable SMILES")

    ensemble.eval()
    results: list[dict] = []

    # Process in mini-batches
    for i in range(0, len(graphs), batch_size):
        batch_graphs = graphs[i : i + batch_size]
        batch_smiles = [s for s, _ in batch_graphs]
        batch_data = Batch.from_data_list([g for _, g in batch_graphs])
        batch_data = batch_data.to(device)

        out = ensemble(batch_data)

        # Extract predictions
        if isinstance(out, dict):
            mu_mean = out.get('mu_mean', out.get('collagen'))
            mu_std = out.get('mu_std', torch.zeros_like(mu_mean))
            total_var = out.get('total_var', mu_std ** 2)
            si_pred = out.get('si', None)
        elif isinstance(out, Tensor):
            mu_mean = out
            mu_std = torch.zeros_like(out)
            total_var = mu_std
            si_pred = None
        elif isinstance(out, tuple):
            mu_mean = out[0]
            mu_std = torch.zeros_like(mu_mean)
            total_var = mu_std
            si_pred = None
        else:
            raise ValueError(f"Unexpected ensemble output type: {type(out)}")

        total_std = total_var.sqrt()

        for j, smi in enumerate(batch_smiles):
            entry = {
                "smiles":       smi,
                "delta_g_mean": mu_mean[j].item(),
                "delta_g_std":  mu_std[j].item(),
                "total_std":    total_std[j].item(),
            }
            if include_si and si_pred is not None:
                entry["si_mean"] = si_pred[j].item()
                # SI uncertainty from ensemble spread (approximation)
                if 'all_mu' in out:
                    all_si = out.get('all_si')
                    if all_si is not None:
                        entry["si_std"] = all_si[:, j].std().item()
                    else:
                        entry["si_std"] = 0.0
                else:
                    entry["si_std"] = 0.0
            results.append(entry)

    logger.info(f"Screened {len(results)} / {len(smiles_list)} compounds")
    return results
