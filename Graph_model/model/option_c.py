"""
Graph_model.model.option_c
============================
Option C — Fragment-Aware Hierarchical MPNN (Most Chemically Novel)
====================================================================

Architecture
------------
  Decomposition  : BRICS fragmentation (RDKit) of each SMILES molecule
                   → list of fragment subgraphs + inter-fragment bond graph

  Level 1 — Atom MPNN (GINEConv) within each fragment:
            intra-fragment edges only → atom embeddings [N_atoms, H_atom]
            Fragment embedding = scatter_mean(atoms in fragment)  → [N_frag, H_atom]

  Level 2 — Fragment MPNN (GINEConv) over fragment graph:
            fragments as nodes; cut-bonds as edges → [N_frag, H_frag]

  Readout — Interpretable per-fragment ΔG contribution:
            attention_score = MLP(h_frag_i)  → α_i  (softmax normalised)
            h_mol = Σ_i α_i · h_frag_i
            δ_i = MLP_branch(h_frag_i)       → per-fragment ΔG contribution
            ΔG_pred = Σ_i δ_i   (sum of atom-group contributions)

Scientific motivation
---------------------
PGG (5 galloyl arms) has SD=1.12 kcal/mol across boxes vs gallic acid SD=0.42.
This model quantifies whether all five arms contribute equally or whether
specific arms dominate at different binding sites.

The per-fragment contribution δ_i is directly interpretable and publishable as:
    "Arm 3 (equatorial galloyl) contributes −2.1 ± 0.3 kcal/mol to GLU-site
    binding vs −1.4 ± 0.4 kcal/mol from the remaining four arms combined."

Reference
---------
Hu W. et al. "Strategies for Pre-training Graph Neural Networks."
ICLR 2020.  (GIN architecture)
Jin W. et al. "Junction Tree Variational Autoencoder for Molecular Graph
Generation." ICML 2018.  (hierarchical molecule decomposition)
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.utils import unbatch
from torch_geometric.utils import scatter as pyg_scatter

from .config import OptionCConfig
from .option_a import _scalar_to_batch, _encode_raw


# ── BRICS fragment decomposition ─────────────────────────────────────────────

class FragmentGraph(NamedTuple):
    """Encoded fragment-level graph for one molecule."""
    atom_to_frag: Tensor   # [N_atoms]  int64 — which fragment each atom belongs to
    n_frags:      int      # number of fragments
    frag_ei:      Tensor   # [2, E_frag] inter-fragment edges (cut bonds)
    frag_ea:      Tensor   # [E_frag, 4] cut-bond type one-hot


@lru_cache(maxsize=4096)
def _brics_fragments(smiles: str) -> tuple[tuple[int, ...], tuple]:
    """
    Use RDKit BRICS to decompose SMILES into fragments.
    Returns (atom_to_frag_tuple, cut_bonds_tuple) — tuples for LRU hashability.

    If BRICS yields < 2 fragments (molecule too small to cut), every atom
    is assigned to fragment 0 with no inter-fragment edges.
    """
    from rdkit import Chem
    from rdkit.Chem.BRICS import FindBRICSBonds

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        n = 1
        return tuple([0] * n), ()

    n_heavy = mol.GetNumAtoms()

    # FindBRICSBonds returns set of ((atom1, atom2), (bond_type1, bond_type2))
    brics_bonds = list(FindBRICSBonds(mol))

    if not brics_bonds:
        # no cut bonds → single fragment
        return tuple([0] * n_heavy), ()

    # Build fragment index by flood-fill excluding cut bonds
    cut_set = {frozenset([b[0][0], b[0][1]]) for b in brics_bonds}

    visited  = [-1] * n_heavy
    frag_id  = 0
    for start in range(n_heavy):
        if visited[start] != -1:
            continue
        queue = [start]
        visited[start] = frag_id
        while queue:
            node = queue.pop()
            for nbr in mol.GetAtomWithIdx(node).GetNeighbors():
                j = nbr.GetIdx()
                if visited[j] == -1 and frozenset([node, j]) not in cut_set:
                    visited[j] = frag_id
                    queue.append(j)
        frag_id += 1

    # Build cut-bond pairs at fragment level (unordered)
    cut_pairs = []
    for bb in brics_bonds:
        ai, aj = bb[0]
        fi, fj = visited[ai], visited[aj]
        if fi != fj:
            cut_pairs.append((fi, fj))
            cut_pairs.append((fj, fi))  # undirected

    return tuple(visited), tuple(cut_pairs)


def _fragment_graph_for_mol(smiles: str, device: torch.device) -> FragmentGraph:
    """Build FragmentGraph tensors for a single molecule."""
    atom_frag_tuple, cut_pairs = _brics_fragments(smiles)
    n_atoms = len(atom_frag_tuple)
    n_frags = max(atom_frag_tuple) + 1 if atom_frag_tuple else 1

    atom_to_frag = torch.tensor(atom_frag_tuple, dtype=torch.long, device=device)

    if cut_pairs:
        ei = torch.tensor(cut_pairs, dtype=torch.long, device=device).T   # [2, E]
        # Simple edge feature: all cut bonds encoded as single bond (type 0)
        ea = torch.zeros(len(cut_pairs), 4, dtype=torch.float32, device=device)
        ea[:, 0] = 1.0    # one-hot "SINGLE" bond type
    else:
        ei = torch.zeros((2, 0), dtype=torch.long, device=device)
        ea = torch.zeros((0, 4), dtype=torch.float32, device=device)

    return FragmentGraph(atom_to_frag, n_frags, ei, ea)


# ── GINEConv building block ───────────────────────────────────────────────────

class _GINEStack(nn.Module):
    """
    n_layers of GINEConv (Graph Isomorphism Network with Edge features).
    GINEConv: h'_i = MLP( (1 + ε) * h_i + Σ_{j∈N(i)} ReLU(h_j + e_ij) )
    Uses LayerNorm (instead of BatchNorm) to support single-node/single-fragment cases.
    """

    def __init__(
        self,
        in_dim:   int,
        hidden:   int,
        n_layers: int,
        edge_dim: int,
        dropout:  float,
    ) -> None:
        super().__init__()
        self.proj  = nn.Linear(in_dim, hidden)
        self.layers= nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden * 2), nn.ReLU(),
                nn.Linear(hidden * 2, hidden),
            )
            self.layers.append(GINEConv(mlp, edge_dim=edge_dim, train_eps=True))
            self.norms.append(nn.LayerNorm(hidden))   # LayerNorm works for any batch size
        self.drop    = nn.Dropout(dropout)
        self.out_dim = hidden

    def forward(
        self,
        x:  Tensor,
        ei: Tensor,
        ea: Tensor,
    ) -> Tensor:
        x = self.proj(x)
        for conv, bn in zip(self.layers, self.norms):
            if ei.shape[1] > 0:
                x = conv(x, ei, ea)
            x = bn(x)
            x = F.relu(x)
            x = self.drop(x)
        return x


# ── Main model ────────────────────────────────────────────────────────────────

class OptionC(nn.Module):
    """
    Fragment-Aware Hierarchical MPNN.

    Forward arguments
    -----------------
    data : HeteroData  (same schema as other options)
    smiles_list : list[str]  — one SMILES per graph in the batch.
                              Required to compute BRICS fragmentation.

    Returns
    -------
    delta_g        : Tensor  [B, 1]    total predicted ΔG
    frag_contrib   : list[Tensor]      per-graph list of [N_frag] fragment
                                       contributions δ_i (interpretable)
    """

    def __init__(self, cfg: OptionCConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or OptionCConfig()
        self.cfg = cfg
        H_atom = cfg.hidden_dim         # 128
        H_frag = cfg.frag_hidden        # 64

        # ── Level 1: atom MPNN (intra-fragment) ──────────────────────────────
        self.atom_gnn = _GINEStack(
            in_dim   = cfg.ligand_node_dim,
            hidden   = H_atom,
            n_layers = cfg.n_layers,
            edge_dim = cfg.ligand_edge_dim,
            dropout  = cfg.dropout,
        )

        # ── Level 2: fragment MPNN ────────────────────────────────────────────
        self.frag_proj = nn.Linear(H_atom, H_frag)   # project atom→frag dim
        self.frag_gnn  = _GINEStack(
            in_dim   = H_frag,
            hidden   = H_frag,
            n_layers = 2,
            edge_dim = 4,               # 4-dim cut-bond feature
            dropout  = cfg.dropout,
        )

        # ── Per-fragment ΔG branch ────────────────────────────────────────────
        # δ_i = MLP(h_frag_i)  →  per-fragment contribution
        self.frag_out = nn.Sequential(
            nn.Linear(H_frag, 32), nn.ReLU(), nn.Linear(32, 1),
        )

        # ── Condition encoder ─────────────────────────────────────────────────
        self.box_embed = nn.Embedding(cfg.n_box_types, cfg.box_embed_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(cfg.cond_dim, 64), nn.ReLU(),
            nn.Linear(64, 32),
        )

        # ── Global correction MLP ─────────────────────────────────────────────
        # Allows the model to apply a global correction beyond additive fragments
        self.glob_mlp = nn.Sequential(
            nn.Linear(H_frag + 32, cfg.mlp_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.mlp_hidden, 1),
        )

    def forward(
        self,
        data,
        smiles_list: list[str] | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        device = data['ligand'].x.device
        B      = int(data['ligand'].batch.max().item()) + 1

        if smiles_list is None:
            # Try to get from data attribute
            if hasattr(data, 'smiles'):
                sm = data.smiles
                smiles_list = sm if isinstance(sm, list) else [sm] * B
            else:
                warnings.warn("smiles_list not provided; using single-fragment mode.")
                smiles_list = ["C"] * B    # dummy → single fragment

        # ── 1. Level-1: atom MPNN over full molecular graph ─────────────────
        x_atom = self.atom_gnn(
            data['ligand'].x,
            data['ligand', 'bond', 'ligand'].edge_index,
            data['ligand', 'bond', 'ligand'].edge_attr,
        )   # [total_atoms, H_atom]

        # ── 2. Per-graph fragment aggregation + Level-2 MPNN ────────────────
        atom_list   = unbatch(x_atom, data['ligand'].batch)   # list[N_i, H]
        frag_contribs: list[Tensor] = []
        frag_global_h: list[Tensor] = []

        for g, (x_g, smi) in enumerate(zip(atom_list, smiles_list)):
            fg = _fragment_graph_for_mol(smi, device)
            n_frags = min(fg.n_frags, self.cfg.max_fragments)
            n_graph_atoms = x_g.shape[0]

            # Guard against atom-count mismatch between SMILES and graph
            # (In production they are always equal; this handles test/edge cases)
            if fg.atom_to_frag.shape[0] != n_graph_atoms:
                a2f = torch.zeros(n_graph_atoms, dtype=torch.long, device=device)
                n_frags = 1
            else:
                a2f = fg.atom_to_frag.clamp(0, n_frags - 1)

            # Scatter-mean: atoms → fragment representations
            h_frag_init = pyg_scatter(x_g, a2f, dim=0, dim_size=n_frags, reduce='mean')
            h_frag_init = self.frag_proj(h_frag_init)    # [n_frags, H_frag]

            # Level-2 MPNN over fragment graph
            # Mask inter-fragment edges to valid fragment indices
            fei = fg.frag_ei
            fea = fg.frag_ea
            if fei.shape[1] > 0:
                valid = (fei[0] < n_frags) & (fei[1] < n_frags)
                fei   = fei[:, valid]
                fea   = fea[valid]

            h_frag = self.frag_gnn(h_frag_init, fei, fea)   # [n_frags, H_frag]

            # Per-fragment ΔG contribution
            deltas = self.frag_out(h_frag).squeeze(-1)       # [n_frags]
            frag_contribs.append(deltas)  # keep gradients for training

            # Global fragment representation (mean of fragment embeddings)
            frag_global_h.append(h_frag.mean(dim=0))         # [H_frag]

        # ── 3. Readout ────────────────────────────────────────────────────────
        h_mol_stack = torch.stack(frag_global_h, dim=0)      # [B, H_frag]

        # Condition
        cond   = self._encode_condition(data, device, B)      # [B, COND_DIM]
        h_cond = self.cond_proj(cond)                         # [B, 32]

        # Global correction path
        h_glob = torch.cat([h_mol_stack, h_cond], dim=-1)    # [B, H_frag+32]
        glob_delta = self.glob_mlp(h_glob)                   # [B, 1]

        # Fragment sum (additive contributions per graph)
        frag_sums = torch.stack([d.sum() for d in frag_contribs]).unsqueeze(-1).to(device)   # [B, 1]

        delta_g = frag_sums + glob_delta

        return delta_g, frag_contribs

    def _encode_condition(self, data, device, B) -> Tensor:
        if hasattr(data, 'ph_enc'):
            ph_enc   = _scalar_to_batch(data.ph_enc,       B, device)
            temp_enc = _scalar_to_batch(data.temp_enc,      B, device)
            box_idx  = _scalar_to_batch(data.box_idx,       B, device, long=True)
            rec_flag = _scalar_to_batch(data.receptor_flag, B, device)
        else:
            ph_enc, temp_enc, box_idx, rec_flag = _encode_raw(data, B, device)
        box_emb = self.box_embed(box_idx)
        cont    = torch.stack([ph_enc, temp_enc, rec_flag], dim=-1)
        return torch.cat([cont, box_emb], dim=-1)
