"""
Graph_model.model.option_b
============================
Option B — Dual Encoder with Cross-Attention (Primary Model)
=============================================================

Architecture
------------
  GNN_L : 4-layer GATv2  on ligand molecular graph
          → h_atom  [total_atoms, H]  (atom-level embeddings)

  GNN_P : 4-layer GATv2  on protein binding-site graph
          → h_residue  [total_residues, H]  (residue-level embeddings)

  Cross-Attention (stacked n_cross_layers times):
    Q = h_atom,     K = V = h_residue
    → interaction matrix  AttnWeights [N_atoms, N_res]   (per graph)
    → updated h_atom via weighted residue context
    (optionally + geometry bias from Level-3 bipartite edge distances)

  Readout:
    global mean pool over updated h_atom → h_lig  [B, H]
    Concat condition → MLP → ΔG

Interpretability
----------------
After calling model.forward(), the cross-attention weights are accessible via:
    model.last_attn_weights   list of dicts { 'atom_idx': ..., 'res_idx': ...,
                                               'weight': ...  }
For the paper's binding-mode validation:
  At GLU_cluster22, the highest-weight atom–residue pairs should reproduce
  the six ARG496–OH, ARG499–OH, LYS723–COOH contacts identified from docking.

References
----------
Vaswani A. et al. "Attention is All You Need." NeurIPS 2017.
Bahdanau D. et al. "Neural Machine Translation by Jointly Learning to Align
and Translate." ICLR 2015.  arXiv:1409.0473
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.utils import unbatch
from torch.nn.utils.rnn import pad_sequence

from .config import OptionBConfig
from .option_a import _scalar_to_batch, _encode_raw


# ── GATv2 building block with edge features ───────────────────────────────────

class _GATv2Stack(nn.Module):
    """
    n_layers of GATv2Conv with residual + BN + GELU.
    Final layer averages heads (concat=False) → output dim = head_dim.
    """

    def __init__(
        self,
        in_dim:    int,
        hidden:    int,
        head_dim:  int,
        heads:     int,
        n_layers:  int,
        edge_dim:  int,
        dropout:   float,
    ) -> None:
        super().__init__()
        H0 = hidden       # remember original hidden for post-projection
        self.proj = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            concat = (i < n_layers - 1)
            out_ch = head_dim
            self.layers.append(
                GATv2Conv(
                    in_channels=hidden,
                    out_channels=out_ch,
                    heads=heads,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    concat=concat,
                )
            )
            out_total = out_ch * heads if concat else out_ch
            setattr(self, f"bn_{i}", nn.BatchNorm1d(out_total))
            if hidden != out_total:
                setattr(self, f"skip_{i}", nn.Linear(hidden, out_total, bias=False))
            else:
                setattr(self, f"skip_{i}", nn.Identity())
            hidden = out_total
        # Project final GAT output (head_dim) back to H0 so callers see stable dim
        self.post_proj = nn.Linear(hidden, H0) if hidden != H0 else nn.Identity()
        self.out_dim = H0
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, ei: Tensor, ea: Tensor) -> Tensor:
        x = self.proj(x)
        for i, conv in enumerate(self.layers):
            bn   = getattr(self, f"bn_{i}")
            skip = getattr(self, f"skip_{i}")
            out  = conv(x, ei, ea)
            out  = bn(out)
            out  = F.gelu(out)
            out  = self.drop(out)
            x    = out + skip(x)
        return self.post_proj(x)


# ── Per-graph cross-attention ─────────────────────────────────────────────────

class _CrossAttentionLayer(nn.Module):
    """
    Multi-head cross-attention: atoms attend over residues.

    Input  : q [N_lig, H], kv [N_res, H]
    Output : q_updated [N_lig, H], attn_weights [heads, N_lig, N_res]

    Also supports an optional geometry bias tensor [N_lig, N_res] (inverse
    distance from Level-3 edges) added to unnormalised attention logits.
    """

    def __init__(self, hidden: int, heads: int, dropout: float) -> None:
        super().__init__()
        assert hidden % heads == 0, "hidden must be divisible by heads"
        self.heads   = heads
        self.d_head  = hidden // heads
        self.scale   = self.d_head ** -0.5

        self.q_proj  = nn.Linear(hidden, hidden, bias=False)
        self.k_proj  = nn.Linear(hidden, hidden, bias=False)
        self.v_proj  = nn.Linear(hidden, hidden, bias=False)
        self.out_proj= nn.Linear(hidden, hidden)
        self.norm    = nn.LayerNorm(hidden)
        self.drop    = nn.Dropout(dropout)

    def forward(
        self,
        q:    Tensor,                   # [N_L, H]
        kv:   Tensor,                   # [N_R, H]
        bias: Tensor | None = None,     # [N_L, N_R]  optional geometry bias
    ) -> tuple[Tensor, Tensor]:
        N_L, H = q.shape
        N_R    = kv.shape[0]
        h, d   = self.heads, self.d_head

        # Project
        Q = self.q_proj(q) .view(N_L, h, d).transpose(0, 1)   # [h, N_L, d]
        K = self.k_proj(kv).view(N_R, h, d).transpose(0, 1)   # [h, N_R, d]
        V = self.v_proj(kv).view(N_R, h, d).transpose(0, 1)   # [h, N_R, d]

        # Scaled dot-product attention
        logits = self.scale * torch.bmm(Q, K.transpose(1, 2))  # [h, N_L, N_R]

        # Optional geometry bias: replicated across heads
        if bias is not None:
            logits = logits + bias.unsqueeze(0)

        attn  = F.softmax(logits, dim=-1)                       # [h, N_L, N_R]
        attn  = self.drop(attn)

        ctx   = torch.bmm(attn, V)                              # [h, N_L, d]
        ctx   = ctx.transpose(0, 1).reshape(N_L, H)            # [N_L, H]

        out   = self.out_proj(ctx)                              # [N_L, H]
        out   = self.norm(out + q)                              # residual LN

        # Average attention weights across heads for return
        attn_mean = attn.mean(dim=0)                            # [N_L, N_R]
        return out, attn_mean


# ── Main model ────────────────────────────────────────────────────────────────

class OptionB(nn.Module):
    """
    Dual Encoder with stacked Cross-Attention.

    Forward signature
    -----------------
    data : torch_geometric.data.HeteroData
      Required keys:
        data['ligand'].x, .edge_index (bond→ligand), .edge_attr, .batch
        data['residue'].x, .edge_index (contact→residue), .edge_attr, .batch
        data.ph_enc / data.temp_enc / data.box_idx / data.receptor_flag
      Optional for geometry bias:
        data['ligand','interacts','residue'].edge_index  [2, E_bip]
        data['ligand','interacts','residue'].edge_attr   [E_bip, 8]

    Returns
    -------
    delta_g       : Tensor  [B, 1]
    attn_summaries: list[dict]  — per-graph attention summary  (always populated)
    """

    def __init__(self, cfg: OptionBConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or OptionBConfig()
        self.cfg    = cfg
        H           = cfg.gat_head_dim * cfg.gat_heads   # default 128

        # Ligand GNN
        self.gnn_l = _GATv2Stack(
            in_dim   = cfg.ligand_node_dim,
            hidden   = H,
            head_dim = cfg.gat_head_dim,
            heads    = cfg.gat_heads,
            n_layers = cfg.n_layers,
            edge_dim = cfg.ligand_edge_dim,
            dropout  = cfg.dropout,
        )

        # Protein GNN
        self.gnn_p = _GATv2Stack(
            in_dim   = cfg.protein_node_dim,
            hidden   = H,
            head_dim = cfg.gat_head_dim,
            heads    = cfg.gat_heads,
            n_layers = cfg.n_layers,
            edge_dim = cfg.protein_edge_dim,
            dropout  = cfg.dropout,
        )

        # Geometry bias projector (bipartite dist_norm → scalar bias per atom-residue pair)
        if cfg.use_bipartite:
            self.bip_bias_proj = nn.Linear(cfg.bipartite_edge_dim, 1)
        else:
            self.bip_bias_proj = None

        # Stacked cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            _CrossAttentionLayer(H, cfg.attn_heads, cfg.attn_dropout)
            for _ in range(cfg.n_cross_layers)
        ])

        # Condition encoder
        H_cond = cfg.gat_head_dim   # 32
        self.box_embed = nn.Embedding(cfg.n_box_types, cfg.box_embed_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(cfg.cond_dim, 64), nn.ReLU(),
            nn.Linear(64, H_cond),
        )

        # Readout MLP
        self.mlp = nn.Sequential(
            nn.Linear(H + H_cond, cfg.mlp_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.mlp_hidden, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        # Storage for last-forward attention weights (interpretability)
        self.last_attn_weights: list[dict] = []

    def forward(self, data) -> tuple[Tensor, list[dict]]:
        device = data['ligand'].x.device

        # ── 1. GNN encoding ───────────────────────────────────────────────────
        x_lig = self.gnn_l(
            data['ligand'].x,
            data['ligand', 'bond', 'ligand'].edge_index,
            data['ligand', 'bond', 'ligand'].edge_attr,
        )   # [total_atoms, H]

        # Protein graph — handle empty graphs (N_res=0) gracefully
        H = x_lig.shape[-1]
        if data['residue'].x.shape[0] > 0:
            x_res = self.gnn_p(
                data['residue'].x,
                data['residue', 'contact', 'residue'].edge_index,
                data['residue', 'contact', 'residue'].edge_attr,
            )   # [total_residues, H]
        else:
            # No binding-site residues → zero context; cross-attention returns q as-is
            B = int(data['ligand'].batch.max().item()) + 1
            x_res = torch.zeros(B, H, device=device)

        # ── 2. Optional bipartite geometry bias ───────────────────────────────
        bip_bias_dict: dict[int, Tensor] = {}
        if self.bip_bias_proj is not None:
            try:
                bip_ei = data['ligand', 'interacts', 'residue'].edge_index
                bip_ea = data['ligand', 'interacts', 'residue'].edge_attr
                if bip_ei.shape[1] > 0 and x_res.shape[0] > 0:
                    bip_bias_dict = self._build_bip_bias(
                        bip_ei, bip_ea,
                        data['ligand'].batch,
                        data['residue'].batch,
                        data['ligand'].x.shape[0],   # use original node count
                        data['residue'].x.shape[0],  # use original node count
                    )
            except Exception:
                bip_bias_dict = {}

        # ── 3. Per-graph cross-attention ──────────────────────────────────────
        lig_batch = data['ligand'].batch
        # residue.batch may not cover all B graphs if some have 0 residues;
        # unbatch would then return fewer entries → index error.
        # Build a safe per-graph residue list that always has B elements.
        lig_list = unbatch(x_lig, lig_batch)   # list[Tensor [N_i, H]]
        B = len(lig_list)

        if x_res.shape[0] > 0 and hasattr(data['residue'], 'batch'):
            res_batch = data['residue'].batch
            _res_unbatched = unbatch(x_res, res_batch)
            # Map unbatched residues to their graph indices
            res_dict: dict[int, Tensor] = {}
            seen_batches = res_batch.unique().tolist()
            for idx_pos, graph_idx in enumerate(seen_batches):
                if idx_pos < len(_res_unbatched):
                    res_dict[int(graph_idx)] = _res_unbatched[idx_pos]
            res_list = [res_dict.get(g, torch.zeros(0, H, device=device)) for g in range(B)]
        else:
            res_list = [torch.zeros(0, H, device=device)] * B
        updated_lig: list[Tensor] = []
        attn_summaries: list[dict] = []

        for g in range(B):
            q_g   = lig_list[g]            # [N_L_g, H]
            kv_g  = res_list[g]            # [N_R_g, H]
            bias_g = bip_bias_dict.get(g, None)  # [N_L_g, N_R_g] or None

            # Validate bias dimensions match actual graph tensors;
            # mismatches can occur when PDBbind graphs with 0 residues
            # shift the residue batch indexing.
            if bias_g is not None and (
                bias_g.shape[0] != q_g.shape[0]
                or bias_g.shape[1] != kv_g.shape[0]
            ):
                bias_g = None

            # If residue graph is empty, skip cross-attention
            if kv_g.shape[0] == 0:
                updated_lig.append(q_g)
                attn_summaries.append({})
                continue

            attn_all_layers: list[Tensor] = []
            for ca_layer in self.cross_attn_layers:
                q_g, attn_w = ca_layer(q_g, kv_g, bias=bias_g)
                attn_all_layers.append(attn_w)    # [N_L_g, N_R_g]

            # Store for interpretability: average across stacked layers
            final_attn = torch.stack(attn_all_layers, 0).mean(0)   # [N_L, N_R]
            attn_summaries.append({
                'attn_weights': final_attn.detach().cpu(),  # [N_L, N_R]
                'graph_idx':    g,
            })
            updated_lig.append(q_g)

        self.last_attn_weights = attn_summaries

        # ── 4. Readout ────────────────────────────────────────────────────────
        # Reconstruct flat tensor from per-graph list
        x_updated = torch.cat(updated_lig, dim=0)          # [total_atoms, H]
        h_lig = global_mean_pool(x_updated, lig_batch)     # [B, H]

        # Condition
        cond = self._encode_condition(data, device, B)     # [B, cond_dim]
        h_cond = self.cond_proj(cond)                      # [B, H_cond]

        h = torch.cat([h_lig, h_cond], dim=-1)
        return self.mlp(h), attn_summaries                 # [B, 1], list

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_bip_bias(
        self,
        bip_ei:    Tensor,
        bip_ea:    Tensor,
        lig_batch: Tensor,
        res_batch: Tensor,
        n_lig:     int,
        n_res:     int,
    ) -> dict[int, Tensor]:
        """
        Convert Level-3 bipartite edges into per-graph dense geometry bias
        tensors [N_L_g, N_R_g].

        Each edge (atom_i, res_j) contributes bias = bip_bias_proj(edge_attr).
        Positions with no edge remain 0 (neutral bias).
        """
        B = int(lig_batch.max().item()) + 1
        # Offset lookup: map global atom/res index → graph-local index
        lig_offset = torch.zeros(n_lig, dtype=torch.long, device=lig_batch.device)
        res_offset = torch.zeros(n_res, dtype=torch.long, device=res_batch.device)
        lig_sizes = torch.zeros(B, dtype=torch.long, device=lig_batch.device)
        res_sizes = torch.zeros(B, dtype=torch.long, device=res_batch.device)

        for g in range(B):
            lig_mask = lig_batch == g
            res_mask = res_batch == g
            lig_idx_g = lig_mask.nonzero(as_tuple=True)[0]
            res_idx_g = res_mask.nonzero(as_tuple=True)[0]
            if len(lig_idx_g): lig_offset[lig_idx_g] = torch.arange(
                len(lig_idx_g), device=lig_batch.device)
            if len(res_idx_g): res_offset[res_idx_g] = torch.arange(
                len(res_idx_g), device=res_batch.device)
            lig_sizes[g] = lig_mask.sum()
            res_sizes[g] = res_mask.sum()

        bias_per_graph: dict[int, Tensor] = {}
        a_idx  = bip_ei[0]    # global atom indices
        r_idx  = bip_ei[1]    # global residue indices
        scalar_bias = self.bip_bias_proj(bip_ea).squeeze(-1)   # [E_bip]

        g_of_a = lig_batch[a_idx]   # which graph each edge belongs to

        for g in range(B):
            mask = g_of_a == g
            if not mask.any():
                continue
            nl = int(lig_sizes[g].item())
            nr = int(res_sizes[g].item())
            dense = torch.zeros(nl, nr, device=bip_ea.device)
            loc_a = lig_offset[a_idx[mask]]
            loc_r = res_offset[r_idx[mask]]
            dense.index_put_((loc_a, loc_r), scalar_bias[mask], accumulate=True)
            bias_per_graph[g] = dense

        return bias_per_graph

    def _encode_condition(self, data, device: torch.device, B: int) -> Tensor:
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
