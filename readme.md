# Collagen–Gallic Acid Crosslinking

Graph neural network (Models A–I) code for Vinardo ΔG / CSI prediction on collagen and MMP-1 docking data.

- **Code:** [`Graph_model/`](./Graph_model)
- Datasets / `.pkl` caches are **not** included; generate locally from docking outputs.

## Install

```bash
# from repository root
pip install -e .

# or
pip install -r Graph_model/requirements.txt

# optional conda
conda env create -f Graph_model/environment.yml
conda activate graph-model
pip install -e .
```

Install files: `pyproject.toml`, `Graph_model/requirements.txt`, `Graph_model/environment.yml`.

See `Graph_model/README.md` for architecture table, layout, and reproduction notes.
