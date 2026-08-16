"""
Graph_model
===========
Graph Neural Network pipeline for collagen-crosslinker binding-energy prediction.

Three-tier data architecture
  Tier 1  Anchor   — 6,156 Vinardo ΔG records from the 9-ligand/2-receptor campaign
  Tier 2  Transfer — PDBbind v2020 general set (~19,000 complexes), pre-training backbone
  Tier 3  Augment  — ChEMBL/PubChem phenolic binders, structural diversity injection

Novel features
  • Galloyl-unit fragment graph layer  (Section 1b)
  • Engineered condition vectors       (Section 1c)

Entry points
  from Graph_model.data.dataset  import CollagenDockingDataset
  from Graph_model.data.splitter import StratifiedSplitter
"""
