# --*-- conding:utf-8 --*--
# @time:11/5/25 12:17
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:plt_GRN.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export a clean architecture diagram of Ghost RMSD Net (GRN) using torchview.
Generates vector graphics: GRN_structure.pdf and GRN_structure.svg

Assumptions:
- Your GRN implementation is available as GRNClassifier in model.py
- If you also have a GRNDataModule to compute feature_dim(), you can enable AUTO_DIM.
"""

import os
import sys
from pathlib import Path

import torch
from torch import nn

try:
    from torchview import draw_graph
except ImportError as e:
    raise SystemExit("torchview not found. Install via: pip install torchview") from e

# === User config ===
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_IMPORT_PATH = PROJECT_ROOT / "model.py"  # ensure model.py exists in the same folder
OUT_BASENAME = "GRN_structure"                 # will create GRN_structure.pdf / .svg
AUTO_DIM = False                               # set True if you have a DataModule to detect feature_dim()

# Preferred hidden dims and flags (must match your training)
HIDDEN_DIMS = [256, 128]
DROPOUT = 0.30
USE_BN = True
USE_RANK_HEAD = True

# Fallback manual input dimension (when AUTO_DIM=False)
FALLBACK_IN_DIM = 285  # e.g., base features + sequence stats

# Try optional DataModule import for auto in_dim (edit as needed)
def infer_in_dim_via_datamodule():
    """
    Optional: If you have a GRNDataModule like in your data.py, use it to infer feature_dim().
    Adjust paths/imports accordingly.
    """
    try:
        from data import GRNDataModule  # your datamodule
        dm = GRNDataModule(data_dir="training_dataset")
        dm.setup()
        return dm.feature_dim()
    except Exception:
        return None

def build_model(in_dim: int) -> nn.Module:
    """
    Import and instantiate your GRNClassifier from model.py
    """
    try:
        # Ensure the parent path is on sys.path
        sys.path.insert(0, str(PROJECT_ROOT))
        from model import GRNClassifier  # your class
    except Exception as e:
        raise SystemExit("Cannot import GRNClassifier from model.py. "
                         "Make sure model.py is in the same directory and defines GRNClassifier.") from e

    model = GRNClassifier(
        in_dim=in_dim,
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT,
        use_bn=USE_BN,
        score_mode="expected_rel",
        use_rank_head=USE_RANK_HEAD,
    )
    return model

def main():
    # 1) Decide input feature dimension
    if AUTO_DIM:
        auto_dim = infer_in_dim_via_datamodule()
        in_dim = auto_dim if isinstance(auto_dim, int) and auto_dim > 0 else FALLBACK_IN_DIM
    else:
        in_dim = FALLBACK_IN_DIM

    # 2) Build model and dummy input
    model = build_model(in_dim)
    model.eval()  # deterministic graph

    # TorchView supports input_data or input_size. Use input_data for clarity.
    x = torch.randn(1, in_dim)

    # 3) Create the architecture graph
    # expand_nested=True will expand nn.Sequential / submodules for clarity
    graph = draw_graph(
        model,
        input_data=x,
        expand_nested=True,
        depth=3,             # increase if you want to go deeper into nested modules
        graph_name="Ghost_RMSD_Net",
        roll=True            # compact repeated modules
    )

    # 4) Styling (via underlying graphviz Digraph)
    G = graph.visual_graph
    # Global graph attrs
    G.graph_attr.update({
        "rankdir": "TB",     # top-to-bottom; use "LR" for left-to-right
        "splines": "ortho",  # orthogonal edges for readability
        "fontsize": "10",
        "bgcolor": "white"
    })
    # Node style
    G.node_attr.update({
        "shape": "box",
        "style": "rounded,filled",
        "color": "#555555",
        "fillcolor": "#F8F9FB",
        "fontname": "Helvetica",
        "fontsize": "9"
    })
    # Edge style
    G.edge_attr.update({
        "color": "#555555",
        "arrowsize": "0.7"
    })

    # 5) Render outputs (vector graphics: PDF and SVG)
    out_pdf = graph.visual_graph.render(filename=OUT_BASENAME, format="pdf", cleanup=True)
    out_svg = graph.visual_graph.render(filename=OUT_BASENAME, format="svg", cleanup=True)

    print(f"[OK] Saved: {out_pdf}")
    print(f"[OK] Saved: {out_svg}")
    print("Tip: include in LaTeX via \\includegraphics[width=0.9\\linewidth]{GRN_structure.pdf}")

if __name__ == "__main__":
    main()
