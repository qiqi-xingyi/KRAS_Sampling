# --*-- conding:utf-8 --*--
# @time:1/6/26 22:33
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:kras_analysis_addon_2.py

# tools/kras_analysis_addon.py
# --*-- coding:utf-8 --*--
"""
Add-on script to supplement closed-loop KRAS sampling analysis.

Outputs (under <analysis_dir>/addon/):
  - basin_sample_sizes.csv        : n, sum_w, n_eff per (label, basin_id)
  - basin_sample_sizes_by_basin.csv: pivot view by basin
  - basin_tests.csv               : KS + Wasserstein tests for key basins and metrics
  - exported_pdb/                 : representative CA-trace PDBs (WT vs G12C/G12D) for key basins

This script is robust to different file names by searching the analysis directory.

Assumptions:
  - analysis_dir contains representatives.csv
  - analysis_dir contains a merged table (merged.csv / pooled.csv / something with columns below),
    OR decoded.jsonl paths are available in representatives/merged and we can fetch main_positions.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional scipy for stats; we provide fallbacks if not available.
try:
    from scipy.stats import ks_2samp, wasserstein_distance
except Exception:
    ks_2samp = None  # type: ignore
    wasserstein_distance = None  # type: ignore


AA3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    "X": "GLY"
}


# -----------------------------
# Utilities
# -----------------------------

def _read_jsonl_line(path: Path, line_index: int) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == line_index:
                return json.loads(line)
    raise IndexError(f"line_index {line_index} out of range for {path}")


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _find_first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _find_analysis_files(analysis_dir: Path) -> Dict[str, Optional[Path]]:
    """
    Try to locate key files in analysis_dir.
    """
    reps = analysis_dir / "representatives.csv"
    metrics = analysis_dir / "metrics.json"

    # Find a "merged" table. We search for CSVs likely to contain merged fields.
    candidate_csvs = sorted(analysis_dir.glob("*.csv"))
    merged_like = []
    for p in candidate_csvs:
        name = p.name.lower()
        if "merged" in name or "pooled" in name or "samples" in name:
            merged_like.append(p)

    # If not found by name, we will try any csv with expected columns.
    return {
        "representatives": reps if reps.exists() else None,
        "metrics": metrics if metrics.exists() else None,
        "merged_guess": merged_like[0] if merged_like else (candidate_csvs[0] if candidate_csvs else None),
    }


def _load_best_merged_table(analysis_dir: Path) -> Optional[pd.DataFrame]:
    """
    Search CSV files under analysis_dir and pick the one that looks like the merged sample table.
    Expected useful columns:
      - label, basin_id
      - E_total and/or backbone_rmsd
      - weight or p_mass
      - (optional) decoded_file/line_index or main_positions/sequence/scale_factor
    """
    csvs = sorted(analysis_dir.glob("*.csv"))
    if not csvs:
        return None

    best_score = -1
    best_df: Optional[pd.DataFrame] = None
    best_path: Optional[Path] = None

    wanted = {
        "label", "basin_id", "E_total", "backbone_rmsd", "weight", "p_mass",
        "decoded_file", "line_index", "bitstring", "sequence", "scale_factor",
        "main_positions"
    }

    for p in csvs:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        cols = set(df.columns)
        score = len(cols.intersection(wanted))
        # Heuristic: must contain label or fragment_id
        if "label" not in cols and "fragment_id" not in cols:
            continue
        if score > best_score:
            best_score = score
            best_df = df
            best_path = p

    if best_df is None:
        return None

    # Attach source path for debugging
    best_df.attrs["_source_path"] = str(best_path) if best_path else ""
    return best_df


def _pick_weight_column(df: pd.DataFrame) -> Optional[str]:
    for c in ["weight", "p_mass", "prob", "p", "w"]:
        if c in df.columns:
            return c
    return None


def _effective_n(weights: np.ndarray) -> float:
    """
    Kish effective sample size:
      n_eff = (sum_w)^2 / sum(w^2)
    Works for unnormalized weights too.
    """
    w = np.asarray(weights, dtype=float)
    sw = float(np.sum(w))
    if sw <= 0:
        return 0.0
    denom = float(np.sum(w * w))
    if denom <= 0:
        return 0.0
    return (sw * sw) / denom


def _ks_test(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Return (ks_stat, p_value). If scipy not available, returns (stat, nan).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return (float("nan"), float("nan"))

    if ks_2samp is not None:
        r = ks_2samp(x, y, alternative="two-sided", mode="auto")
        return (float(r.statistic), float(r.pvalue))

    # Fallback: compute KS statistic only
    xs = np.sort(x)
    ys = np.sort(y)
    allv = np.sort(np.unique(np.concatenate([xs, ys])))
    # Empirical CDFs
    cdfx = np.searchsorted(xs, allv, side="right") / len(xs)
    cdfy = np.searchsorted(ys, allv, side="right") / len(ys)
    stat = float(np.max(np.abs(cdfx - cdfy)))
    return (stat, float("nan"))


def _wasserstein(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    if wasserstein_distance is not None:
        return float(wasserstein_distance(x, y))
    # Fallback: approximate by quantile L1 distance
    qs = np.linspace(0, 1, 101)
    xq = np.quantile(x, qs)
    yq = np.quantile(y, qs)
    return float(np.mean(np.abs(xq - yq)))


def _coerce_list(obj: Any) -> Optional[List[Any]]:
    if obj is None:
        return None
    if isinstance(obj, list):
        return obj
    if isinstance(obj, str):
        s = obj.strip()
        # Might be JSON string
        if s.startswith("[") and s.endswith("]"):
            try:
                return json.loads(s)
            except Exception:
                return None
    return None


def _extract_positions_from_row(row: pd.Series) -> Optional[List[List[float]]]:
    """
    Try to obtain main_positions from a merged row.
    """
    if "main_positions" in row.index:
        mp = _coerce_list(row["main_positions"])
        if mp is not None and len(mp) > 0:
            return mp  # type: ignore
    return None


def _resolve_decoded_entry(
    merged_df: Optional[pd.DataFrame],
    label: str,
    fragment_id: str,
    bitstring: str,
    decoded_file: Optional[str],
    line_index: Optional[int],
) -> Tuple[Optional[Dict[str, Any]], Optional[Path]]:
    """
    Best-effort: locate decoded entry either from merged_df fields or by reading decoded.jsonl.
    """
    # 1) If merged_df exists and has matching row with positions, use it.
    if merged_df is not None:
        df = merged_df
        # try match by label+bitstring (most stable)
        mask = pd.Series([True] * len(df))
        if "label" in df.columns:
            mask &= (df["label"].astype(str) == str(label))
        if "bitstring" in df.columns:
            mask &= (df["bitstring"].astype(str) == str(bitstring))
        if mask.any():
            r = df.loc[mask].iloc[0]
            mp = _extract_positions_from_row(r)
            if mp is not None:
                entry = {"sequence": r.get("sequence", None), "main_positions": mp, "scale_factor": r.get("scale_factor", None)}
                return entry, None

        # fallback: fragment_id + bitstring
        mask = pd.Series([True] * len(df))
        if "fragment_id" in df.columns:
            mask &= (df["fragment_id"].astype(str) == str(fragment_id))
        if "bitstring" in df.columns:
            mask &= (df["bitstring"].astype(str) == str(bitstring))
        if mask.any():
            r = df.loc[mask].iloc[0]
            mp = _extract_positions_from_row(r)
            if mp is not None:
                entry = {"sequence": r.get("sequence", None), "main_positions": mp, "scale_factor": r.get("scale_factor", None)}
                return entry, None

    # 2) Read decoded_file jsonl
    if decoded_file:
        dpath = Path(decoded_file)
        if dpath.exists():
            # if line_index is provided, read directly
            if line_index is not None and line_index >= 0:
                try:
                    entry = _read_jsonl_line(dpath, int(line_index))
                    return entry, dpath
                except Exception:
                    pass
            # otherwise scan by bitstring
            try:
                for e in _iter_jsonl(dpath):
                    if str(e.get("bitstring", "")) == str(bitstring):
                        return e, dpath
            except Exception:
                pass

    # 3) Try common location from fragment_id (pp_result/<frag>/<inner>/decoded.jsonl)
    # This is a best guess; only used if user runs from project root.
    guess = Path("pp_result") / fragment_id
    if guess.exists():
        for p in guess.glob("*/decoded.jsonl"):
            try:
                for e in _iter_jsonl(p):
                    if str(e.get("bitstring", "")) == str(bitstring):
                        return e, p
            except Exception:
                continue

    return None, None


def _positions_to_pdb(
    out_path: Path,
    sequence: str,
    main_positions: List[List[float]],
    scale_factor: float = 1.0,
    chain_id: str = "A",
):
    """
    Write CA-trace PDB. main_positions may have length = len(sequence)+1 (an extra anchor point).
    We will try to align lengths.
    """
    seq = sequence or ""
    L = len(seq)
    coords = np.asarray(main_positions, dtype=float)

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("main_positions must be Nx3")

    # handle extra point
    if L > 0:
        if len(coords) == L + 1:
            coords = coords[1:]  # drop anchor
        elif len(coords) == L:
            pass
        else:
            # If length mismatch, try take first L
            coords = coords[:L]
    else:
        # no sequence; keep all
        L = len(coords)
        seq = "X" * L

    coords = coords * float(scale_factor)

    lines = []
    atom_serial = 1
    res_seq = 1
    for i in range(L):
        aa = seq[i]
        resname = AA3.get(aa.upper(), "GLY")
        x, y, z = coords[i]
        line = (
            f"ATOM  {atom_serial:5d}  CA  {resname:>3s} {chain_id}{res_seq:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C"
        )
        lines.append(line)
        atom_serial += 1
        res_seq += 1

    lines.append("END")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------
# Core logic
# -----------------------------

def compute_basin_sample_sizes(merged: pd.DataFrame) -> pd.DataFrame:
    if "basin_id" not in merged.columns:
        raise KeyError("merged table has no 'basin_id' column")

    if "label" not in merged.columns:
        # Try fallback: some tables may use 'set' or 'variant'
        for alt in ["variant", "set", "group", "case"]:
            if alt in merged.columns:
                merged = merged.rename(columns={alt: "label"})
                break
    if "label" not in merged.columns:
        raise KeyError("merged table has no 'label' (or variant/set) column")

    wcol = _pick_weight_column(merged)
    df = merged.copy()
    if wcol is None:
        df["_w"] = 1.0
        wcol = "_w"

    rows = []
    for (lab, bid), g in df.groupby(["label", "basin_id"], dropna=False):
        w = g[wcol].to_numpy(dtype=float)
        rows.append({
            "label": str(lab),
            "basin_id": int(bid) if (pd.notna(bid) and str(bid).isdigit()) else bid,
            "n": int(len(g)),
            "sum_w": float(np.sum(w)),
            "n_eff": float(_effective_n(w)),
        })
    out = pd.DataFrame(rows).sort_values(["label", "basin_id"]).reset_index(drop=True)
    return out


def run_basin_tests(
    merged: pd.DataFrame,
    basins: List[int],
    metrics: List[str],
    pairs: List[Tuple[str, str]],
) -> pd.DataFrame:
    """
    For each basin and metric, compare distributions between label pairs.
    """
    if "basin_id" not in merged.columns or "label" not in merged.columns:
        raise KeyError("merged must include basin_id and label")

    rows = []
    for bid in basins:
        m_b = merged.loc[merged["basin_id"] == bid]
        for m in metrics:
            if m not in merged.columns:
                continue
            for a, b in pairs:
                xa = m_b.loc[m_b["label"] == a, m].to_numpy()
                xb = m_b.loc[m_b["label"] == b, m].to_numpy()
                ks_stat, ks_p = _ks_test(xa, xb)
                wd = _wasserstein(xa, xb)
                rows.append({
                    "basin_id": int(bid),
                    "metric": m,
                    "pair": f"{a}-vs-{b}",
                    "n_a": int(np.isfinite(xa).sum()),
                    "n_b": int(np.isfinite(xb).sum()),
                    "ks_stat": ks_stat,
                    "ks_pvalue": ks_p,
                    "wasserstein": wd,
                    "mean_a": float(np.nanmean(xa)) if len(xa) else float("nan"),
                    "mean_b": float(np.nanmean(xb)) if len(xb) else float("nan"),
                    "delta_mean_a_minus_b": float(np.nanmean(xa) - np.nanmean(xb)) if (len(xa) and len(xb)) else float("nan"),
                })
    return pd.DataFrame(rows).sort_values(["basin_id", "metric", "pair"]).reset_index(drop=True)


def export_representative_pdbs(
    analysis_dir: Path,
    merged: Optional[pd.DataFrame],
    basins: List[int],
    labels: List[str],
    pair_targets: List[Tuple[str, str]],
):
    """
    Export representative CA-trace PDB for each basin and label using representatives.csv.
    Priorities:
      - Use representatives.csv rows (within_label preferred); fallback to global.
      - Fetch main_positions from merged row if available; otherwise from decoded_file/line_index.
    """
    reps_path = analysis_dir / "representatives.csv"
    if not reps_path.exists():
        raise FileNotFoundError(f"missing {reps_path}")

    reps = pd.read_csv(reps_path)

    required = {"basin_id", "label", "fragment_id", "bitstring"}
    if not required.issubset(set(reps.columns)):
        raise KeyError(f"representatives.csv missing required columns: {sorted(required - set(reps.columns))}")

    out_dir = analysis_dir / "addon" / "exported_pdb"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pick preferred rows: within_label > global
    def _pick_rep_row(basin_id: int, label: str) -> Optional[pd.Series]:
        r = reps.loc[(reps["basin_id"] == basin_id) & (reps["label"] == label)]
        if len(r) == 0:
            return None
        if "scope" in r.columns:
            w = r.loc[r["scope"].astype(str) == "within_label"]
            if len(w) > 0:
                return w.iloc[0]
            g = r.loc[r["scope"].astype(str) == "global"]
            if len(g) > 0:
                return g.iloc[0]
        return r.iloc[0]

    # For each basin, export PDB for the requested labels
    for bid in basins:
        for lab in labels:
            rr = _pick_rep_row(int(bid), str(lab))
            if rr is None:
                continue

            fragment_id = str(rr["fragment_id"])
            bitstring = str(rr["bitstring"])
            decoded_file = str(rr["decoded_file"]) if "decoded_file" in rr.index and pd.notna(rr["decoded_file"]) else None
            line_index = int(rr["line_index"]) if "line_index" in rr.index and pd.notna(rr["line_index"]) else None

            entry, _ = _resolve_decoded_entry(
                merged_df=merged,
                label=str(lab),
                fragment_id=fragment_id,
                bitstring=bitstring,
                decoded_file=decoded_file,
                line_index=line_index,
            )
            if entry is None:
                continue

            seq = str(entry.get("sequence") or rr.get("sequence") or "")
            mp = entry.get("main_positions", None)
            if mp is None:
                continue

            scale_factor = None
            for cand in [entry.get("scale_factor", None), rr.get("scale_factor", None)]:
                if cand is not None and not (isinstance(cand, float) and math.isnan(cand)):
                    scale_factor = float(cand)
                    break
            if scale_factor is None:
                # default: your pipeline often uses 3.8 Å scale
                scale_factor = 3.8

            out_path = out_dir / f"basin{bid:02d}_{lab}.pdb"
            _positions_to_pdb(out_path, seq, mp, scale_factor=scale_factor, chain_id="A")

    # Also export pair-named convenience copies for key pairs
    # (WT-vs-G12D etc) in key basins to make it easier to screenshot.
    for bid in basins:
        for a, b in pair_targets:
            pa = out_dir / f"basin{bid:02d}_{a}.pdb"
            pb = out_dir / f"basin{bid:02d}_{b}.pdb"
            if pa.exists() and pb.exists():
                (out_dir / f"PAIR_basin{bid:02d}_{a}_vs_{b}.txt").write_text(
                    f"Load these two PDBs and align by CA:\n  {pa}\n  {pb}\n",
                    encoding="utf-8"
                )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--analysis_dir",
        type=str,
        default="KRAS_sampling_results/analysis_closed_loop",
        help="Directory of closed-loop analysis outputs."
    )
    ap.add_argument(
        "--basins",
        type=str,
        default="1,2,5,6",
        help="Comma-separated basin ids to analyze/export (default: union basins 1,2,5,6)."
    )
    ap.add_argument(
        "--metrics",
        type=str,
        default="E_total,backbone_rmsd",
        help="Comma-separated metrics to test (KS/Wasserstein)."
    )
    args = ap.parse_args()

    analysis_dir = Path(args.analysis_dir).expanduser().resolve()
    if not analysis_dir.exists():
        raise FileNotFoundError(f"analysis_dir not found: {analysis_dir}")

    basins = [int(x) for x in args.basins.split(",") if x.strip() != ""]
    metrics = [x.strip() for x in args.metrics.split(",") if x.strip() != ""]

    out_dir = analysis_dir / "addon"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load merged table if we can find it
    merged = _load_best_merged_table(analysis_dir)
    if merged is None:
        print("[WARN] No merged-like CSV table found. Will still export PDBs if decoded_file info exists in representatives.")
    else:
        src = merged.attrs.get("_source_path", "")
        print(f"[OK] Loaded merged table from: {src} | shape={merged.shape}")

    # 1) basin sample sizes
    if merged is not None and "basin_id" in merged.columns and ("label" in merged.columns or any(c in merged.columns for c in ["variant", "set", "group", "case"])):
        sizes = compute_basin_sample_sizes(merged)
        sizes_path = out_dir / "basin_sample_sizes.csv"
        sizes.to_csv(sizes_path, index=False)

        # pivot view
        piv = sizes.pivot_table(index="basin_id", columns="label", values=["n", "sum_w", "n_eff"], aggfunc="first")
        piv_path = out_dir / "basin_sample_sizes_by_basin.csv"
        piv.to_csv(piv_path)
        print(f"[OK] Wrote: {sizes_path}")
        print(f"[OK] Wrote: {piv_path}")
    else:
        print("[WARN] Skip basin_sample_sizes because merged lacks required columns.")

    # 2) basin distribution tests
    if merged is not None and "basin_id" in merged.columns and "label" in merged.columns:
        # Only run tests for labels present
        labels_present = sorted(merged["label"].astype(str).unique())
        pairs = []
        for a, b in [("WT", "G12C"), ("WT", "G12D")]:
            if a in labels_present and b in labels_present:
                pairs.append((a, b))
        tests = run_basin_tests(merged, basins=basins, metrics=metrics, pairs=pairs)
        tests_path = out_dir / "basin_tests.csv"
        tests.to_csv(tests_path, index=False)
        print(f"[OK] Wrote: {tests_path}")
    else:
        print("[WARN] Skip basin_tests because merged lacks required columns.")

    # 3) export representative PDBs
    # Use representatives.csv labels if available; otherwise export for WT/G12C/G12D.
    reps_path = analysis_dir / "representatives.csv"
    if reps_path.exists():
        reps = pd.read_csv(reps_path)
        labels = sorted(reps["label"].astype(str).unique()) if "label" in reps.columns else ["WT", "G12C", "G12D"]
        export_representative_pdbs(
            analysis_dir=analysis_dir,
            merged=merged,
            basins=basins,
            labels=labels,
            pair_targets=[("WT", "G12C"), ("WT", "G12D")]
        )
        print(f"[OK] Exported PDBs under: {analysis_dir / 'addon' / 'exported_pdb'}")
    else:
        print("[WARN] Skip PDB export because representatives.csv not found.")

    # Lightweight README
    readme = out_dir / "README_addon.txt"
    readme.write_text(
        "Add-on outputs:\n"
        "  - basin_sample_sizes.csv: n/sum_w/n_eff per (label, basin)\n"
        "  - basin_sample_sizes_by_basin.csv: pivot table\n"
        "  - basin_tests.csv: KS + Wasserstein per basin/metric/pair\n"
        "  - exported_pdb/: CA-trace PDB for basin representatives\n",
        encoding="utf-8"
    )
    print(f"[OK] Wrote: {readme}")


if __name__ == "__main__":
    main()
