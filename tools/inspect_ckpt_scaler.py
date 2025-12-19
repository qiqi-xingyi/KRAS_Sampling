# --*-- conding:utf-8 --*--
# @time:12/18/25 23:04
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:inspect_ckpt_scaler.py

# tools/inspect_ckpt_scaler.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List

ENERGY_KEYS: List[str] = [
    "E_total", "E_steric", "E_geom", "E_bond", "E_mj",
    "E_dihedral", "E_hb", "E_hydroph", "E_cbeta", "E_rama",
]


def _as_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _get_mean_std(v: Any) -> Optional[Tuple[float, float]]:
    """
    Try to interpret one scaler entry as (mean, std).

    Supported formats:
      - {"mean": ..., "std": ...}
      - {"mu": ..., "sigma": ...}
      - {"center": ..., "scale": ...}
      - [mean, std] or (mean, std)
      - {"mean_": ..., "scale_": ...}  (some sklearn-like exports)
    """
    if isinstance(v, dict):
        # common keys
        if "mean" in v and "std" in v:
            mu = _as_float(v.get("mean"))
            sd = _as_float(v.get("std"))
            if mu is not None and sd is not None:
                return mu, sd
        if "mu" in v and "sigma" in v:
            mu = _as_float(v.get("mu"))
            sd = _as_float(v.get("sigma"))
            if mu is not None and sd is not None:
                return mu, sd
        if "center" in v and "scale" in v:
            mu = _as_float(v.get("center"))
            sd = _as_float(v.get("scale"))
            if mu is not None and sd is not None:
                return mu, sd

        # sklearn-ish naming
        if "mean_" in v and "scale_" in v:
            mu = _as_float(v.get("mean_"))
            sd = _as_float(v.get("scale_"))
            if mu is not None and sd is not None:
                return mu, sd

    if isinstance(v, (list, tuple)) and len(v) == 2:
        mu = _as_float(v[0])
        sd = _as_float(v[1])
        if mu is not None and sd is not None:
            return mu, sd

    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        type=str,
        default="Ghost_RMSD/checkpoints_full/grn_best.pt",
        help="Path to the GRN checkpoint (.pt).",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="tools/parsed_scaler.json",
        help="Path to write parsed scaler as JSON.",
    )
    ap.add_argument(
        "--print_entries",
        type=int,
        default=8,
        help="How many scaler entries to preview.",
    )
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path.resolve()}")

    try:
        import torch  # noqa: F401
    except Exception as e:
        raise RuntimeError(f"Failed to import torch: {e}")

    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu")

    print("=== Checkpoint scaler inspection ===")
    print(f"ckpt path: {ckpt_path.resolve()}")
    print(f"ckpt keys: {sorted(list(ckpt.keys()))}")

    scaler = ckpt.get("scaler", None)
    if scaler is None:
        print("scaler: None (no scaler saved in checkpoint).")
        return

    print(f"scaler type: {type(scaler)}")

    if not isinstance(scaler, dict):
        print("scaler is not a dict; cannot parse per-feature stats.")
        return

    print(f"scaler dict keys: {sorted(list(scaler.keys()))}")

    # Preview entries
    print("\n=== scaler entry preview ===")
    preview_keys = list(scaler.keys())[: max(0, int(args.print_entries))]
    for k in preview_keys:
        v = scaler[k]
        s = str(v)
        if len(s) > 180:
            s = s[:180] + "..."
        print(f"- {k}: type={type(v)} value={s}")

    # Attempt to parse mean/std per ENERGY_KEYS
    parsed: Dict[str, Dict[str, float]] = {}
    ok = True
    for ek in ENERGY_KEYS:
        if ek not in scaler:
            ok = False
            print(f"[FAIL] scaler missing expected key: {ek}")
            continue
        ms = _get_mean_std(scaler[ek])
        if ms is None:
            ok = False
            print(f"[FAIL] cannot parse mean/std for key={ek} (type={type(scaler[ek])}) value={scaler[ek]}")
            continue
        mu, sd = ms
        if sd == 0.0:
            sd = 1.0
        parsed[ek] = {"mean": float(mu), "std": float(sd)}

    if not ok:
        print("\n[RESULT] Scaler dict exists but is not in a mean/std-friendly format.")
        print("Action: paste one full scaler entry (e.g., scaler['E_total']) and we can implement an exact parser.")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
    print(f"\n[OK] Parsed scaler saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
