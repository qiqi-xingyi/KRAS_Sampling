# --*-- conding:utf-8 --*--
# @time:1/5/26 23:25
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:build_tasks2.py


from __future__ import annotations
from pathlib import Path
import pandas as pd


# =========================
# Parameters (edit here)
# =========================
INFO_FILE = Path("dataset/QDockbank_info.txt")
USED_FILE = Path("dataset/tasks_used.csv")
OUT_FILE = Path("tasks2.csv")

# If True: remove a QDockbank row when either pdbid OR sequence has been used
STRICT_REMOVE_BY_PDBID_OR_SEQ = True

# Normalize sequences to uppercase for matching
NORMALIZE_SEQ_TO_UPPER = True


def _norm_pdbid(x: str) -> str:
    return str(x).strip().lower()


def _norm_seq(x: str) -> str:
    s = str(x).strip()
    if NORMALIZE_SEQ_TO_UPPER:
        s = s.upper()
    return s


def read_qdockbank_info(path: Path) -> pd.DataFrame:
    """
    Read QDockbank_info.txt which is tab-separated with header:
    pdb_id  Residue_sequence  ...
    """
    if not path.exists():
        raise FileNotFoundError(f"Info file not found: {path}")

    # Robust TSV read (handles extra spaces reasonably)
    df = pd.read_csv(path, sep="\t", dtype=str, engine="python")
    df.columns = [c.strip() for c in df.columns]

    required = {"pdb_id", "Residue_sequence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    df["pdb_id"] = df["pdb_id"].map(_norm_pdbid)
    df["Residue_sequence"] = df["Residue_sequence"].map(_norm_seq)

    # Drop empty rows
    df = df[df["pdb_id"].astype(bool) & df["Residue_sequence"].astype(bool)].copy()
    return df


def read_used_tasks(path: Path) -> tuple[set[str], set[str]]:
    """
    Read tasks_used.csv with header: pdbid,main_chain_residue_seq
    Return used_pdbids, used_seqs
    """
    if not path.exists():
        raise FileNotFoundError(f"Used tasks file not found: {path}")

    used = pd.read_csv(path, dtype=str)
    used.columns = [c.strip() for c in used.columns]

    required = {"pdbid", "main_chain_residue_seq"}
    missing = required - set(used.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    used["pdbid"] = used["pdbid"].map(_norm_pdbid)
    used["main_chain_residue_seq"] = used["main_chain_residue_seq"].map(_norm_seq)

    used = used[used["pdbid"].astype(bool) & used["main_chain_residue_seq"].astype(bool)].copy()

    used_pdbids = set(used["pdbid"].tolist())
    used_seqs = set(used["main_chain_residue_seq"].tolist())
    return used_pdbids, used_seqs


def main() -> None:
    info_df = read_qdockbank_info(INFO_FILE)
    used_pdbids, used_seqs = read_used_tasks(USED_FILE)

    # Remove used
    if STRICT_REMOVE_BY_PDBID_OR_SEQ:
        mask_keep = (~info_df["pdb_id"].isin(used_pdbids)) & (~info_df["Residue_sequence"].isin(used_seqs))
    else:
        # Only remove if both match (less strict, usually不建议)
        mask_keep = ~(info_df["pdb_id"].isin(used_pdbids) & info_df["Residue_sequence"].isin(used_seqs))

    remain = info_df.loc[mask_keep, ["pdb_id", "Residue_sequence"]].copy()

    # Remove duplicate sequences (keep first occurrence)
    remain = remain.drop_duplicates(subset=["Residue_sequence"], keep="first")

    # Output in tasks format
    out_df = pd.DataFrame({
        "pdbid": remain["pdb_id"],
        "main_chain_residue_seq": remain["Residue_sequence"],
    })

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_FILE, index=False)

    print("=== build_tasks2 completed ===")
    print(f"Info rows: {len(info_df)}")
    print(f"Used pdbids: {len(used_pdbids)}, used sequences: {len(used_seqs)}")
    print(f"Remain (after removing used): {len(remain)}")
    print(f"Output: {OUT_FILE}  (rows={len(out_df)})")


if __name__ == "__main__":
    main()
