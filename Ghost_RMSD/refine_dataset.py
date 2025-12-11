# --*-- conding:utf-8 --*--
# @time:11/1/25 02:36
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:refine_dataset.py

# -*- coding: utf-8 -*-
# Group all_examples.jsonl by pdb_id (protein)
# Add group_id (integer) and remove useless fields (e.g., fifth_bit)
# Reorder keys: group_id first, protein second

import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent
IN_FILE = ROOT / "training_data" / "all_examples.jsonl"
OUT_FILE = ROOT / "training_data" / "all_examples_grouped.jsonl"


def reorder_keys(d: dict, first_keys: list[str]) -> dict:
    """Reorder dict so that first_keys appear first, others keep original order."""
    ordered = {}
    for k in first_keys:
        if k in d:
            ordered[k] = d[k]
    for k, v in d.items():
        if k not in first_keys:
            ordered[k] = v
    return ordered


def main():
    if not IN_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {IN_FILE}")

    with IN_FILE.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    # Group by protein (pdb_id)
    groups = defaultdict(list)
    for rec in records:
        pdbid = rec.get("protein", "unknown")
        groups[pdbid].append(rec)

    # Sort pdbids alphabetically and assign group_id
    pdbids = sorted(groups.keys())
    group_map = {pdbid: idx + 1 for idx, pdbid in enumerate(pdbids)}

    out_lines = 0
    with OUT_FILE.open("w", encoding="utf-8") as fout:
        for pdbid in pdbids:
            gid = group_map[pdbid]
            for rec in groups[pdbid]:
                # Remove unused field
                rec.pop("fifth_bit", None)

                # Add group_id at front
                rec["group_id"] = gid
                rec = reorder_keys(rec, ["group_id", "protein"])

                fout.write(json.dumps(rec) + "\n")
                out_lines += 1

    print(f"[OK] Wrote grouped file: {OUT_FILE} ({out_lines} lines)")
    print(f"Groups assigned for {len(pdbids)} proteins")


if __name__ == "__main__":
    main()
