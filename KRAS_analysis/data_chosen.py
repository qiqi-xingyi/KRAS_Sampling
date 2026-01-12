# --*-- conding:utf-8 --*--
# @time:1/12/26 02:58
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:data_chosen.py


from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Tuple


# Closed-loop order (prefix numbers will follow this list)
FILES_USED_ORDERED: List[str] = [
    # 1) Pipeline definition / reproducibility
    "metrics.json",

    # 2) Global mutation sensitivity on quantum distributions
    "metrics_bootstrap.csv",
    "metrics_stability_by_seed.csv",

    # 3) Manifold + basin assignment (mother table)
    "merged_points_with_basin.csv",

    # 4) Basin reweighting + significance
    "basin_occupancy.csv",
    "basin_delta_summary.csv",
    "occupancy_delta_ci.csv",

    # 5) Mechanism inside key basins
    "basin_energy_contrast.csv",
    "key_basin_story.csv",
    "key_basin_story_ranked.csv",

    # 6) Structural interpretability (representative conformations)
    "representatives.csv",
    "rep_ca_rmsd.csv",

    # 7) Supporting descriptive statistics
    "basin_stats.csv",
]


def find_all_matches(root: Path, target_names: List[str]) -> Dict[str, List[Path]]:
    target_set = set(target_names)
    name_to_paths: Dict[str, List[Path]] = {n: [] for n in target_names}

    for p in root.rglob("*"):
        if p.is_file() and p.name in target_set:
            name_to_paths[p.name].append(p)

    for k in name_to_paths:
        name_to_paths[k] = sorted(name_to_paths[k])

    return name_to_paths


def choose_one(name: str, matches: List[Path]) -> Path:
    """
    Pick a single match deterministically.
    Preference: shortest relative path (closest to root), then lexicographic.
    """
    if len(matches) == 1:
        return matches[0]

    def key(p: Path) -> Tuple[int, str]:
        # shorter relative path = fewer parts
        return (len(p.parts), str(p))

    return sorted(matches, key=key)[0]


def main() -> int:
    project_root = Path(__file__).resolve().parent
    src_root = project_root / "data_summary"
    dst_root = project_root / "data_used"

    if not src_root.exists():
        raise FileNotFoundError(f"Source directory not found: {src_root}")

    dst_root.mkdir(parents=True, exist_ok=True)

    found = find_all_matches(src_root, FILES_USED_ORDERED)

    # Report missing and multi-matches
    missing = [n for n in FILES_USED_ORDERED if len(found.get(n, [])) == 0]
    multi = {n: paths for n, paths in found.items() if len(paths) > 1}

    copied = 0
    for idx, name in enumerate(FILES_USED_ORDERED, start=1):
        matches = found.get(name, [])
        if not matches:
            print(f"[MISS] {name}")
            continue

        src_path = choose_one(name, matches)
        prefix = f"{idx:02d}_"
        dst_path = dst_root / f"{prefix}{name}"

        # If destination exists, overwrite (keeps run idempotent)
        shutil.copy2(src_path, dst_path)
        copied += 1

        print(f"[COPY] {src_path.relative_to(src_root)} -> {dst_path.name}")

        if len(matches) > 1:
            print(f"       [NOTE] {name} had {len(matches)} matches; selected: {src_path.relative_to(src_root)}")

    print("\n==== Summary ====")
    print(f"Source:      {src_root}")
    print(f"Destination: {dst_root}")
    print(f"Requested:   {len(FILES_USED_ORDERED)} file names")
    print(f"Copied:      {copied} files")

    if missing:
        print("\n[WARN] Missing requested files:")
        for n in missing:
            print(f"  - {n}")

    if multi:
        print("\n[INFO] Duplicate-name matches detected (kept only ONE per name):")
        for n, paths in multi.items():
            print(f"  - {n}: {len(paths)} matches")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
