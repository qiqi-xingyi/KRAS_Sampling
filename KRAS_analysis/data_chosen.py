# --*-- conding:utf-8 --*--
# @time:1/12/26 02:58
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:data_chosen.py

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List


FILES_USED = [
    "metrics.json",
    "metrics_bootstrap.csv",
    "metrics_stability_by_seed.csv",
    "merged_points_with_basin.csv",
    "basin_occupancy.csv",
    "basin_delta_summary.csv",
    "occupancy_delta_ci.csv",
    "basin_energy_contrast.csv",
    "key_basin_story.csv",
    "key_basin_story_ranked.csv",
    "representatives.csv",
    "rep_ca_rmsd.csv",
    "basin_stats.csv",
]


def find_files(root: Path, target_names: List[str]) -> Dict[str, List[Path]]:
    name_to_paths: Dict[str, List[Path]] = {n: [] for n in target_names}
    target_set = set(target_names)

    for p in root.rglob("*"):
        if p.is_file() and p.name in target_set:
            name_to_paths[p.name].append(p)

    return name_to_paths


def copy_preserving_structure(src_root: Path, dst_root: Path, src_path: Path) -> Path:
    rel = src_path.relative_to(src_root)
    dst_path = dst_root / rel
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)
    return dst_path


def main() -> int:
    project_root = Path(__file__).resolve().parent
    src_root = project_root / "data_summary"
    dst_root = project_root / "data_used"

    if not src_root.exists():
        raise FileNotFoundError(f"Source directory not found: {src_root}")

    dst_root.mkdir(parents=True, exist_ok=True)

    found = find_files(src_root, FILES_USED)

    missing = [name for name, paths in found.items() if len(paths) == 0]
    multi = {name: paths for name, paths in found.items() if len(paths) > 1}

    copied_count = 0
    for name, paths in found.items():
        if not paths:
            continue

        # If multiple files share the same name, copy all of them (structure preserved).
        for src_path in sorted(paths):
            dst_path = copy_preserving_structure(src_root, dst_root, src_path)
            copied_count += 1
            print(f"[COPY] {src_path} -> {dst_path}")

    print("\n==== Summary ====")
    print(f"Source:      {src_root}")
    print(f"Destination: {dst_root}")
    print(f"Requested:   {len(FILES_USED)} file names")
    print(f"Copied:      {copied_count} files")

    if missing:
        print("\n[WARN] Missing requested files:")
        for n in missing:
            print(f"  - {n}")

    if multi:
        print("\n[INFO] Multiple matches found (copied all, preserving structure):")
        for n, paths in multi.items():
            print(f"  - {n}: {len(paths)} matches")
            for p in paths:
                print(f"      * {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
