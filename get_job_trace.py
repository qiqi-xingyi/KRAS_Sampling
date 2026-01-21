# --*-- conding:utf-8 --*--
# @time:1/20/26 18:16
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:get_job_trace.py

from pathlib import Path
import shutil

# ---- Config (edit if needed) ----
SRC_ROOT = Path("QDock_sampling_results")
DST_ROOT = Path("QDock_sampling_results") / "_timing_progress_collected"
OVERWRITE = True
# --------------------------------

def extract_pdbid(folder_name: str) -> str:
    # Expected: KRAS_{pdbid}_{rep}, e.g., KRAS_1hdq_1
    if not folder_name.startswith("KRAS_"):
        return ""
    rest = folder_name[len("KRAS_"):]          # "1hdq_1"
    pdbid = rest.rsplit("_", 1)[0]             # "1hdq"
    return pdbid

def copy_one(src_dir: Path, dst_dir: Path) -> None:
    folder_name = src_dir.name
    pdbid = extract_pdbid(folder_name)
    if not pdbid:
        return

    timing_src = src_dir / f"{folder_name}_timing.csv"
    progress_src = src_dir / "progress.json"

    timing_dst = dst_dir / f"{pdbid}_timing.csv"
    progress_dst = dst_dir / f"{pdbid}_progress.json"

    missing = []
    if not timing_src.is_file():
        missing.append(str(timing_src))
    if not progress_src.is_file():
        missing.append(str(progress_src))
    if missing:
        print(f"[SKIP] {folder_name} (missing: {', '.join(missing)})")
        return

    dst_dir.mkdir(parents=True, exist_ok=True)

    if (not OVERWRITE) and (timing_dst.exists() or progress_dst.exists()):
        print(f"[SKIP] {folder_name} (destination exists)")
        return

    shutil.copy2(timing_src, timing_dst)
    shutil.copy2(progress_src, progress_dst)
    print(f"[OK] {folder_name} -> {timing_dst.name}, {progress_dst.name}")

def main():
    if not SRC_ROOT.is_dir():
        raise FileNotFoundError(f"SRC_ROOT not found: {SRC_ROOT.resolve()}")


    DST_ROOT.mkdir(parents=True, exist_ok=True)

    count = 0
    for sub in sorted(SRC_ROOT.iterdir()):
        if sub.is_dir() and sub.name.startswith("KRAS_"):
            copy_one(sub, DST_ROOT)
            count += 1

    print(f"\nDone. Scanned {count} KRAS_* folders.")
    print(f"Output: {DST_ROOT.resolve()}")

if __name__ == "__main__":
    main()
