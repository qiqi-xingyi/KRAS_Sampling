# --*-- conding:utf-8 --*--
# @time:1/11/26 21:46
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:path_name.py

from pathlib import Path

if __name__ == '__main__':

    ROOT = Path(__file__).resolve().parent.parent  # tools/ -> repo root
    ANALYSIS = ROOT / "KRAS_sampling_results" / "analysis_closed_loop"

    print("ANALYSIS:", ANALYSIS)
    print("\n[Top-level files]")
    for p in sorted(ANALYSIS.glob("*")):
        if p.is_file():
            print(" -", p.name)

    print("\n[addons/ files]")
    for sub in ["addons", "addon"]:
        d = ANALYSIS / sub
        if d.exists():
            print("\nDIR:", d)
            for p in sorted(d.glob("**/*")):
                if p.is_file():
                    print(" -", p.relative_to(ANALYSIS))
        else:
            print("\nDIR missing:", d)
