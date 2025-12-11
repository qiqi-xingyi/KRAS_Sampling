# plot_grn_results.py
# Standalone visualization script for GRN training and prediction results.
# Generates:
#   - Training curves (loss, acc, ndcg, spearman, monitor) for first 500 epochs
#   - Validation/test score–normalized-RMSD scatter plots (per-group min-max normalization)
# All figures are saved under checkpoints_full/figs/

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ====== Configurable paths ======
BASE_DIR = Path(__file__).resolve().parent
LOG_CSV = BASE_DIR / "checkpoints_full/train_log.csv"
VAL_NPZ = BASE_DIR / "checkpoints_full/val_predictions.npz"
TEST_NPZ = BASE_DIR / "checkpoints_full/test_predictions.npz"
OUT_DIR = BASE_DIR / "checkpoints_full/figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
EPOCH_LIMIT = 500
# =================================


def read_train_log_csv(csv_path: Path):
    rows = []
    if not csv_path.exists():
        raise FileNotFoundError(f"train_log.csv not found: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: float(v) if k != "epoch" else int(v) for k, v in r.items()})
    return rows


def spearmanr(a: np.ndarray, b: np.ndarray) -> float:
    """Simple Spearman correlation implementation."""
    def rankdata(x):
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(x), dtype=np.float64)
        vals = x[order]
        i = 0
        while i < len(vals):
            j = i + 1
            while j < len(vals) and vals[j] == vals[i]:
                j += 1
            if j - i > 1:
                avg = (i + j - 1) / 2.0
                ranks[order[i:j]] = avg
            i = j
        return ranks

    ra = rankdata(a)
    rb = rankdata(b)
    ra = (ra - ra.mean()) / (ra.std() + 1e-12)
    rb = (rb - rb.mean()) / (rb.std() + 1e-12)
    return float(np.clip(np.corrcoef(ra, rb)[0, 1], -1.0, 1.0))


def scatter_with_fit(x, y, title, xlabel, ylabel, out_path: Path):
    plt.figure(figsize=(6, 5))
    hb = plt.hexbin(x, y, gridsize=60, mincnt=1, cmap="viridis")
    cb = plt.colorbar(hb)
    cb.set_label("count")

    if len(x) >= 2:
        A = np.vstack([x, np.ones_like(x)]).T
        w, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = w[0] * x_fit + w[1]
        plt.plot(x_fit, y_fit, color="red", linewidth=2)

    rho = spearmanr(x, y)
    plt.title(f"{title}\nSpearman={rho:.4f}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=600)
    plt.close()


def plot_training_curves(rows, out_dir: Path, epoch_limit: int = 500):
    # trim to first `epoch_limit` epochs by index order
    rows = rows[:epoch_limit] if len(rows) > epoch_limit else rows

    epochs = np.array([r["epoch"] for r in rows])
    train_loss = np.array([r["train_loss"] for r in rows])
    val_acc = np.array([r["val_acc"] for r in rows])
    ndcg5 = np.array([r["val_ndcg@5"] for r in rows])
    ndcg10 = np.array([r["val_ndcg@10"] for r in rows])
    ndcg20 = np.array([r["val_ndcg@20"] for r in rows])
    spearman = np.array([r["val_spearman"] for r in rows])
    monitor = np.array([r["monitor"] for r in rows])


    # Training loss
    plt.figure(figsize=(6, 3))
    plt.plot(epochs, train_loss, label="train_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "loss.png", dpi=600)
    plt.close()

    # Validation accuracy
    plt.figure(figsize=(6, 3))
    plt.plot(epochs, val_acc, label="val_acc", color="orange")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Validation Accuracy")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "val_acc.png", dpi=600)
    plt.close()

    # Ranking metrics
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, ndcg5, label="ndcg@5")
    plt.plot(epochs, ndcg10, label="ndcg@10")
    plt.plot(epochs, ndcg20, label="ndcg@20")
    plt.plot(epochs, spearman, label="spearman")
    plt.xlabel("epoch")
    plt.ylabel("metric")
    plt.title("Ranking Metrics")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "ranking_metrics.png", dpi=600)
    plt.close()

    # Mixed monitor
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, monitor, label="monitor", color="green")
    plt.xlabel("epoch")
    plt.ylabel("monitor")
    plt.title("Mixed Monitor")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "monitor.png", dpi=600)
    plt.close()


def normalize_rmsd_per_group(rmsd: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Min-max normalize RMSD within each group to [0,1]."""
    rmsd = rmsd.astype(float).copy()
    out = np.empty_like(rmsd, dtype=float)
    uniq = np.unique(groups)
    for g in uniq:
        idx = (groups == g)
        r = rmsd[idx]
        rmin = float(np.min(r))
        rmax = float(np.max(r))
        out[idx] = (r - rmin) / (rmax - rmin + 1e-12)
    return out


def plot_prediction_scatter(npz_path: Path, tag: str, out_dir: Path):
    if not npz_path.exists():
        print(f"[WARN] {tag} predictions not found, skip.")
        return
    data = np.load(npz_path, allow_pickle=True)
    scores = data["scores"].astype(float).ravel()
    rmsd = data["rmsd"].astype(float).ravel()
    groups = data.get("groups", None)
    if groups is None:
        print(f"[WARN] `{tag}` predictions missing `groups`; skip normalization.")
        norm_rmsd = rmsd
    else:
        groups = groups.astype(int).ravel()
        norm_rmsd = normalize_rmsd_per_group(rmsd, groups)

    scatter_with_fit(
        x=scores,
        y=norm_rmsd,
        title=f"{tag} set: Predicted score vs Normalized RMSD (per-group)",
        xlabel="Predicted score (higher is better)",
        ylabel="Normalized RMSD (per group)",
        out_path=out_dir / f"{tag.lower()}_score_vs_rmsd.png",
    )


def main():
    print("=== GRN Visualization ===")
    rows = read_train_log_csv(LOG_CSV)
    plot_training_curves(rows, OUT_DIR, epoch_limit=EPOCH_LIMIT)
    plot_prediction_scatter(VAL_NPZ, "Validation", OUT_DIR)
    plot_prediction_scatter(TEST_NPZ, "Test", OUT_DIR)
    print(f"[DONE] All figures saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
