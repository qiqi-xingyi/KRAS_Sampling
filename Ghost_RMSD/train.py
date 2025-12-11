# --*-- coding:utf-8 --*--
# @time: 11/01/25 03:53
# @Author : Yuqi Zhang
# @File   : train.py
#
# GRN training script:
# - Device selection (CUDA/MPS/CPU) happens before data/model build
# - RankNet + CE multitask training
# - Mixed early-stop monitor (alpha * NDCG@10 + (1 - alpha) * Spearman)
# - Per-epoch CSV/JSON logs + optional dumps of predictions for plotting

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from GRN.data import GRNDataModule
from GRN.model import build_grn_from_datamodule
from GRN.metrics import summarize_classification, summarize_ranking
from GRN.losses import build_pairs, ranknet_loss


# -------------------------------
# utils
# -------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device(arg_device: str | None):
    """Pick device by precedence: user arg -> CUDA -> MPS -> CPU."""
    if arg_device and arg_device != "auto":
        return torch.device(arg_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


def append_csv_row(csv_path: Path, row: Dict[str, float], header_written: bool):
    fieldnames = list(row.keys())
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not header_written:
            writer.writeheader()
        writer.writerow(row)


# -------------------------------
# eval / train
# -------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, return_details: bool = False):
    model.eval()
    all_logits, all_scores, all_labels, all_groups, all_rmsd = [], [], [], [], []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        out = model(x)
        all_logits.append(out["logits"].cpu().numpy())
        all_scores.append(out["score"].cpu().numpy())
        all_labels.append(batch["y"].cpu().numpy())
        all_groups.append(batch["group_id"].cpu().numpy())
        all_rmsd.append(batch["rmsd"].cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    groups = np.concatenate(all_groups, axis=0)
    rmsd = np.concatenate(all_rmsd, axis=0)

    cls = summarize_classification(logits, labels)
    rnk = summarize_ranking(logits, scores, labels, rmsd, groups, ks=(5, 10, 20))
    metrics = {**cls, **rnk}

    if return_details:
        details = {
            "logits": logits,
            "scores": scores,
            "labels": labels,
            "groups": groups,
            "rmsd": rmsd,
        }
        return metrics, details
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    ce_loss: nn.Module,
    lambda_ce: float,
    max_pairs_per_group: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        gid = batch["group_id"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        logits = out["logits"]
        scores = out["score"]

        # classification loss
        L_ce = ce_loss(logits, y)

        # ranknet loss
        pairs = build_pairs(gid.cpu(), y.cpu(), max_pairs_per_group=max_pairs_per_group)
        L_rank = ranknet_loss(scores, pairs.to(device)) if pairs.numel() > 0 else torch.zeros([], device=device)

        loss = L_rank + lambda_ce * L_ce
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs
    return total_loss / max(1, total_n)


def compute_class_weight(train_ds) -> torch.Tensor:
    y = train_ds.y.numpy()
    counts = np.bincount(y, minlength=4).astype(np.float64)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    w = inv * (counts.sum() / len(counts))
    return torch.tensor(w, dtype=torch.float32)


# -------------------------------
# main
# -------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="training_dataset")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")  # "auto", "cuda", "cuda:0", "cpu", "mps"
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--no_early_stop", action="store_true", default=False)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--score_mode", type=str, default="expected_rel",
                        choices=["prob_rel3", "logit_rel3", "expected_rel"])
    parser.add_argument("--lambda_ce", type=float, default=0.33)
    parser.add_argument("--max_pairs_per_group", type=int, default=128)
    parser.add_argument("--use_rank_head", action="store_true", default=True)
    parser.add_argument("--monitor_alpha", type=float, default=0.5,
                        help="weight for NDCG in early-stop mixed metric")
    parser.add_argument("--dump_predictions", action="store_true", default=False,
                        help="save validation/test predictions arrays for plotting")
    args = parser.parse_args()

    # seed
    set_seed(args.seed)

    # device selection BEFORE building data/model
    device = pick_device(args.device)
    if device.type == "cuda":
        print(f"[Device] Using GPU: {torch.cuda.get_device_name(0)} (build CUDA {torch.version.cuda})")
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    elif device.type == "mps":
        print("[Device] Using Apple MPS backend")
    else:
        print(f"[Device] Using CPU")

    # I/O
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_log = save_dir / "train_log.csv"
    json_log = save_dir / "train_log.json"
    header_written = csv_log.exists()
    history = []

    # data
    dm = GRNDataModule(data_dir=args.data_dir, batch_size=args.batch_size)
    dm.setup()

    # model
    model = build_grn_from_datamodule(
        dm,
        dropout=args.dropout,
        score_mode=args.score_mode,
        use_rank_head=args.use_rank_head,
    )
    model.to(device)

    # optim / loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    class_weight = compute_class_weight(dm.ds_train).to(device)
    ce_loss = nn.CrossEntropyLoss(weight=class_weight)

    # early stop and checkpoints
    best_monitor = -1.0
    best_epoch = -1
    patience_left = args.patience
    ckpt_best = save_dir / "grn_best.pt"
    ckpt_last = save_dir / "grn_last.pt"

    # training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, dm.train_dataloader(), optimizer, device,
            ce_loss, args.lambda_ce, args.max_pairs_per_group
        )
        val_metrics = evaluate(model, dm.valid_dataloader(), device)

        alpha = args.monitor_alpha
        mixed_monitor = alpha * val_metrics.get("ndcg@10", 0.0) + (1.0 - alpha) * val_metrics.get("spearman", 0.0)

        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} | "
              f"val_acc={val_metrics['acc']:.4f} | val_ndcg@10={val_metrics['ndcg@10']:.4f} | "
              f"val_spearman={val_metrics['spearman']:.4f} | monitor={mixed_monitor:.4f}")

        # log to CSV/JSON
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_acc": val_metrics["acc"],
            "val_ndcg@5": val_metrics["ndcg@5"],
            "val_ndcg@10": val_metrics["ndcg@10"],
            "val_ndcg@20": val_metrics["ndcg@20"],
            "val_spearman": val_metrics["spearman"],
            "monitor": mixed_monitor,
            "lr": optimizer.param_groups[0]["lr"],
            "lambda_ce": args.lambda_ce,
            "max_pairs_per_group": args.max_pairs_per_group,
            "dropout": args.dropout,
        }
        append_csv_row(csv_log, row, header_written=header_written)
        header_written = True
        history.append(row)
        with json_log.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        # checkpointing
        if mixed_monitor > best_monitor:
            best_monitor = mixed_monitor
            best_epoch = epoch
            patience_left = args.patience
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "base_feature_names": dm.base_feature_names,
                "seq_feature_names": dm.seq_feature_names,
                "scaler": dm.scaler,
                "args": vars(args),
                "val_metrics": val_metrics,
                "monitor": mixed_monitor,
            }, ckpt_best)
            print(f"  -> Saved new best to {ckpt_best}")
        else:
            patience_left -= 1

        # always save last checkpoint
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "args": vars(args),
        }, ckpt_last)

        # early stopping
        if not args.no_early_stop and patience_left <= 0:
            print(f"Early stopping at epoch {epoch}. Best epoch {best_epoch} (monitor={best_monitor:.4f}).")
            break

    # final test with best checkpoint
    if ckpt_best.exists():
        ckpt = torch.load(ckpt_best, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        test_metrics, test_details = evaluate(model, dm.test_dataloader(), device, return_details=True)
        print(f"[TEST] acc={test_metrics['acc']:.4f} | "
              f"ndcg@5={test_metrics['ndcg@5']:.4f} | ndcg@10={test_metrics['ndcg@10']:.4f} | "
              f"ndcg@20={test_metrics['ndcg@20']:.4f} | spearman={test_metrics['spearman']:.4f}")

        if args.dump_predictions:
            out_val_path = save_dir / "val_predictions.npz"
            out_test_path = save_dir / "test_predictions.npz"
            val_metrics, val_details = evaluate(model, dm.valid_dataloader(), device, return_details=True)
            np.savez_compressed(out_val_path, **val_details, **{"metrics": val_metrics})
            np.savez_compressed(out_test_path, **test_details, **{"metrics": test_metrics})
            print(f"[SAVED] {out_val_path} and {out_test_path}")
    else:
        print("No best checkpoint found; skip final test.")


if __name__ == "__main__":
    main()
