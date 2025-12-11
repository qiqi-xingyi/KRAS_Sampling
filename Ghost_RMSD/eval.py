# --*-- conding:utf-8 --*--
# @time:11/1/25 03:53
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:eval.py

# grn_simple/eval.py
import argparse
from pathlib import Path
import numpy as np
import torch

from GRN.data import GRNDataModule
from GRN.model import build_grn_from_datamodule
from GRN.metrics import summarize_classification, summarize_ranking


@torch.no_grad()
def run_eval(model: GRNClassifier, loader, device: torch.device):
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
    return {**cls, **rnk}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/grn_best.pt")
    parser.add_argument("--data_dir", type=str, default="training_dataset")
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)

    # Rebuild minimal datamodule with saved scaler/feature order
    dm = GRNDataModule(data_dir=args.data_dir, batch_size=1024)
    dm.setup()
    dm.feature_names = ckpt["feature_names"]
    dm.scaler = ckpt["scaler"]

    model = GRNClassifier(in_dim=len(dm.feature_names))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    if args.split == "train":
        loader = dm.train_dataloader()
    elif args.split == "valid":
        loader = dm.valid_dataloader()
    else:
        loader = dm.test_dataloader()

    metrics = run_eval(model, loader, device)
    print(f"[{args.split.upper()}] "
          f"acc={metrics['acc']:.4f} | "
          f"ndcg@5={metrics['ndcg@5']:.4f} | ndcg@10={metrics['ndcg@10']:.4f} | "
          f"ndcg@20={metrics['ndcg@20']:.4f} | spearman={metrics['spearman']:.4f}")


if __name__ == "__main__":
    main()
