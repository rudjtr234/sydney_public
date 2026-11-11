# src/training/train.py
# K-fold MIL 학습 (TRIDENT 임베딩 + make_splits.py split JSON)
#
# fold마다: train(bag_size 랜덤 샘플) -> val loss 기준 early stopping -> best weight로 test(전체 패치) 평가
# 결과: <out_dir>/config.json, fold{k}/{best_model.pt,history.csv,test_predictions.csv,metrics.json},
#       summary.csv / summary.json (fold별 test 지표 + mean/std)
#
# 사용 예:
#   python -m src.training.train --model_name abmil \
#       --split_file splits/label_k5_seed42.json \
#       --feat_dir /path/to/sydney/embedding/20x_224px_0px_overlap/features_hoptimus0 \
#       --out_dir outputs/hoptimus0_abmil_v0.1.0

import argparse
import copy
import json
import logging
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.dataset import H5BagDataset, feature_path, load_split_file, read_feature_info  # noqa: E402
from src.models import create_mil_model, get_available_models, normalize_model_name  # noqa: E402
from src.training.metrics import compute_metrics  # noqa: E402

logger = logging.getLogger("train")


def parse_args():
    p = argparse.ArgumentParser(description="K-fold MIL training on TRIDENT patch embeddings")
    p.add_argument("--split_file", required=True, help="JSON from src/data/make_splits.py")
    p.add_argument("--feat_dir", required=True, help=".../20x_224px_0px_overlap/features_<encoder>")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--model_name", default="abmil", choices=get_available_models() + ["clam"])
    p.add_argument("--model_kwargs", type=json.loads, default={},
                   help='JSON overrides of model defaults, e.g. \'{"dropout": 0.1, "embed_dim": 256}\'')
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--bag_size", type=int, default=2000,
                   help="Patches sampled per slide per training step (val/test always use all patches). 0 = all")
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience on val loss (epochs)")
    p.add_argument("--min_delta", type=float, default=1e-3)
    p.add_argument("--lr_patience", type=int, default=5, help="ReduceLROnPlateau patience on val loss")
    p.add_argument("--class_weighted", action="store_true", help="Inverse-frequency class weights in the loss")
    p.add_argument("--folds", type=int, nargs="+", default=None, help="Run only these folds (1-based)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--gpu", type=int, default=0, help="-1 = CPU")
    p.add_argument("--overwrite", action="store_true", help="Re-run folds that already have metrics.json")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(out_dir: str):
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(out_dir, "train.log"))):
        h.setFormatter(fmt)
        logger.addHandler(h)


def make_loader(dataset, shuffle, num_workers):
    return DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=num_workers, pin_memory=True,
                      persistent_workers=num_workers > 0)


def run_epoch(model, loader, device, loss_fn, optimizer=None):
    """One pass over loader. Returns (mean loss, labels, probs [N, C], slide_ids)."""
    train = optimizer is not None
    model.train(train)
    total_loss, labels, probs, slide_ids = 0.0, [], [], []

    with torch.set_grad_enabled(train):
        for features, label, slide_id in loader:
            features, label = features.to(device, non_blocking=True), label.to(device)
            results, _ = model(h=features, loss_fn=loss_fn, label=label)
            loss = results["loss"]
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            probs.append(torch.softmax(results["logits"].detach().float(), dim=1).cpu().numpy())
            labels.append(label.cpu().numpy())
            slide_ids.extend(slide_id)

    return total_loss / max(1, len(loader)), np.concatenate(labels), np.concatenate(probs), slide_ids


def class_weights(train_ids, labels, num_classes):
    counts = np.bincount([labels[s] for s in train_ids], minlength=num_classes).astype(float)
    weights = counts.sum() / (num_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def train_fold(fold, split, args, in_dim, num_classes, device):
    fold_dir = os.path.join(args.out_dir, f"fold{fold['fold']}")
    os.makedirs(fold_dir, exist_ok=True)
    labels = split["labels"]
    bag_size = args.bag_size if args.bag_size > 0 else None

    train_ds = H5BagDataset(fold["train"], labels, args.feat_dir, bag_size=bag_size)
    val_ds = H5BagDataset(fold["val"], labels, args.feat_dir, bag_size=None)
    test_ds = H5BagDataset(fold["test"], labels, args.feat_dir, bag_size=None)
    logger.info(f"Fold {fold['fold']}: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_loader = make_loader(train_ds, True, args.num_workers)
    val_loader = make_loader(val_ds, False, args.num_workers)
    test_loader = make_loader(test_ds, False, args.num_workers)

    set_seed(args.seed + fold["fold"])
    model, model_kwargs = create_mil_model(args.model_name, in_dim=in_dim, num_classes=num_classes, **args.model_kwargs)
    model = model.to(device)

    weight = class_weights(fold["train"], labels, num_classes).to(device) if args.class_weighted else None
    # batch_size=1: reduction='mean'은 sum(w*l)/sum(w) = l이라 class weight가 사라진다 -> 'sum'
    loss_fn = nn.CrossEntropyLoss(weight=weight, reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.lr_patience, factor=0.5)

    best = {"loss": float("inf"), "epoch": 0, "state": None, "val_metrics": None}
    history = []
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_y, tr_p, _ = run_epoch(model, train_loader, device, loss_fn, optimizer)
        va_loss, va_y, va_p, _ = run_epoch(model, val_loader, device, loss_fn)
        scheduler.step(va_loss)
        tr_m, va_m = compute_metrics(tr_y, tr_p), compute_metrics(va_y, va_p)

        history.append({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                        "train_loss": tr_loss, "val_loss": va_loss,
                        **{f"train_{k}": v for k, v in tr_m.items()},
                        **{f"val_{k}": v for k, v in va_m.items()}})

        improved = va_loss < best["loss"] - args.min_delta
        if improved:
            best = {"loss": va_loss, "epoch": epoch, "state": copy.deepcopy(model.state_dict()),
                    "val_metrics": {"loss": va_loss, **va_m}}
            bad_epochs = 0
        else:
            bad_epochs += 1

        logger.info(f"[fold {fold['fold']}] epoch {epoch:3d} | train loss {tr_loss:.4f} auc {tr_m['auc']:.4f} | "
                    f"val loss {va_loss:.4f} auc {va_m['auc']:.4f} bacc {va_m['balanced_accuracy']:.4f} | "
                    f"lr {optimizer.param_groups[0]['lr']:.1e} | {time.time() - t0:.0f}s{' *' if improved else ''}")

        if bad_epochs >= args.patience:
            logger.info(f"[fold {fold['fold']}] early stop at epoch {epoch} (best epoch {best['epoch']})")
            break

    pd.DataFrame(history).to_csv(os.path.join(fold_dir, "history.csv"), index=False)
    model.load_state_dict(best["state"])
    torch.save({
        "model_state_dict": best["state"],
        "model_name": normalize_model_name(args.model_name),
        "model_kwargs": model_kwargs,
        "label_map": split["label_map"],
        "best_epoch": best["epoch"],
        "feat_dir": os.path.abspath(args.feat_dir),
        "split_file": os.path.abspath(args.split_file),
        "fold": fold["fold"],
    }, os.path.join(fold_dir, "best_model.pt"))

    te_loss, te_y, te_p, te_ids = run_epoch(model, test_loader, device, loss_fn)
    te_m = compute_metrics(te_y, te_p)
    pred = pd.DataFrame({"slide_id": te_ids, "label": te_y, "pred": te_p.argmax(axis=1)})
    for c, i in split["label_map"].items():
        pred[f"prob_{c}"] = te_p[:, i]
    pred.to_csv(os.path.join(fold_dir, "test_predictions.csv"), index=False)

    result = {"fold": fold["fold"], "best_epoch": best["epoch"], "val": best["val_metrics"],
              "test": {"loss": te_loss, **te_m}}
    with open(os.path.join(fold_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"[fold {fold['fold']}] TEST " + " ".join(f"{k}={v:.4f}" for k, v in te_m.items()
                                                         if isinstance(v, float)))
    return result


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    setup_logging(args.out_dir)
    set_seed(args.seed)

    split = load_split_file(args.split_file)
    num_classes = len(split["label_map"])
    first_slide = split["folds"][0]["train"][0]
    info = read_feature_info(feature_path(args.feat_dir, first_slide))
    in_dim = info["dim"]
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")

    logger.info(f"Model {args.model_name} | encoder '{info['encoder']}' in_dim={in_dim} | "
                f"classes {split['label_map']} | device {device}")
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump({**vars(args), "in_dim": in_dim, "encoder": info["encoder"], "num_classes": num_classes,
                   "label_map": split["label_map"]}, f, indent=2, ensure_ascii=False)

    results = []
    for fold in split["folds"]:
        if args.folds and fold["fold"] not in args.folds:
            continue
        done = os.path.join(args.out_dir, f"fold{fold['fold']}", "metrics.json")
        if os.path.exists(done) and not args.overwrite:
            logger.info(f"[fold {fold['fold']}] already done, skipping (use --overwrite to re-run)")
            with open(done) as f:
                results.append(json.load(f))
            continue
        results.append(train_fold(fold, split, args, in_dim, num_classes, device))

    rows = [{"fold": r["fold"], "best_epoch": r["best_epoch"], **r["test"]} for r in results]
    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if c not in ("fold", "best_epoch", "tp", "tn", "fp", "fn")]
    summary = {c: {"mean": float(df[c].mean()), "std": float(df[c].std(ddof=0))} for c in metric_cols}
    pd.concat([df, pd.DataFrame([{"fold": "mean", **{c: summary[c]["mean"] for c in metric_cols}},
                                 {"fold": "std", **{c: summary[c]["std"] for c in metric_cols}}])]
              ).to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"folds": rows, "summary": summary}, f, indent=2)

    logger.info("=" * 80)
    logger.info(f"{args.model_name} | {info['encoder']} | {len(results)} folds (test)")
    for c in ("auc", "balanced_accuracy", "accuracy", "f1"):
        logger.info(f"  {c:<18} {summary[c]['mean']:.4f} ± {summary[c]['std']:.4f}")


if __name__ == "__main__":
    main()
