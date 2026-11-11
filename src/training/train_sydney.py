# src/training/train_sydney.py
# Sydney 5개 항목 Grade(0~3) Multi-task ABMIL K-fold 학습 (make_sydney_splits.py split + TRIDENT 임베딩)
#
# 학습: 항목별 loss의 가중 평균 하나로 역전파한다. 항목별 attention·head는 자기 항목 loss로만,
#       shared adapter는 5개 항목의 gradient를 모두 받아 학습된다. 결측 Grade(-1)는 그 항목 loss에서만 빠진다.
# 추론: 항목별 예측 Grade를 split의 tasks 순서로 이어 붙인 것이 Sydney CODE (CODE를 하나의 클래스로 학습하지 않는다).
# --head ce:    Grade logit 4개 + cross entropy
# --head coral: 순서 임계값 logit 3개 + CORAL loss
#
# 결과 구조는 train.py와 같다. test_predictions.csv = slide_id, code_label, code_pred, <task>_label/_pred/_prob<g>
# 지표는 항목별 qwk/mae/within1/accuracy/balanced_accuracy/f1/auc + mean_qwk, mean_balanced_accuracy, code_exact
#
# 사용 예:
#   python -m src.training.train_sydney --split_file splits/sydney_k5_seed42.json \
#       --feat_dir /path/to/sydney/embedding/20x_224px_0px_overlap/features_hoptimus0 \
#       --head ce --out_dir outputs/sydney_v0.1.0/hoptimus0_ce --gpu 0

import argparse
import copy
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.dataset import H5BagDataset, feature_path, read_feature_info  # noqa: E402
from src.models.multitask_abmil import (HEADS, MultiTaskLoss, create_multitask_abmil,  # noqa: E402
                                        grade_probs)
from src.training.metrics import compute_ordinal_metrics  # noqa: E402
from src.training.train import logger, make_loader, set_seed, setup_logging  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="K-fold multi-task ABMIL for Sydney item grades")
    p.add_argument("--split_file", required=True, help="JSON from src/data/make_sydney_splits.py")
    p.add_argument("--feat_dir", required=True, help=".../20x_224px_0px_overlap/features_<encoder>")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--head", default="ce", choices=HEADS, help="ce = 4 grade logits, coral = 3 ordinal thresholds")
    p.add_argument("--model_kwargs", type=json.loads, default={},
                   help='JSON overrides of model defaults, e.g. \'{"embed_dim": 512, "dropout": 0.1}\'')
    p.add_argument("--task_weights", type=json.loads, default={},
                   help='JSON loss weight per task (default 1 each), e.g. \'{"hp": 2.0}\'')
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--bag_size", type=int, default=2000,
                   help="Patches sampled per slide per training step (val/test always use all patches). 0 = all")
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience on val loss (epochs)")
    p.add_argument("--min_delta", type=float, default=1e-3)
    p.add_argument("--lr_patience", type=int, default=5, help="ReduceLROnPlateau patience on val loss")
    p.add_argument("--class_weighted", action="store_true",
                   help="Inverse-frequency grade weights per task (computed on each fold's train set)")
    p.add_argument("--folds", type=int, nargs="+", default=None, help="Run only these folds (1-based)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--gpu", type=int, default=0, help="-1 = CPU")
    p.add_argument("--overwrite", action="store_true", help="Re-run folds that already have metrics.json")
    return p.parse_args()


def load_sydney_split(path):
    with open(path) as f:
        split = json.load(f)
    for key in ("tasks", "num_grades", "labels", "folds"):
        if key not in split:
            sys.exit(f"{path} has no '{key}'. Make Sydney splits with src/data/make_sydney_splits.py")
    return split


def to_code(grades) -> str:
    return "".join("x" if g < 0 else str(int(g)) for g in grades)


def run_epoch(model, loader, device, criterion, optimizer=None):
    """
    One pass over loader.
    Returns (mean total loss, {task: mean loss}, labels [N, T], probs {task: [N, K]}, preds {task: [N]}, slide_ids)
    """
    train = optimizer is not None
    model.train(train)
    total_loss, task_loss, task_n = 0.0, Counter(), Counter()
    labels, logits_all, slide_ids = [], {t: [] for t in model.tasks}, []

    with torch.set_grad_enabled(train):
        for features, label, slide_id in loader:
            features, label = features.to(device, non_blocking=True), label.to(device)  # label [B, T]
            logits, _ = model(features)
            loss, per_task = criterion(logits, label)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            for t, v in per_task.items():
                task_loss[t] += v
                task_n[t] += 1
            for t in model.tasks:
                logits_all[t].append(logits[t].detach().float())
            labels.append(label.cpu().numpy())
            slide_ids.extend(slide_id)

    probs, preds = {}, {}
    for t in model.tasks:
        p, pred = grade_probs(torch.cat(logits_all[t]), model.head)
        probs[t], preds[t] = p.cpu().numpy(), pred.cpu().numpy()
    task_loss = {t: task_loss[t] / task_n[t] for t in task_n}
    return total_loss / max(1, len(loader)), task_loss, np.concatenate(labels), probs, preds, slide_ids


def evaluate(tasks, labels, probs, preds):
    """Flat dict: <task>_<metric> for each task + mean_qwk, mean_balanced_accuracy, code_exact."""
    m = {}
    for i, t in enumerate(tasks):
        m.update({f"{t}_{k}": v for k, v in compute_ordinal_metrics(labels[:, i], probs[t], preds[t]).items()})
    m["mean_qwk"] = float(np.nanmean([m[f"{t}_qwk"] for t in tasks]))
    m["mean_balanced_accuracy"] = float(np.nanmean([m[f"{t}_balanced_accuracy"] for t in tasks]))
    # CODE 전체 일치: 결측이 아닌 항목이 모두 맞은 슬라이드 비율
    pred_mat = np.stack([preds[t] for t in tasks], axis=1)
    m["code_exact"] = float(((pred_mat == labels) | (labels < 0)).all(axis=1).mean())
    return m


def grade_weights(train_ids, labels, num_tasks, num_grades):
    """[T, K] inverse-frequency weights per task, ignoring missing grades."""
    y = np.array([labels[s] for s in train_ids])
    w = np.ones((num_tasks, num_grades))
    for i in range(num_tasks):
        counts = np.bincount(y[y[:, i] >= 0, i], minlength=num_grades).astype(float)
        w[i] = counts.sum() / (num_grades * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)


def train_fold(fold, split, args, in_dim, device):
    fold_dir = os.path.join(args.out_dir, f"fold{fold['fold']}")
    os.makedirs(fold_dir, exist_ok=True)
    tasks, num_grades, labels = split["tasks"], split["num_grades"], split["labels"]
    bag_size = args.bag_size if args.bag_size > 0 else None

    train_ds = H5BagDataset(fold["train"], labels, args.feat_dir, bag_size=bag_size)
    val_ds = H5BagDataset(fold["val"], labels, args.feat_dir, bag_size=None)
    test_ds = H5BagDataset(fold["test"], labels, args.feat_dir, bag_size=None)
    logger.info(f"Fold {fold['fold']}: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_loader = make_loader(train_ds, True, args.num_workers)
    val_loader = make_loader(val_ds, False, args.num_workers)
    test_loader = make_loader(test_ds, False, args.num_workers)

    set_seed(args.seed + fold["fold"])
    model, model_kwargs = create_multitask_abmil(in_dim, tasks, num_grades, args.head, **args.model_kwargs)
    model = model.to(device)

    cw = grade_weights(fold["train"], labels, len(tasks), num_grades) if args.class_weighted else None
    if cw is not None:
        logger.info("Grade weights: " + " | ".join(f"{t} " + "/".join(f"{w:.2f}" for w in cw[i].tolist())
                                                   for i, t in enumerate(tasks)))
    criterion = MultiTaskLoss(tasks, args.head, num_grades, class_weights=cw, task_weights=args.task_weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.lr_patience, factor=0.5)

    best = {"loss": float("inf"), "epoch": 0, "state": None, "val_metrics": None}
    history = []
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_tl, tr_y, tr_p, tr_pred, _ = run_epoch(model, train_loader, device, criterion, optimizer)
        va_loss, va_tl, va_y, va_p, va_pred, _ = run_epoch(model, val_loader, device, criterion)
        scheduler.step(va_loss)
        tr_m, va_m = evaluate(tasks, tr_y, tr_p, tr_pred), evaluate(tasks, va_y, va_p, va_pred)

        history.append({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                        "train_loss": tr_loss, "val_loss": va_loss,
                        **{f"train_{t}_loss": v for t, v in tr_tl.items()},
                        **{f"val_{t}_loss": v for t, v in va_tl.items()},
                        **{f"train_{k}": v for k, v in tr_m.items()},
                        **{f"val_{k}": v for k, v in va_m.items()}})

        improved = va_loss < best["loss"] - args.min_delta
        if improved:
            best = {"loss": va_loss, "epoch": epoch, "state": copy.deepcopy(model.state_dict()),
                    "val_metrics": {"loss": va_loss, **va_m}}
            bad_epochs = 0
        else:
            bad_epochs += 1

        task_qwk = " ".join(f"{t} {va_m[t + '_qwk']:.2f}" for t in tasks)
        logger.info(f"[fold {fold['fold']}] epoch {epoch:3d} | train loss {tr_loss:.4f} qwk {tr_m['mean_qwk']:.3f} | "
                    f"val loss {va_loss:.4f} qwk {va_m['mean_qwk']:.3f} bacc {va_m['mean_balanced_accuracy']:.3f} "
                    f"[{task_qwk}] | "
                    f"lr {optimizer.param_groups[0]['lr']:.1e} | {time.time() - t0:.0f}s{' *' if improved else ''}")

        if bad_epochs >= args.patience:
            logger.info(f"[fold {fold['fold']}] early stop at epoch {epoch} (best epoch {best['epoch']})")
            break

    pd.DataFrame(history).to_csv(os.path.join(fold_dir, "history.csv"), index=False)
    model.load_state_dict(best["state"])
    torch.save({
        "model_state_dict": best["state"],
        "model_name": "multitask_abmil",
        "model_kwargs": model_kwargs,
        "tasks": tasks,
        "num_grades": num_grades,
        "best_epoch": best["epoch"],
        "feat_dir": os.path.abspath(args.feat_dir),
        "split_file": os.path.abspath(args.split_file),
        "fold": fold["fold"],
    }, os.path.join(fold_dir, "best_model.pt"))

    te_loss, te_tl, te_y, te_p, te_pred, te_ids = run_epoch(model, test_loader, device, criterion)
    te_m = evaluate(tasks, te_y, te_p, te_pred)
    pred_mat = np.stack([te_pred[t] for t in tasks], axis=1)
    pred = pd.DataFrame({"slide_id": te_ids,
                         "code_label": [to_code(r) for r in te_y],
                         "code_pred": [to_code(r) for r in pred_mat]})
    for i, t in enumerate(tasks):
        pred[f"{t}_label"] = te_y[:, i]
        pred[f"{t}_pred"] = te_pred[t]
        for g in range(num_grades):
            pred[f"{t}_prob{g}"] = te_p[t][:, g]
    pred.to_csv(os.path.join(fold_dir, "test_predictions.csv"), index=False)

    result = {"fold": fold["fold"], "best_epoch": best["epoch"], "val": best["val_metrics"],
              "test": {"loss": te_loss, **{f"{t}_loss": v for t, v in te_tl.items()}, **te_m}}
    with open(os.path.join(fold_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"[fold {fold['fold']}] TEST mean_qwk={te_m['mean_qwk']:.4f} "
                f"mean_bacc={te_m['mean_balanced_accuracy']:.4f} code_exact={te_m['code_exact']:.4f}")
    for t in tasks:
        logger.info(f"[fold {fold['fold']}]   {t:<5} qwk={te_m[f'{t}_qwk']:.4f} bacc={te_m[f'{t}_balanced_accuracy']:.4f} "
                    f"mae={te_m[f'{t}_mae']:.4f} auc={te_m[f'{t}_auc']:.4f} n={te_m[f'{t}_n']}")
    return result


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    setup_logging(args.out_dir)
    set_seed(args.seed)

    split = load_sydney_split(args.split_file)
    tasks = split["tasks"]
    unknown = sorted(set(args.task_weights) - set(tasks))
    if unknown:
        sys.exit(f"--task_weights has unknown tasks {unknown}. Tasks: {tasks}")
    first_slide = split["folds"][0]["train"][0]
    info = read_feature_info(feature_path(args.feat_dir, first_slide))
    in_dim = info["dim"]
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")

    logger.info(f"Multi-task ABMIL ({args.head}) | encoder '{info['encoder']}' in_dim={in_dim} | "
                f"tasks {tasks} x {split['num_grades']} grades | device {device}")
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump({**vars(args), "model_name": "multitask_abmil", "in_dim": in_dim, "encoder": info["encoder"],
                   "tasks": tasks, "num_grades": split["num_grades"]}, f, indent=2, ensure_ascii=False)

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
        results.append(train_fold(fold, split, args, in_dim, device))

    rows = [{"fold": r["fold"], "best_epoch": r["best_epoch"], **r["test"]} for r in results]
    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if c not in ("fold", "best_epoch") and not c.endswith("_n")]
    summary = {c: {"mean": float(df[c].mean()), "std": float(df[c].std(ddof=0))} for c in metric_cols}
    pd.concat([df, pd.DataFrame([{"fold": "mean", **{c: summary[c]["mean"] for c in metric_cols}},
                                 {"fold": "std", **{c: summary[c]["std"] for c in metric_cols}}])]
              ).to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"folds": rows, "summary": summary}, f, indent=2)

    def ms(c):
        return f"{summary[c]['mean']:.4f} ± {summary[c]['std']:.4f}"

    logger.info("=" * 80)
    logger.info(f"multitask_abmil ({args.head}) | {info['encoder']} | {len(results)} folds (test)")
    logger.info(f"  {'task':<6} {'qwk':<17} {'balanced_acc':<17} {'mae':<17} auc")
    for t in tasks:
        logger.info(f"  {t:<6} {ms(f'{t}_qwk'):<17} {ms(f'{t}_balanced_accuracy'):<17} {ms(f'{t}_mae'):<17} {ms(f'{t}_auc')}")
    logger.info(f"  mean_qwk {ms('mean_qwk')} | mean_balanced_accuracy {ms('mean_balanced_accuracy')} | "
                f"code_exact {ms('code_exact')}")


if __name__ == "__main__":
    main()
