# src/training/train_sydney_mtl.py
# SydneyMTL (Kim et al., Histopathology 2026, doi:10.1111/his.70258) 학습 절차 재현
#
# 공개 코드 github.com/SMF-AI/SydneyMTL (commit f015f5c) experiments/train_model.py, trainer.py, datasets.py와 같은 설정:
#   모델   src/models/sydney_mtl.py (shared adaptor + 항목별 attention/head), 항목마다 4등급
#   손실   항목별 logit-adjusted CE (tau 0.75, prior = fold train 라벨 분포) 의 합. 비교용 ce, wce
#   학습   Adam lr 1e-4 (weight decay 없음), batch 1 + gradient accumulation 8, 최대 100 epoch,
#          val total loss가 10 epoch 연속 안 줄면 멈추고 가장 좋았던 가중치로 되돌림. 패치는 샘플링 없이 전부 사용
#   분할   --split code (기본, 논문): 5자리 CODE 문자열로 StratifiedKFold(5, shuffle, seed 42) -> test.
#          나머지의 12.5%를 val (CODE로 층화, 1건뿐인 CODE가 있어 안 되면 무작위)
#          --split multilabel (v0.3.0~): 항목별 등급(5×4) + 생검 부위(4)를 함께 맞추는 iterative stratification
#          (Sechidis 2011)으로 test 5등분, 같은 방법으로 나머지에서 val 12.5%. 가장 드문 칸(HP 3등급 등)부터
#          사례를 그 칸이 가장 모자란 fold에 넣는다. python random(seed)만 써서 서버·numpy 버전이 달라도 같은 분할.
#   지표   항목별 accuracy, QWK, lenient accuracy (+ mae, within1, balanced accuracy, f1, auc)
# 우리 데이터에 맞춘 부분:
#   - 사례(case_id) 1개 = bag 1개. 슬라이드가 여러 장인 사례는 슬라이드 특징을 이어 붙인다 (라벨이 사례 단위라서)
#   - 라벨 = 메타데이터 CSV의 --label_col (code_review 또는 code_init), exclude_recommended 사례는 뺀다
#   - CODE 자리 순서 = HP-NEU-MONO-ATR-IM. 위축 N/A 등급은 이 데이터에 없어 모든 항목 4등급
#   - 특징은 처음에 한 번 메모리로 읽어 둔다 (fold·epoch마다 NFS를 다시 읽지 않도록)
# 생검 부위 입력 (--site_mode, 논문에 없는 확장, 기본 none = 논문 그대로):
#   - 부위 = 메타데이터 region (antrum / body / angle / 나머지는 other). 사례마다 부위가 하나
#   - head: 항목별 점수에 부위별 보정값, film: 패치 특징을 부위에 따라 조정 (src/models/sydney_mtl.py)
#   - 학습 중 --site_dropout 비율의 사례는 부위를 '모름'(0)으로 넣는다 → 추론 때 부위가 없어도 쓸 수 있게
#   - 부위 보정값은 0에서 시작해서 lr 1e-4로는 거의 안 움직인다 → --site_lr로 따로 준다
#   - test는 부위를 넣은 결과(test_predictions.csv)와 부위 모름 결과(test_predictions_nosite.csv)를 둘 다 남긴다
# 버전 관리:
#   - 결과 폴더 = <output_root>/sydney_mtl_<encoder>_<배율>x_v<version> (배율은 feat_dir의 20x_224px_... 폴더 이름에서).
#     이미 있는 버전을 다른 설정·코드·라벨로 돌리면 멈춘다
#   - 학습에 쓴 코드를 결과 폴더 code/에 복사하고, 코드·메타데이터 CSV의 md5를 config.json에 남긴다
#   - 끝나면 MLflow (experiment "sydney")에 자동으로 올린다 (src/training/mlflow_sydney.py, --no_mlflow로 끔)
#
# 사용 예:
#   python -m src.training.train_sydney_mtl --version 0.1.0 --meta_csv metadata/sydney_meta.csv \
#       --label_col code_review --label_source "CRF v2" --loss la --gpu 0 \
#       --feat_dir /path/to/sydney/embedding/hibou_l/20x_224px_0px_overlap/features_hibou_l
#   -> outputs/sydney_mtl_hibou_l_20x_v0.1.0/

import argparse
import copy
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.sydney_mtl import LOSSES, SITE_MODES, GradeLoss, SydneyMTL  # noqa: E402
from src.training.metrics import compute_ordinal_metrics, lenient_accuracy  # noqa: E402
from src.training.plot_sydney_mtl import plot_curves, plot_fold, plot_run  # noqa: E402
from src.training.train import logger, set_seed, setup_logging  # noqa: E402

TASKS = ["hp", "neu", "mono", "atr", "im"]
NUM_GRADES = 4
SITES = ["antrum", "body", "angle", "other"]  # site 번호 = 1 + 순서, 0 = 부위 모름
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE_FILES = ["src/models/sydney_mtl.py", "src/training/train_sydney_mtl.py", "src/training/metrics.py",
              "src/training/plot_sydney_mtl.py", "src/training/mlflow_sydney.py"]
# 같은 버전 번호로 다시 돌릴 때 같아야 하는 설정 (gpu, folds 같은 실행 옵션은 제외)
VERSION_KEYS = ["meta_csv_md5", "label_col", "feat_dir", "loss", "tau", "adaptor_dim", "epochs", "patience", "lr",
                "accumulation_steps", "kfold", "val_size", "seed", "split_method", "code_md5", "site_mode",
                "site_dropout", "site_lr"]
# 부위 옵션이 생기기 전 config에는 키가 없다 → 그때 동작 (부위 안 씀)으로 비교
VERSION_DEFAULTS = {"site_mode": "none", "site_dropout": 0.0, "site_lr": None}
MLFLOW_PYTHON = os.environ.get("MLFLOW_PYTHON", sys.executable)  # mlflow 가 설치된 인터프리터


def parse_args():
    p = argparse.ArgumentParser(description="SydneyMTL training on patch embeddings")
    p.add_argument("--meta_csv", required=True, help="metadata/sydney_meta.csv")
    p.add_argument("--label_col", required=True, choices=["code_review", "code_init"])
    p.add_argument("--feat_dir", required=True, help=".../<mag>x_224px_0px_overlap/features_<encoder>")
    p.add_argument("--version", required=True,
                   help="Result version x.y.z -> <output_root>/sydney_mtl_<encoder>_<mag>x_v<x.y.z>")
    p.add_argument("--output_root", default=os.path.join(PROJECT_DIR, "outputs"))
    p.add_argument("--label_source", default="", help='Free text for MLflow, e.g. "CRF v2"')
    p.add_argument("--loss", default="la", choices=LOSSES)
    p.add_argument("--tau", type=float, default=0.75, help="logit adjustment temperature (paper code default)")
    p.add_argument("--adaptor_dim", type=int, default=256)
    p.add_argument("--site_mode", default="none", choices=SITE_MODES,
                   help="biopsy site input: none (paper), head (per-site logit offset), film (per-site feature scaling)")
    p.add_argument("--site_dropout", type=float, default=0.2,
                   help="Fraction of training bags fed with site unknown (only with --site_mode head/film)")
    p.add_argument("--site_lr", type=float, default=None,
                   help="Adam lr for the site parameters only (default: same as --lr)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--accumulation_steps", type=int, default=8)
    p.add_argument("--kfold", type=int, default=5)
    p.add_argument("--val_size", type=float, default=0.125)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", default="code", choices=["code", "multilabel"],
                   help="code = paper (stratify on the 5-digit CODE string); "
                        "multilabel = balance every task grade and biopsy site across folds")
    p.add_argument("--folds", type=int, nargs="+", default=None, help="Run only these folds (1-based)")
    p.add_argument("--load_workers", type=int, default=8, help="Threads for reading h5 files")
    p.add_argument("--gpu", type=int, default=0, help="-1 = CPU")
    p.add_argument("--overwrite", action="store_true", help="Re-run folds that already have metrics.json")
    p.add_argument("--no_mlflow", action="store_true", help="Do not upload to MLflow at the end")
    p.add_argument("--mlflow_python", default=MLFLOW_PYTHON, help="Python with mlflow (sydney env has none)")
    args = p.parse_args()
    if not all(x.isdigit() for x in args.version.split(".")) or args.version.count(".") != 2:
        p.error(f"--version must look like 0.1.0, got '{args.version}'")
    args.encoder = os.path.basename(os.path.normpath(args.feat_dir)).removeprefix("features_")
    m = re.match(r"(\d+(?:\.\d+)?)x_", os.path.basename(os.path.dirname(os.path.normpath(args.feat_dir))))
    if not m:
        p.error(f"--feat_dir must be inside a TRIDENT <mag>x_224px_0px_overlap folder, got '{args.feat_dir}'")
    args.mag = m.group(1)
    if args.site_mode == "none":
        args.site_dropout, args.site_lr = 0.0, None  # 부위를 안 쓰면 의미 없는 값이라 config에 기본값으로 남긴다
    args.out_dir = os.path.join(args.output_root, f"sydney_mtl_{args.encoder}_{args.mag}x_v{args.version}")
    return args


def md5(paths):
    h = hashlib.md5()
    for path in paths:
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def check_version(out_dir, run_cfg):
    """Stop if this version folder was made with different settings, code or labels (bump --version instead)."""
    path = os.path.join(out_dir, "config.json")
    if not os.path.exists(path):
        return
    with open(path) as f:
        old = json.load(f)
    diff = {k: (old.get(k, VERSION_DEFAULTS.get(k)), run_cfg[k]) for k in VERSION_KEYS
            if old.get(k, VERSION_DEFAULTS.get(k)) != run_cfg[k]}
    if diff:
        sys.exit(f"{out_dir} already holds a run with different settings {diff}. Use a new --version.")


def upload_mlflow(args):
    cmd = [args.mlflow_python, "-m", "src.training.mlflow_sydney", "--out_dir", args.out_dir]
    try:
        out = subprocess.run(cmd, cwd=PROJECT_DIR, capture_output=True, text=True, timeout=1800)
        logger.info((out.stdout.strip().splitlines() or [""])[-1])
        if out.returncode != 0:
            raise RuntimeError(out.stderr.strip().splitlines()[-1] if out.stderr.strip() else out.returncode)
    except Exception as e:  # noqa: BLE001  업로드 실패가 학습 결과를 망치지 않도록
        logger.info(f"MLflow upload failed ({e}). Retry: cd {PROJECT_DIR} && {' '.join(cmd)}")


def load_cases(meta_csv, label_col):
    """Rows kept for training, one per case: case_id, slide_ids (sorted), code, grades [5], site (1-4, SITES)."""
    meta = pd.read_csv(meta_csv, dtype=str)
    meta = meta[meta["exclude_recommended"] != "True"]
    cases = []
    for case_id, g in meta.groupby("case_id", sort=True):
        codes = g[label_col].unique()
        if len(codes) != 1 or not (len(codes[0]) == 5 and set(codes[0]) <= set("0123")):
            sys.exit(f"{case_id}: {label_col} must be one 5-digit 0-3 code per case, got {list(codes)}")
        region = g["region"].iloc[0]
        cases.append({"case_id": case_id, "slide_ids": sorted(g["slide_id"]), "code": codes[0],
                      "grades": [int(c) for c in codes[0]],
                      "site": 1 + (SITES.index(region) if region in SITES[:-1] else len(SITES) - 1)})
    return cases


def load_bags(cases, feat_dir, workers):
    """case_id -> float32 tensor [N, D] (slides of a case concatenated)."""
    def read(slide_id):
        with h5py.File(os.path.join(feat_dir, f"{slide_id}.h5"), "r") as f:
            return f["features"][:]

    slide_ids = [s for c in cases for s in c["slide_ids"]]
    missing = [s for s in slide_ids if not os.path.exists(os.path.join(feat_dir, f"{s}.h5"))]
    if missing:
        sys.exit(f"{len(missing)} feature files missing in {feat_dir}, e.g. {missing[:3]}")
    with ThreadPoolExecutor(workers) as ex:
        feats = dict(zip(slide_ids, ex.map(read, slide_ids)))
    return {c["case_id"]: torch.from_numpy(np.concatenate([feats[s] for s in c["slide_ids"]]).astype(np.float32))
            for c in cases}


def _iterative_stratify(Y, ratios, rng):
    """Iterative stratification (Sechidis et al. 2011). Y: rows x labels (0/1), ratios: share per part.
    Takes the rarest label among unassigned rows and gives each of its rows to the part that still wants that
    label most (ties: the part that wants the most rows, then random). Returns the part index of every row."""
    Y = np.asarray(Y, dtype=np.int64)
    assert (Y.sum(1) > 0).all(), "every row needs at least one label"
    desired = np.outer(ratios, Y.sum(0)).astype(float)  # parts x labels still wanted
    left = np.asarray(ratios, dtype=float) * len(Y)      # rows still wanted per part
    part = -np.ones(len(Y), dtype=int)
    while (part < 0).any():
        rem = part < 0
        cnt = Y[rem].sum(0).astype(float)
        cnt[cnt == 0] = np.inf
        lab = int(np.argmin(cnt))
        rows = [int(i) for i in np.flatnonzero(rem & (Y[:, lab] == 1))]
        rng.shuffle(rows)
        for i in rows:
            cand = np.flatnonzero(desired[:, lab] == desired[:, lab].max())
            cand = cand[left[cand] == left[cand].max()]
            k = int(cand[rng.randrange(len(cand))])
            part[i] = k
            desired[k] -= Y[i]
            left[k] -= 1
    return part


def make_folds_multilabel(cases, kfold, val_size, seed):
    """Balance every task grade (5 x 4 one-hot) and biopsy site (4) across test folds, then across train/val."""
    Y = np.zeros((len(cases), len(TASKS) * NUM_GRADES + len(SITES)), dtype=np.int64)
    for i, c in enumerate(cases):
        for j, g in enumerate(c["grades"]):
            Y[i, j * NUM_GRADES + g] = 1
        Y[i, len(TASKS) * NUM_GRADES + c["site"] - 1] = 1
    rng = random.Random(seed)  # not numpy: Generator streams may change between numpy versions (서버 간)
    part = _iterative_stratify(Y, [1 / kfold] * kfold, rng)
    folds = []
    for k in range(kfold):
        test, train_val = np.flatnonzero(part == k), np.flatnonzero(part != k)
        sub = _iterative_stratify(Y[train_val], [1 - val_size, val_size], rng)
        folds.append({"fold": k + 1, "train": sorted(train_val[sub == 0].tolist()),
                      "val": sorted(train_val[sub == 1].tolist()), "test": sorted(test.tolist()),
                      "val_stratified": True})
    return folds


def make_folds(codes, kfold, val_size, seed):
    """Same procedure as SydneyBatch.kfold_generator: stratified k-fold on the CODE string, then a val split."""
    codes = np.asarray(codes)
    folds = []
    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=seed)
    for k, (train_val, test) in enumerate(skf.split(np.zeros(len(codes)), codes), start=1):
        try:
            train, val = train_test_split(train_val, test_size=val_size, random_state=seed, stratify=codes[train_val])
            stratified = True
        except ValueError:
            train, val = train_test_split(train_val, test_size=val_size, random_state=seed)
            stratified = False
        folds.append({"fold": k, "train": sorted(train.tolist()), "val": sorted(val.tolist()),
                      "test": sorted(test.tolist()), "val_stratified": stratified})
    return folds


def run_epoch(model, idx, bags, y, losses, device, optimizer=None, accumulation_steps=1, rng=None, sites=None,
              site_dropout=0.0):
    """
    One pass. Training follows the paper trainer: loss / accumulation_steps per bag, optimizer step every
    accumulation_steps bags (and at the end). Returns (mean total loss, {task: mean loss}, {task: probs [n, K]}, order).
    sites: site number per case (None = all unknown). In training, site_dropout of the bags get site 0 (unknown).
    """
    train = optimizer is not None
    model.train(train)
    order = rng.permutation(idx) if train else np.asarray(idx)
    total, task_sum = 0.0, {t: 0.0 for t in TASKS}
    probs = {t: [] for t in TASKS}
    with torch.set_grad_enabled(train):
        for step, i in enumerate(order, start=1):
            s = sites[i] if sites is not None else 0
            if train and s and site_dropout > 0 and rng.random() < site_dropout:
                s = 0
            logits, _ = model(bags[i].to(device, non_blocking=True).unsqueeze(0),
                              torch.tensor([s], device=device))
            target = y[i].to(device)
            loss = 0.0
            for j, t in enumerate(TASKS):
                lt = losses[t](logits[t], target[j:j + 1])
                loss = loss + lt
                task_sum[t] += lt.item()
                probs[t].append(losses[t].predict_proba(logits[t].detach()).float().cpu())
            total += loss.item()
            if train:
                (loss / accumulation_steps).backward()
                if step % accumulation_steps == 0 or step == len(order):
                    optimizer.step()
                    optimizer.zero_grad()
    n = max(1, len(order))
    return total / n, {t: v / n for t, v in task_sum.items()}, {t: torch.cat(p).numpy() for t, p in probs.items()}, order


def evaluate(labels, probs):
    """labels [n, 5] -> flat dict <task>_<metric> + mean_qwk, mean_lenient_accuracy, mean_accuracy, code_exact."""
    m, preds = {}, {}
    for j, t in enumerate(TASKS):
        preds[t] = probs[t].argmax(1)
        m.update({f"{t}_{k}": v for k, v in compute_ordinal_metrics(labels[:, j], probs[t], preds[t]).items()})
        m[f"{t}_lenient_accuracy"] = lenient_accuracy(labels[:, j], preds[t])
    for k in ("qwk", "lenient_accuracy", "accuracy"):
        m[f"mean_{k}"] = float(np.mean([m[f"{t}_{k}"] for t in TASKS]))
    m["code_exact"] = float((np.stack([preds[t] for t in TASKS], 1) == labels).all(1).mean())
    return m, preds


def _try_plot(fn, *args):
    """Plotting must never stop training."""
    try:
        fn(*args)
    except Exception as e:  # noqa: BLE001
        logger.info(f"plot skipped ({fn.__name__}): {e}")


def train_fold(fold, cases, bags, y, args, in_dim, device):
    fold_dir = os.path.join(args.out_dir, f"fold{fold['fold']}")
    os.makedirs(fold_dir, exist_ok=True)
    tr, va, te = fold["train"], fold["val"], fold["test"]
    logger.info(f"Fold {fold['fold']}: train={len(tr)} val={len(va)} test={len(te)} "
                f"(val stratified: {fold['val_stratified']})")

    set_seed(args.seed + fold["fold"])
    rng = np.random.default_rng(args.seed + fold["fold"])
    model = SydneyMTL(in_dim, {t: NUM_GRADES for t in TASKS}, args.adaptor_dim, args.site_mode, len(SITES)).to(device)
    sites = [c["site"] for c in cases] if args.site_mode != "none" else None
    y_train = torch.stack([y[i] for i in tr])
    losses = {t: GradeLoss(args.loss, y_train[:, j], NUM_GRADES, args.tau).to(device) for j, t in enumerate(TASKS)}
    site_params = [p for n, p in model.named_parameters() if n.startswith("site_")]
    groups = [{"params": [p for n, p in model.named_parameters() if not n.startswith("site_")]}]
    if site_params:
        groups.append({"params": site_params, "lr": args.site_lr or args.lr})
    optimizer = torch.optim.Adam(groups, lr=args.lr)
    y_np = torch.stack(y).numpy()

    best = {"loss": float("inf"), "epoch": 0, "state": copy.deepcopy(model.state_dict()), "val_metrics": None}
    history, bad_epochs = [], 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_tl, tr_p, tr_order = run_epoch(model, tr, bags, y, losses, device, optimizer,
                                                   args.accumulation_steps, rng, sites, args.site_dropout)
        va_loss, va_tl, va_p, _ = run_epoch(model, va, bags, y, losses, device, sites=sites)
        tr_m, _ = evaluate(y_np[tr_order], tr_p)
        va_m, _ = evaluate(y_np[va], va_p)
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
                        **{f"train_{t}_loss": v for t, v in tr_tl.items()},
                        **{f"val_{t}_loss": v for t, v in va_tl.items()},
                        **{f"train_{k}": v for k, v in tr_m.items()},
                        **{f"val_{k}": v for k, v in va_m.items()}})

        improved = va_loss < best["loss"]
        if improved:
            best = {"loss": va_loss, "epoch": epoch, "state": copy.deepcopy(model.state_dict()),
                    "val_metrics": {"loss": va_loss, **va_m}}
            bad_epochs = 0
        else:
            bad_epochs += 1
        task_qwk = " ".join(f"{t} {va_m[t + '_qwk']:.2f}" for t in TASKS)
        logger.info(f"[fold {fold['fold']}] epoch {epoch:3d} | train loss {tr_loss:.4f} | val loss {va_loss:.4f} "
                    f"qwk {va_m['mean_qwk']:.3f} lenient {va_m['mean_lenient_accuracy']:.3f} [{task_qwk}] | "
                    f"{time.time() - t0:.0f}s{' *' if improved else ''}")
        pd.DataFrame(history).to_csv(os.path.join(fold_dir, "history.csv"), index=False)
        if epoch % 5 == 0:  # 학습 중에도 curves.png로 진행 상황을 볼 수 있게
            _try_plot(plot_curves, fold_dir, best["epoch"])
        if bad_epochs >= args.patience:
            logger.info(f"[fold {fold['fold']}] early stop at epoch {epoch} (best epoch {best['epoch']})")
            break

    model.load_state_dict(best["state"])
    torch.save({"model_state_dict": best["state"], "model_name": "sydney_mtl", "in_dim": in_dim,
                "adaptor_dim": args.adaptor_dim, "tasks": TASKS, "num_grades": NUM_GRADES, "loss": args.loss,
                "tau": args.tau, "log_prior": {t: losses[t].log_prior.cpu() for t in TASKS},
                "unseen": {t: losses[t].unseen.cpu() for t in TASKS}, "best_epoch": best["epoch"],
                "feat_dir": os.path.abspath(args.feat_dir), "label_col": args.label_col, "fold": fold["fold"],
                "site_mode": args.site_mode, "sites": SITES},
               os.path.join(fold_dir, "best_model.pt"))

    def test(test_sites, name):
        loss, tl, p, _ = run_epoch(model, te, bags, y, losses, device, sites=test_sites)
        m, pr = evaluate(y_np[te], p)
        df = pd.DataFrame({"case_id": [cases[i]["case_id"] for i in te],
                           "slide_ids": [";".join(cases[i]["slide_ids"]) for i in te],
                           "site": [SITES[cases[i]["site"] - 1] for i in te],
                           "code_label": [cases[i]["code"] for i in te],
                           "code_pred": ["".join(str(int(pr[t][k])) for t in TASKS) for k in range(len(te))]})
        for j, t in enumerate(TASKS):
            df[f"{t}_label"] = y_np[te, j]
            df[f"{t}_pred"] = pr[t]
            for g in range(NUM_GRADES):
                df[f"{t}_prob{g}"] = p[t][:, g]
        df.to_csv(os.path.join(fold_dir, name), index=False)
        return {"loss": loss, **{f"{t}_loss": v for t, v in tl.items()}, **m}

    te_m = test(sites, "test_predictions.csv")
    result = {"fold": fold["fold"], "best_epoch": best["epoch"], "val": best["val_metrics"], "test": te_m}
    if sites is not None:  # 같은 모델에 부위를 '모름'으로 넣은 결과 (부위 없이 추론할 때 성능)
        result["test_nosite"] = test(None, "test_predictions_nosite.csv")
    with open(os.path.join(fold_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    _try_plot(plot_fold, fold_dir, best["epoch"])
    logger.info(f"[fold {fold['fold']}] TEST mean_qwk={te_m['mean_qwk']:.4f} "
                f"mean_lenient={te_m['mean_lenient_accuracy']:.4f} code_exact={te_m['code_exact']:.4f}")
    for t in TASKS:
        logger.info(f"[fold {fold['fold']}]   {t:<5} qwk={te_m[f'{t}_qwk']:.4f} "
                    f"lenient={te_m[f'{t}_lenient_accuracy']:.4f} acc={te_m[f'{t}_accuracy']:.4f} "
                    f"mae={te_m[f'{t}_mae']:.4f} n={te_m[f'{t}_n']}")
    if "test_nosite" in result:
        ns = result["test_nosite"]
        logger.info(f"[fold {fold['fold']}] TEST site unknown: mean_qwk={ns['mean_qwk']:.4f} "
                    f"mean_lenient={ns['mean_lenient_accuracy']:.4f} atr_qwk={ns['atr_qwk']:.4f}")
    return result


def main():
    args = parse_args()
    os.umask(0)  # NFS: 어느 계정이 돌려도 서로 결과를 정리할 수 있도록
    args.meta_csv_md5 = md5([args.meta_csv])
    args.code_md5 = md5([os.path.join(PROJECT_DIR, f) for f in CODE_FILES])
    args.split_method = {"code": "code_stratified", "multilabel": "multilabel_iterative"}[args.split]
    check_version(args.out_dir, vars(args))
    os.makedirs(args.out_dir, exist_ok=True)
    code_dir = os.path.join(args.out_dir, "code")
    if not os.path.exists(code_dir):
        for f in CODE_FILES:
            os.makedirs(os.path.dirname(os.path.join(code_dir, f)), exist_ok=True)
            shutil.copy2(os.path.join(PROJECT_DIR, f), os.path.join(code_dir, f))
    setup_logging(args.out_dir)
    logger.info(f"version {args.version} -> {args.out_dir}")
    set_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")

    cases = load_cases(args.meta_csv, args.label_col)
    t0 = time.time()
    bag_by_case = load_bags(cases, args.feat_dir, args.load_workers)
    bags = [bag_by_case[c["case_id"]] for c in cases]
    y = [torch.tensor(c["grades"]) for c in cases]
    in_dim = bags[0].shape[1]
    logger.info(f"SydneyMTL | loss {args.loss} (tau {args.tau}) | site {args.site_mode} (dropout {args.site_dropout}, "
                f"lr {args.site_lr or args.lr}) "
                f"| labels {args.label_col} | {len(cases)} cases, "
                f"{sum(len(c['slide_ids']) for c in cases)} slides, {sum(b.shape[0] for b in bags)} patches, "
                f"in_dim {in_dim} | loaded in {time.time() - t0:.0f}s | device {device}")
    for j, t in enumerate(TASKS):
        counts = np.bincount([c["grades"][j] for c in cases], minlength=NUM_GRADES)
        logger.info(f"  {t:<5} grades 0-3: {counts.tolist()}")
    logger.info("  sites: " + ", ".join(f"{s} {sum(c['site'] == k for c in cases)}" for k, s in enumerate(SITES, 1)))

    if args.split == "multilabel":
        folds = make_folds_multilabel(cases, args.kfold, args.val_size, args.seed)
    else:
        folds = make_folds([c["code"] for c in cases], args.kfold, args.val_size, args.seed)
    logger.info(f"split {args.split_method}: test " + " ".join(str(len(fd["test"])) for fd in folds)
                + " | val " + " ".join(str(len(fd["val"])) for fd in folds))
    with open(os.path.join(args.out_dir, "splits.json"), "w") as f:
        json.dump({"label_col": args.label_col, "seed": args.seed, "tasks": TASKS,
                   "folds": [{**fd, **{k: [cases[i]["case_id"] for i in fd[k]] for k in ("train", "val", "test")}}
                             for fd in folds]}, f, indent=1)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump({**vars(args), "model_name": "sydney_mtl", "in_dim": in_dim, "tasks": TASKS,
                   "num_grades": NUM_GRADES, "n_cases": len(cases)}, f, indent=2, ensure_ascii=False)

    results = []
    for fold in folds:
        if args.folds and fold["fold"] not in args.folds:
            continue
        done = os.path.join(args.out_dir, f"fold{fold['fold']}", "metrics.json")
        if os.path.exists(done) and not args.overwrite:
            logger.info(f"[fold {fold['fold']}] already done, skipping (use --overwrite to re-run)")
            with open(done) as f:
                results.append(json.load(f))
            continue
        results.append(train_fold(fold, cases, bags, y, args, in_dim, device))

    rows = [{"fold": r["fold"], "best_epoch": r["best_epoch"], **r["test"]} for r in results]
    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if c not in ("fold", "best_epoch") and not c.endswith("_n")]
    summary = {c: {"mean": float(df[c].mean()), "std": float(df[c].std(ddof=0))} for c in metric_cols}
    pd.concat([df, pd.DataFrame([{"fold": "mean", **{c: summary[c]["mean"] for c in metric_cols}},
                                 {"fold": "std", **{c: summary[c]["std"] for c in metric_cols}}])]
              ).to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    out = {"folds": rows, "summary": summary}
    if all("test_nosite" in r for r in results):  # 부위 입력 모델: 부위 모름으로 넣은 test도 같이
        dn = pd.DataFrame([r["test_nosite"] for r in results])
        out["summary_nosite"] = {c: {"mean": float(dn[c].mean()), "std": float(dn[c].std(ddof=0))}
                                 for c in metric_cols if c in dn}
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    _try_plot(plot_run, args.out_dir)

    def ms(c):
        return f"{summary[c]['mean']:.4f} ± {summary[c]['std']:.4f}"

    logger.info("=" * 80)
    logger.info(f"SydneyMTL ({args.loss}) | {os.path.basename(args.feat_dir)} | {args.label_col} | "
                f"{len(results)} folds (test)")
    logger.info(f"  {'task':<6} {'qwk':<17} {'lenient_acc':<17} {'accuracy':<17} mae")
    for t in TASKS:
        logger.info(f"  {t:<6} {ms(f'{t}_qwk'):<17} {ms(f'{t}_lenient_accuracy'):<17} "
                    f"{ms(f'{t}_accuracy'):<17} {ms(f'{t}_mae')}")
    logger.info(f"  mean_qwk {ms('mean_qwk')} | mean_lenient_accuracy {ms('mean_lenient_accuracy')} | "
                f"code_exact {ms('code_exact')}")
    if "summary_nosite" in out:
        ns = out["summary_nosite"]
        logger.info(f"  site unknown: mean_qwk {ns['mean_qwk']['mean']:.4f} ± {ns['mean_qwk']['std']:.4f} | "
                    f"atr_qwk {ns['atr_qwk']['mean']:.4f} | mean_lenient_accuracy "
                    f"{ns['mean_lenient_accuracy']['mean']:.4f}")
    if not args.no_mlflow and len(results) == args.kfold:
        upload_mlflow(args)


if __name__ == "__main__":
    main()
