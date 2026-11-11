# src/training/mlflow_sydney.py
# train_sydney_mtl.py 결과 폴더 하나를 MLflow 서버에 올린다. 학습이 끝나면 자동으로 불리고, 직접 실행해서 올릴 수도 있다.
#   experiment "sydney", run 이름 = 결과 폴더 이름 (예: sydney_mtl_hibou_l_20x_v0.1.0)
#   parent run  설정(params), 설명·버전 태그, fold 평균/표준편차 test 지표, epoch별 fold 평균 곡선,
#               결과 폴더 전체(그림, 표, 예측, 모델, 코드)를 artifact "results/"로
#   child run   fold마다 1개 (run 이름 fold1..). epoch별 train/val 지표 전체 + 그 fold의 test 지표
# 같은 run 이름이 이미 있으면 올리지 않는다 (--replace면 기존 run을 지우고 다시 올림).
# run 설명(Description) 맨 위 = 이 버전이 기준 버전과 무엇이 다르고 왜 바꿨는지 (VERSION_NOTES). 새 버전을 만들면 여기에 추가.
#   이미 올린 run의 설명만 고치기: --notes_only (다시 올리지 않고 태그만 바꿈)
# 학습 환경에 mlflow가 없으면 mlflow가 설치된 다른 환경의 python으로 실행한다 (MLFLOW_PYTHON).
#
# 사용:
#   python -m src.training.mlflow_sydney \
#       --out_dir outputs/sydney_mtl_hibou_l_20x_v0.1.0

import argparse
import glob
import json
import os
import re
import time

# 자체 서명 인증서 서버라면 실행 전에 MLFLOW_TRACKING_INSECURE_TLS=true 를 설정한다

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.entities import Metric  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT = os.environ.get("SYDNEY_MLFLOW_EXPERIMENT", "sydney")  # 시험할 때만 바꿈
ENCODER_NAMES = {"hibou_l": "Hibou-L", "hoptimus0": "H-Optimus-0", "virchow": "Virchow", "midnight12k": "Midnight-12k"}
CURVE_COLS = ["train_loss", "val_loss", "train_mean_qwk", "val_mean_qwk", "val_mean_lenient_accuracy"]
TASKS = ["hp", "neu", "mono", "atr", "im"]
# 버전마다: base = 비교 기준 버전, change = 기준과 다른 점, why = 바꾼 이유, compare = 결과를 어떻게 비교하나
VERSION_NOTES = {
    "0.1.0": {"base": "-", "change": "없음 (논문 재현 기준선)",
              "why": "SydneyMTL 공개 코드와 같은 모델·손실·학습 설정, 논문 분할(5자리 CODE 층화 5-fold)로 재현하고 FM 비교",
              "compare": "v0.1.x·v0.2.x는 모두 같은 splits.json → fold별 짝 비교 가능"},
    "0.2.0": {"base": "0.1.0", "change": "생검 부위 입력 A: 항목별 점수에 부위별 보정값 (--site_mode head, site_dropout 0.2)",
              "why": "생검 부위에 따라 정상 조직 구성이 달라 같은 소견도 판독 기준이 달라짐",
              "compare": "v0.1.0과 같은 splits.json → fold별 짝 비교"},
    "0.2.1": {"base": "0.1.0", "change": "생검 부위 입력 B: 패치 특징을 부위별로 조정 (--site_mode film, site_dropout 0.2)",
              "why": "생검 부위에 따라 정상 조직 구성이 달라 같은 소견도 판독 기준이 달라짐",
              "compare": "v0.1.0과 같은 splits.json → fold별 짝 비교"},
    "0.2.2": {"base": "0.2.0", "change": "v0.2.0 + 부위 파라미터만 lr 1e-2 (--site_lr 1e-2)",
              "why": "0에서 시작하는 부위 보정값은 기본 lr 1e-4로는 거의 움직이지 않음",
              "compare": "v0.1.0과 같은 splits.json → fold별 짝 비교"},
    "0.2.3": {"base": "0.2.1", "change": "v0.2.1 + 부위 파라미터만 lr 1e-2 (--site_lr 1e-2)",
              "why": "0에서 시작하는 부위 보정값은 기본 lr 1e-4로는 거의 움직이지 않음",
              "compare": "v0.1.0과 같은 splits.json → fold별 짝 비교"},
    "0.3.0": {"base": "0.1.0",
              "change": "fold 분할만 바꿈: --split multilabel (항목별 등급 5×4칸 + 생검 부위 4칸을 fold마다 고르게, "
                        "iterative stratification). 모델·손실·학습 설정·입력(WSI 임베딩만)은 v0.1.0과 같음",
              "why": "5자리 CODE 층화는 CODE 종류가 많아 드문 등급이 fold마다 고르지 않음 → fold마다 시험 구성이 달라 "
                     "점수가 흔들림",
              "compare": "분할이 달라 v0.1.0과 fold별 짝 비교는 안 됨 → fold 평균·표준편차와 전체 사례를 합친 지표로 비교. "
                         "v0.3.x끼리는 같은 분할"},
}


def _batch_log(client, run_id, metrics):
    for i in range(0, len(metrics), 900):
        client.log_batch(run_id, metrics=metrics[i:i + 900])


def _history_metrics(history, cols, ts):
    out = []
    for _, row in history.iterrows():
        for c in cols:
            if c in row and pd.notna(row[c]):
                out.append(Metric(c, float(row[c]), ts, int(row["epoch"])))
    return out


def version_note(cfg):
    """Markdown block: what this version changes against its base version, why, and how to compare."""
    v, n = cfg["version"], VERSION_NOTES.get(cfg["version"])
    if n is None:
        return f"**v{v}**: 버전 설명 없음 (src/training/mlflow_sydney.py VERSION_NOTES에 추가)"
    head = f"**v{v}** (기준선)" if n["base"] == "-" else f"**v{v} — v{n['base']}과 다른 점**"
    lines = [f"{head}: {n['change']}", f"**왜**: {n['why']}", f"**비교 방법**: {n['compare']}"]
    if str(cfg["mag"]) != "20":
        lines.append(f"**배율 {cfg['mag']}x**: 같은 버전의 20x와 설정·분할이 같고 배율만 다름")
    return "  \n".join(lines)


def describe(cfg, n_folds):
    enc = ENCODER_NAMES.get(cfg["encoder"], cfg["encoder"])
    loss = {"la": f"logit-adjusted CE (tau {cfg['tau']})", "ce": "CE", "wce": "class-weighted CE"}[cfg["loss"]]
    return version_note(cfg) + "\n\n" + (f"WSI → TRIDENT HEST tissue seg → patches {cfg['mag']}x/224px → {enc} ({cfg['in_dim']}-dim) → SydneyMTL "
            f"(shared adaptor {cfg['adaptor_dim']} + per-task attention/head, Kim et al. Histopathology 2026) → "
            f"Updated Sydney 5 items (HP/NEU/MONO/ATR/IM) grade 0-3. Loss {loss}, Adam lr {cfg['lr']}, "
            f"batch 1 x accumulation {cfg['accumulation_steps']}, early stop patience {cfg['patience']}. "
            f"{n_folds}-fold CV ({cfg.get('split_method', 'code_stratified')}, train/val/test 70/10/20), "
            f"labels {cfg['label_col']}{' (' + cfg['label_source'] + ')' if cfg.get('label_source') else ''}, "
            f"{cfg['n_cases']} cases.")


def note_tags(cfg, n_folds):
    n = VERSION_NOTES.get(cfg["version"], {})
    d = describe(cfg, n_folds)
    return {"mlflow.note.content": d, "Description": d, "changes": n.get("change", "")[:250],
            "baseline": n.get("base", "")}


def load_cfg(out_dir):
    with open(os.path.join(out_dir, "config.json")) as f:
        cfg = json.load(f)
    cfg.setdefault("encoder", re.sub(r"^features_", "", os.path.basename(os.path.normpath(cfg["feat_dir"]))))
    cfg.setdefault("mag", re.match(r"(\d+(?:\.\d+)?)x_", os.path.basename(os.path.dirname(cfg["feat_dir"]))).group(1))
    m = re.search(r"_v(\d+\.\d+\.\d+)$", os.path.basename(out_dir))
    cfg.setdefault("version", m.group(1) if m else "")
    return cfg


def update_notes(out_dir, experiment=EXPERIMENT):
    """Rewrite only the description/changes tags of an uploaded run (no re-upload)."""
    out_dir = os.path.abspath(out_dir)
    name, cfg = os.path.basename(out_dir), load_cfg(out_dir)
    n_folds = len(glob.glob(os.path.join(out_dir, "fold*", "metrics.json")))
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment)
    runs = client.search_runs([exp.experiment_id], f"attributes.run_name = '{name}' and tags.level = 'parent'")
    if not runs:
        print(f"[mlflow] '{name}' not found in '{experiment}'")
        return None
    for k, v in note_tags(cfg, n_folds).items():
        client.set_tag(runs[0].info.run_id, k, v)
    print(f"[mlflow] notes updated '{name}' ({runs[0].info.run_id})")
    return runs[0].info.run_id


def upload(out_dir, experiment=EXPERIMENT, replace=False, code_dir=None):
    out_dir = os.path.abspath(out_dir)
    name = os.path.basename(out_dir)
    with open(os.path.join(out_dir, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(out_dir, "summary.json")) as f:
        summary = json.load(f)
    cfg.setdefault("encoder", re.sub(r"^features_", "", os.path.basename(os.path.normpath(cfg["feat_dir"]))))
    cfg.setdefault("mag", re.match(r"(\d+(?:\.\d+)?)x_", os.path.basename(os.path.dirname(cfg["feat_dir"]))).group(1))
    m = re.search(r"_v(\d+\.\d+\.\d+)$", name)
    cfg.setdefault("version", m.group(1) if m else "")

    mlflow.set_tracking_uri(TRACKING_URI)
    exp = mlflow.set_experiment(experiment)
    client = MlflowClient()
    old = client.search_runs([exp.experiment_id], f"attributes.run_name = '{name}' and tags.level = 'parent'")
    if old and not replace:
        print(f"[mlflow] '{name}' already exists in '{experiment}' (run_id {old[0].info.run_id}); use --replace")
        return old[0].info.run_id
    for r in old:
        for child in client.search_runs([exp.experiment_id], f"tags.mlflow.parentRunId = '{r.info.run_id}'"):
            client.delete_run(child.info.run_id)
        client.delete_run(r.info.run_id)

    fold_dirs = sorted(glob.glob(os.path.join(out_dir, "fold*")), key=lambda d: int(d.rsplit("fold", 1)[1]))
    histories = {d: pd.read_csv(os.path.join(d, "history.csv")) for d in fold_dirs
                 if os.path.exists(os.path.join(d, "history.csv"))}
    ts = int(time.time() * 1000)

    with mlflow.start_run(run_name=name) as parent:
        pid = parent.info.run_id
        skip = {"gpu", "folds", "overwrite", "load_workers", "no_mlflow", "mlflow_python", "code_md5"}
        mlflow.log_params({k: str(v)[:500] for k, v in cfg.items() if k not in skip and not isinstance(v, dict)})
        mlflow.set_tags({"level": "parent", "version": cfg["version"], "encoder": cfg["encoder"],
                         "magnification": f"{cfg['mag']}x",
                         "embedding_model": ENCODER_NAMES.get(cfg["encoder"], cfg["encoder"]),
                         "model": "SydneyMTL", "label_col": cfg["label_col"], "out_dir": out_dir,
                         **note_tags(cfg, len(fold_dirs))})
        mlflow.log_metrics({f"test_{k}_{s}": v[s] for k, v in summary["summary"].items() for s in ("mean", "std")})
        if "summary_nosite" in summary:  # 부위 입력 모델을 부위 모름으로 test한 결과
            mlflow.log_metrics({f"test_nosite_{k}_{s}": v[s] for k, v in summary["summary_nosite"].items()
                                for s in ("mean", "std")})

        # epoch별 fold 평균 곡선 (fold마다 길이가 달라서 그 epoch까지 간 fold만 평균)
        if histories:
            allh = pd.concat(histories.values())
            cols = CURVE_COLS + [f"val_{t}_qwk" for t in TASKS]
            mean_h = allh.groupby("epoch")[[c for c in cols if c in allh]].mean().reset_index()
            _batch_log(client, pid, _history_metrics(mean_h, [c for c in cols if c in allh], ts))

        for d in fold_dirs:
            with open(os.path.join(d, "metrics.json")) as f:
                fm = json.load(f)
            with mlflow.start_run(run_name=os.path.basename(d), nested=True) as child:
                mlflow.set_tags({"level": "fold", "version": cfg["version"], "encoder": cfg["encoder"],
                                 "magnification": f"{cfg['mag']}x"})
                mlflow.log_params({"fold": fm["fold"], "best_epoch": fm["best_epoch"]})
                mlflow.log_metrics({f"test_{k}": float(v) for k, v in fm["test"].items()
                                    if isinstance(v, (int, float)) and not k.endswith("_n")})
                mlflow.log_metrics({f"test_nosite_{k}": float(v) for k, v in fm.get("test_nosite", {}).items()
                                    if isinstance(v, (int, float)) and not k.endswith("_n")})
                if d in histories:
                    h = histories[d]
                    _batch_log(client, child.info.run_id, _history_metrics(h, [c for c in h if c != "epoch"], ts))
                for png in glob.glob(os.path.join(d, "*.png")):
                    mlflow.log_artifact(png)

        mlflow.log_artifacts(out_dir, artifact_path="results")
        if code_dir:
            mlflow.log_artifacts(code_dir, artifact_path="results/code")
        for png in ("summary_metrics.png", "confusion_all.png"):
            if os.path.exists(os.path.join(out_dir, png)):
                mlflow.log_artifact(os.path.join(out_dir, png))

    info = {"run_id": pid, "experiment": experiment, "run_name": name, "tracking_uri": TRACKING_URI,
            "url": f"{TRACKING_URI}/#/experiments/{exp.experiment_id}/runs/{pid}"}
    try:
        with open(os.path.join(out_dir, "mlflow_run.json"), "w") as f:
            json.dump(info, f, indent=2)
    except OSError:
        pass  # 결과 폴더에 쓸 권한이 없는 계정으로 올린 경우
    print(f"[mlflow] uploaded '{name}' -> {info['url']}")
    return pid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--experiment", default=EXPERIMENT)
    p.add_argument("--replace", action="store_true", help="Delete an existing run with the same name first")
    p.add_argument("--code_dir", default=None, help="Extra code snapshot to attach (results without code/)")
    p.add_argument("--notes_only", action="store_true",
                   help="Only rewrite the description/changes tags of an already uploaded run")
    a = p.parse_args()
    if a.notes_only:
        update_notes(a.out_dir, a.experiment)
    else:
        upload(a.out_dir, a.experiment, a.replace, a.code_dir)


if __name__ == "__main__":
    main()
