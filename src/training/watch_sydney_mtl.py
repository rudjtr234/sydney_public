# src/training/watch_sydney_mtl.py
# 터미널에서 Sydney SydneyMTL 학습 진행 상황을 보는 모니터 (몇 초마다 화면 갱신, Ctrl+C로 종료)
#   1) 실행 중인 train_sydney_mtl 프로세스: PID, 계정, 경과 시간, 버전, 인코더, GPU 사용량
#   2) 결과 폴더별 상태
#      학습 중  현재 fold/epoch, 마지막 val loss·평균 QWK·lenient, 항목별 val QWK, best epoch, early stop까지 남은 epoch,
#               끝난 fold의 test 평균 QWK
#      완료     5-fold test 평균 QWK·lenient (± 표준편차), MLflow 업로드 여부
#   3) 학습 중인 결과의 train.log 마지막 몇 줄
# 결과 파일(train.log, foldN/history.csv, foldN/metrics.json)만 읽으므로 학습에 영향을 주지 않는다.
#
# 사용 (scripts/watch_sydney.sh가 sydney env python으로 실행):
#   bash scripts/watch_sydney.sh                 # 5초마다 갱신
#   bash scripts/watch_sydney.sh --interval 10   # 10초마다
#   bash scripts/watch_sydney.sh --once          # 한 번만 출력
#   bash scripts/watch_sydney.sh --all           # 완료된 결과 전부 표시 (기본은 최근 5개)

import argparse
import glob
import json
import os
import re
import subprocess
import time

import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TASKS = ["hp", "neu", "mono", "atr", "im"]
BOLD, DIM, GREEN, YELLOW, CYAN, RESET = "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[36m", "\033[0m"


def _fmt_time(sec):
    sec = int(sec)
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m" if sec >= 3600 else f"{sec // 60}m{sec % 60:02d}s"


def _rel(path):
    return os.path.relpath(path, PROJECT_DIR) if path.startswith(PROJECT_DIR + os.sep) else path


def _arg(args, name, default=None):
    m = re.search(rf"--{name}[ =](\S+)", args)
    return m.group(1) if m else default


def running_processes():
    """Main train_sydney_mtl processes (not their children) with their resolved out_dir."""
    out = subprocess.run(["ps", "-eo", "pid=,ppid=,user=,etimes=,args="], capture_output=True, text=True).stdout
    procs = []
    rows = [r for r in (line.split(None, 4) for line in out.splitlines() if "src.training.train_sydney_mtl" in line)
            if len(r) == 5 and os.path.basename(r[4].split()[0]).startswith("python")]  # 감싼 bash/nohup 줄 제외
    pids = {r[0] for r in rows}
    for pid, ppid, user, etimes, args in rows:
        if ppid in pids:
            continue
        feat_dir = _arg(args, "feat_dir", "")
        enc = os.path.basename(os.path.normpath(feat_dir)).removeprefix("features_")
        mag = os.path.basename(os.path.dirname(os.path.normpath(feat_dir))).split("_")[0]  # 20x_224px_... -> 20x
        version = _arg(args, "version")
        root = _arg(args, "output_root", os.path.join(PROJECT_DIR, "outputs"))
        out_dir = _arg(args, "out_dir") or (os.path.join(root, f"sydney_mtl_{enc}_{mag}_v{version}") if version else None)
        if out_dir and not os.path.isabs(out_dir):
            out_dir = os.path.join(PROJECT_DIR, out_dir)
        procs.append({"pid": pid, "user": user, "elapsed": int(etimes), "encoder": f"{enc} {mag}", "version": version,
                      "gpu": _arg(args, "gpu", "0"), "out_dir": os.path.normpath(out_dir) if out_dir else None})
    return procs


def gpu_usage():
    """pid -> MiB used, gpu index -> util %."""
    q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout
    mem = {line.split(",")[0].strip(): line.split(",")[1].strip() for line in q.splitlines() if "," in line}
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    util = {r[0]: r[1:] for r in (list(map(str.strip, line.split(","))) for line in q.splitlines()) if len(r) == 4}
    return mem, util


def run_dirs(output_root):
    dirs = []
    for cfg in glob.glob(os.path.join(output_root, "**", "config.json"), recursive=True):
        try:
            with open(cfg) as f:
                if json.load(f).get("model_name") == "sydney_mtl":
                    dirs.append(os.path.dirname(cfg))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(dirs, key=os.path.getmtime)


def fold_status(out_dir):
    with open(os.path.join(out_dir, "config.json")) as f:
        cfg = json.load(f)
    folds = []
    for d in sorted(glob.glob(os.path.join(out_dir, "fold*")), key=lambda d: int(d.rsplit("fold", 1)[1])):
        fs = {"fold": int(d.rsplit("fold", 1)[1]), "done": None, "hist": None}
        if os.path.exists(os.path.join(d, "metrics.json")):
            with open(os.path.join(d, "metrics.json")) as f:
                fs["done"] = json.load(f)
        if os.path.exists(os.path.join(d, "history.csv")):
            try:
                fs["hist"] = pd.read_csv(os.path.join(d, "history.csv"))
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                pass
        folds.append(fs)
    return cfg, folds


def show_running(out_dir, proc, tail):
    cfg, folds = fold_status(out_dir) if os.path.exists(os.path.join(out_dir, "config.json")) else ({}, [])
    kfold, patience = cfg.get("kfold", 5), cfg.get("patience", 10)
    print(f"{BOLD}{CYAN}▶ {_rel(out_dir)}{RESET}  "
          f"(loss {cfg.get('loss', '?')}, labels {cfg.get('label_col', '?')}, {cfg.get('n_cases', '?')} cases)")
    done = [f for f in folds if f["done"]]
    current = next((f for f in folds if not f["done"] and f["hist"] is not None and len(f["hist"])), None)
    if not folds and not current:
        mag = re.search(r"_(\d+)x_v", os.path.basename(out_dir))
        hint = {"10": "1~2분", "20": "2~3분", "40": "5~10분, 둘 이상 동시면 더"}.get(mag.group(1) if mag else "", "1~3분")
        print(f"  {YELLOW}특징 파일 읽는 중 (보통 {hint}, 이 동안 GPU 0%가 정상){RESET}")
    if current:
        h = current["hist"]
        last = h.iloc[-1]
        best_i = h["val_loss"].idxmin()
        since = int(last["epoch"] - h.loc[best_i, "epoch"])
        print(f"  fold {current['fold']}/{kfold} | epoch {int(last['epoch'])} | "
              f"val loss {last['val_loss']:.4f} | val QWK {last['val_mean_qwk']:.3f} | "
              f"lenient {last['val_mean_lenient_accuracy']:.3f}")
        print(f"  best epoch {int(h.loc[best_i, 'epoch'])} (val loss {h.loc[best_i, 'val_loss']:.4f}) | "
              f"early stop까지 {max(patience - since, 0)} epoch 남음")
        print("  val QWK by task: " + "  ".join(f"{t} {last[f'val_{t}_qwk']:.2f}" for t in TASKS))
    for f in done:
        t = f["done"]["test"]
        print(f"  {GREEN}fold {f['fold']} done{RESET} (best epoch {f['done']['best_epoch']}) test QWK "
              f"{t['mean_qwk']:.3f} | lenient {t['mean_lenient_accuracy']:.3f} | "
              + " ".join(f"{k} {t[f'{k}_qwk']:.2f}" for k in TASKS))
    log = os.path.join(out_dir, "train.log")
    if done and os.path.exists(log):
        # 남은 시간: 특징 읽는 시간은 빼고, 이번 실행에서 첫 fold를 시작한 시각부터 잰다
        # (다시 실행해서 건너뛴 fold는 빼고, 이번 실행에서 학습한 fold만으로 fold당 시간을 잰다)
        with open(log) as f:
            session = re.split(r"^\S+ \S+ \| version ", f.read(), flags=re.M)[-1]
        starts = re.findall(r"^(\S+ \S+) \| Fold \d+:", session, re.M)
        n_tested = len(re.findall(r"\] TEST mean_qwk", session))
        if starts and (n_tested or current):
            trained = time.time() - time.mktime(time.strptime(starts[0], "%Y-%m-%d %H:%M:%S"))
            per_fold = trained / (n_tested + (0.5 if current else 0))
            left = per_fold * (kfold - len(done) - (0.5 if current else 0))
            print(f"  {DIM}fold {len(done)}/{kfold} 완료 | 남은 시간 약 {_fmt_time(max(left, 0))}{RESET}")
    if tail and os.path.exists(log):
        with open(log) as f:
            lines = f.readlines()[-tail:]
        print(f"  {DIM}── train.log (last {len(lines)}) ──{RESET}")
        for line in lines:
            print(f"  {DIM}{line.rstrip()[:160]}{RESET}")


def show_finished(out_dir):
    with open(os.path.join(out_dir, "summary.json")) as f:
        s = json.load(f)["summary"]
    mlflow = os.path.exists(os.path.join(out_dir, "mlflow_run.json"))
    print(f"  {GREEN}✔{RESET} {_rel(out_dir):<45} "
          f"QWK {s['mean_qwk']['mean']:.3f}±{s['mean_qwk']['std']:.3f} | "
          f"lenient {s['mean_lenient_accuracy']['mean']:.3f}±{s['mean_lenient_accuracy']['std']:.3f} | "
          + " ".join(f"{t} {s[f'{t}_qwk']['mean']:.2f}" for t in TASKS)
          + f" | MLflow {'uploaded' if mlflow else '-'}")


def render(args):
    procs = running_processes()
    mem, util = gpu_usage()
    print(f"{BOLD}Sydney SydneyMTL monitor{RESET}  {time.strftime('%Y-%m-%d %H:%M:%S')}  "
          f"{DIM}(Ctrl+C로 종료){RESET}\n")
    print(f"{BOLD}실행 중인 학습{RESET}")
    if not procs:
        print("  (없음)")
    for p in procs:
        g = util.get(p["gpu"], ["?", "?", "?"])
        print(f"  PID {p['pid']} ({p['user']}) | {p['encoder']} v{p['version']} | 경과 {_fmt_time(p['elapsed'])} | "
              f"GPU {p['gpu']}: 이 학습 {mem.get(p['pid'], '0')} MiB, GPU 전체 {g[1]}/{g[2]} MiB, 사용률 {g[0]}%")
    print()
    running = {p["out_dir"]: p for p in procs if p["out_dir"]}
    for out_dir, p in running.items():
        show_running(out_dir, p, args.tail)
        print()
    finished = [d for d in run_dirs(args.output_root)
                if d not in running and os.path.exists(os.path.join(d, "summary.json"))]
    print(f"{BOLD}완료된 결과{RESET} (5-fold test 평균)")
    if not finished:
        print("  (없음)")
    for d in (finished if args.all else finished[-5:]):
        show_finished(d)


def main():
    p = argparse.ArgumentParser(description="Terminal monitor for Sydney SydneyMTL training")
    p.add_argument("--interval", type=float, default=5)
    p.add_argument("--once", action="store_true")
    p.add_argument("--tail", type=int, default=6, help="train.log lines to show for running runs (0 = none)")
    p.add_argument("--all", action="store_true", help="Show every finished run (default: latest 5)")
    p.add_argument("--output_root", default=os.path.join(PROJECT_DIR, "outputs"))
    args = p.parse_args()
    if args.once:
        render(args)
        return
    try:
        while True:
            print("\033[2J\033[H", end="")
            render(args)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
