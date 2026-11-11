# src/data/watch_embedding.py
# 터미널에서 TRIDENT 임베딩 추출(seg/coords/feat) 진행 상황을 보는 모니터 (몇 초마다 화면 갱신, Ctrl+C로 종료)
#   1) 실행 중인 TRIDENT (run_batch_of_slides.py): 작업, 모델, 배율, 계정, 경과 시간, 작업자·GPU별 메모리와 사용률
#      진행: 슬라이드 n/N, 패치 기준 진행률, 이번 실행 속도, 남은 시간, 지금 처리 중인 슬라이드(.lock), 오류 난 슬라이드,
#      실행 로그 마지막 몇 줄
#   2) 모델 × 배율 표: 배율 폴더(20x/40x/10x_224px_0px_overlap)마다 좌표·특징 슬라이드 수와 특징 용량
# 결과 폴더의 파일 목록과 로그만 읽으므로 추출에 영향을 주지 않는다.
#
# 사용 (scripts/watch_embedding.sh가 sydney env python으로 실행):
#   bash scripts/watch_embedding.sh                # 10초마다 갱신
#   bash scripts/watch_embedding.sh --interval 30  # 30초마다
#   bash scripts/watch_embedding.sh --once         # 한 번만 출력

import argparse
import json
import os
import socket
import re
import subprocess
import time
from collections import defaultdict

EMB_ROOT = "/path/to/sydney/embedding"
WSI_DIR = "/path/to/sydney/wsi_tiff"
MAG_DIR = re.compile(r"^(\d+(?:\.\d+)?)x_(\d+)px_(\d+)px_overlap$")
# 좌표 h5 크기로 패치 수를 어림한다. TRIDENT는 coords를 (1, 2) 청크로 저장해서 크기 ≈ 12.5 KB + 60.75 B × 패치 수
# (오차 약 2%). h5를 NFS에서 전부 열면 느려서 파일 크기만 본다.
COORDS_HEADER, COORDS_PER_PATCH = 12534, 60.75
BOLD, DIM, GREEN, YELLOW, RED, CYAN, RESET = "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[31m", "\033[36m", "\033[0m"


def _fmt_time(sec):
    sec = int(sec)
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m" if sec >= 3600 else f"{sec // 60}m{sec % 60:02d}s"


def _arg(args, name, default=None):
    m = re.search(rf"--{name}[ =](\S+)", args)
    return m.group(1) if m else default


def _patches(size):
    return max(size - COORDS_HEADER, 0) / COORDS_PER_PATCH


def _count(n):
    return f"{n / 1e6:.2f}M" if n >= 1e5 else f"{n:,.0f}"


def scan(d, suffix):
    """{file name: (size, mtime)} of finished files (no .lock) ending with suffix, and names still locked."""
    try:
        entries = list(os.scandir(d))
    except (FileNotFoundError, NotADirectoryError, PermissionError):  # 복사 중이라 아직 못 읽는 폴더 포함
        return {}, []
    locked = {e.name[:-len(".lock")] for e in entries if e.name.endswith(".lock")}
    done = {}
    for e in entries:
        if e.name.endswith(suffix) and e.name not in locked:
            try:
                st = e.stat()
            except FileNotFoundError:  # 그 사이에 지워짐
                continue
            done[e.name] = (st.st_size, st.st_mtime)
    return done, sorted(locked)


def n_slides(wsi_dir):
    try:
        return sum(1 for n in os.listdir(wsi_dir) if n.lower().endswith((".tiff", ".tif", ".svs")))
    except FileNotFoundError:
        return 0


def trident_jobs():
    """Main run_batch_of_slides.py processes with their worker process trees."""
    out = subprocess.run(["ps", "-eo", "pid=,ppid=,user=,etimes=,args="], capture_output=True, text=True).stdout
    procs = [r for r in (line.split(None, 4) for line in out.splitlines()) if len(r) == 5]
    children, cmd = defaultdict(list), {r[0]: r[4] for r in procs}
    for pid, ppid, *_ in procs:
        children[ppid].append(pid)
    rows = [r for r in procs if "run_batch_of_slides.py" in r[4]
            and os.path.basename(r[4].split()[0]).startswith("python")]  # 감싼 bash/taskset 줄 제외
    pids = {r[0] for r in rows}
    jobs = []
    for pid, ppid, user, etimes, args in rows:
        if ppid in pids:
            continue
        tree, stack = [], [pid]
        while stack:  # 작업자는 spawn으로 떠서 명령줄이 달라 부모-자식 관계로 찾는다
            p = stack.pop()
            tree.append(p)
            stack += children.get(p, [])
        mag = f"{float(_arg(args, 'mag', '20')):g}"
        gpus = re.search(r"--gpus((?:\s+-?\d+)+)", args)
        try:
            pgid = os.getpgid(int(pid))
        except ProcessLookupError:
            continue
        jobs.append({"pid": pid, "pgid": pgid, "user": user, "elapsed": int(etimes), "tree": tree,
                     "workers": sum(1 for p in tree if p != pid and "spawn_main" in cmd[p]
                                    and "resource_tracker" not in cmd[p]),
                     "task": _arg(args, "task", "seg"), "encoder": _arg(args, "patch_encoder"),
                     "job_dir": os.path.normpath(_arg(args, "job_dir", "")), "wsi_dir": _arg(args, "wsi_dir", WSI_DIR),
                     "coords_dir": _arg(args, "coords_dir") or
                     f"{mag}x_{_arg(args, 'patch_size', '512')}px_{_arg(args, 'overlap', '0')}px_overlap",
                     "gpus": gpus.group(1).split() if gpus else [_arg(args, "gpu", "0")]})
    return jobs


def gpu_usage():
    """pid -> MiB used, gpu index -> [util %, used MiB, total MiB]."""
    q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout
    mem = {r[0]: int(r[1]) for r in (list(map(str.strip, line.split(","))) for line in q.splitlines())
           if len(r) == 2 and r[1].isdigit()}
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    util = {r[0]: r[1:] for r in (list(map(str.strip, line.split(","))) for line in q.splitlines()) if len(r) == 4}
    return mem, util


def job_progress(job):
    """Done/total slides and patch weights for one TRIDENT job (weights = 1 per slide unless coords are known)."""
    jd, cd, task = job["job_dir"], os.path.join(job["job_dir"], job["coords_dir"]), job["task"]
    if task == "feat":
        coords, _ = scan(os.path.join(cd, "patches"), "_patches.h5")
        done, locked = scan(os.path.join(cd, f"features_{job['encoder']}"), ".h5")
        weight = {n[:-len("_patches.h5")]: _patches(s) for n, (s, _) in coords.items()}
        total = len(coords)
        log = os.path.join(cd, f"_logs_feats_{job['encoder']}.txt")
    elif task == "coords":
        done, locked = scan(os.path.join(cd, "patches"), "_patches.h5")
        weight, total, log = None, n_slides(job["wsi_dir"]), os.path.join(cd, "_logs_coords.txt")
    else:
        done, locked = scan(os.path.join(jd, "contours"), ".jpg")
        weight, total, log = None, n_slides(job["wsi_dir"]), os.path.join(jd, "_logs_segmentation.txt")
    stem = {n: re.sub(r"(_patches)?\.(h5|jpg)$", "", n) for n in done}
    w = (lambda n: weight.get(stem[n], 0)) if weight else (lambda n: 1)
    start = time.time() - job["elapsed"]
    new = [n for n, (_, mtime) in done.items() if mtime >= start]
    errors, finished = [], set(stem.values())
    if os.path.exists(log):  # TRIDENT 로그는 슬라이드마다 마지막 상태 한 줄 ("CASE-0001.tiff: ERROR: ...")
        with open(log, errors="replace") as f:
            errors = [line.strip() for line in f if "ERROR" in line
                      and os.path.splitext(line.split(":", 1)[0])[0] not in finished]
    return {"done": len(done), "total": total, "locked": [re.sub(r"(_patches)?\.(h5|jpg)$", "", n) for n in locked],
            "w_done": sum(w(n) for n in done), "w_new": sum(w(n) for n in new), "n_new": len(new),
            "w_total": sum(weight.values()) if weight else total, "patch_based": bool(weight), "errors": errors}


def launch_log(job):
    """Newest *.log for the job: EMB_ROOT/logs/ (extract_features_fm.sh), or the older spots job_dir / EMB_ROOT.
    logs/에는 다른 서버·다른 모델 로그도 같이 있어서 파일 이름에 그 모델이 들어 있는 것만 고른다."""
    start = time.time() - job["elapsed"] - 60
    root = os.path.dirname(job["job_dir"])
    logs = [os.path.join(d, n) for d in {job["job_dir"], root, os.path.join(root, "logs")}
            for n in (os.listdir(d) if os.path.isdir(d) else []) if n.endswith(".log")]
    if job["encoder"]:  # extract_hoptimus0+virchow_40x_... 에서 hoptimus0 (hibou_l처럼 _가 든 이름도)
        logs = [p for p in logs if re.search(rf"(^|[_+]){re.escape(job['encoder'])}[_+]", os.path.basename(p))]
    logs = [p for p in logs if os.path.getmtime(p) >= start]
    return max(logs, key=os.path.getmtime) if logs else None


def log_tail(path, n):
    with open(path, "rb") as f:
        f.seek(max(os.path.getsize(path) - 32768, 0))
        text = f.read().decode(errors="replace")
    lines = []
    for seg in re.split(r"[\r\n]+", text):  # tqdm은 \r로 같은 줄을 덮어쓴다
        seg = seg.strip()
        if seg and (not lines or lines[-1] != seg):
            lines.append(seg)
    return lines[-n:]


def show_job(job, mem, util, tail):
    name = {"feat": f"특징 {job['encoder']}", "coords": "좌표", "seg": "조직 분할"}.get(job["task"], job["task"])
    mag = job["coords_dir"].split("_")[0] if job["task"] != "seg" else ""
    print(f"{BOLD}{CYAN}▶ {name} {mag}{RESET}  {job['job_dir']}")
    print(f"  PID {job['pid']} ({job['user']}) | 경과 {_fmt_time(job['elapsed'])} | 작업자 {job['workers'] or 1} | "
          f"{'CPU' if all(g == '-1' for g in job['gpus']) else 'GPU ' + ' '.join(job['gpus'])} | "
          f"중지: kill -- -{job['pgid']}")
    pr = job_progress(job)
    pct = 100 * pr["w_done"] / pr["w_total"] if pr["w_total"] else 0
    line = f"  슬라이드 {pr['done']}/{pr['total']}"
    if pr["patch_based"]:
        line += f" | 패치 약 {_count(pr['w_done'])}/{_count(pr['w_total'])} ({pct:.1f}%)"
    else:
        line += f" ({pct:.1f}%)"
    elapsed = max(job["elapsed"], 1)
    if pr["n_new"]:
        left = (pr["w_total"] - pr["w_done"]) / (pr["w_new"] / elapsed) if pr["w_new"] else 0
        line += f" | 이번 실행 {pr['n_new']}장, {60 * pr['n_new'] / elapsed:.1f}장/분 | 남은 시간 약 {_fmt_time(left)}"
    elif pr["done"] < pr["total"]:
        line += f" | {YELLOW}모델·슬라이드 불러오는 중{RESET}"
    print(line)
    if pr["locked"]:
        print(f"  처리 중 ({len(pr['locked'])}): {', '.join(pr['locked'][:8])}{' ...' if len(pr['locked']) > 8 else ''}")
    if pr["errors"]:
        print(f"  {RED}오류 {len(pr['errors'])}장{RESET} (마지막: {pr['errors'][-1][:140]})")
    gpu_lines = []
    for g in job["gpus"]:
        if g in util:
            u, used, tot = util[g]
            gpu_lines.append(f"GPU {g} {u}% {int(used) / 1024:.1f}/{int(tot) / 1024:.0f} GB")
    job_mem = sum(mem.get(p, 0) for p in job["tree"])
    if gpu_lines:
        print(f"  {' | '.join(gpu_lines)} | 이 작업 {job_mem / 1024:.1f} GB")
    log = launch_log(job)
    if tail and log:
        print(f"  {DIM}── {os.path.basename(log)} (last {tail}) ──{RESET}")
        for line in log_tail(log, tail):
            print(f"  {DIM}{line[:170]}{RESET}")


def lock_host(d, names):
    """Host name TRIDENT wrote into one of these .lock files (create_lock stores pid/hostname as JSON)."""
    for n in names:
        try:
            with open(os.path.join(d, n + ".lock")) as f:
                return json.load(f).get("hostname")
        except (OSError, ValueError):
            continue
    return None


def show_table(emb_root, running):
    print(f"{BOLD}모델 × 배율{RESET} (좌표 / 특징 슬라이드 수, 특징 용량 GB = 10^9 bytes)")
    total = n_slides(WSI_DIR)
    for fm in sorted(d for d in os.listdir(emb_root) if os.path.isdir(os.path.join(emb_root, d))):
        mags = [m for m in os.listdir(os.path.join(emb_root, fm)) if MAG_DIR.match(m)]
        for m in sorted(mags, key=lambda m: -float(MAG_DIR.match(m).group(1))):
            d = os.path.join(emb_root, fm, m)
            coords, _ = scan(os.path.join(d, "patches"), "_patches.h5")
            feats, locked = scan(os.path.join(d, f"features_{fm}"), ".h5")
            gb = sum(s for s, _ in feats.values()) / 1e9
            host = lock_host(os.path.join(d, f"features_{fm}"), locked) if locked else None
            state = (f"{GREEN}✔{RESET}" if coords and len(feats) == len(coords) else
                     f"{CYAN}진행 중{RESET}" if (os.path.join(emb_root, fm), m) in running else
                     f"{CYAN}진행 중 ({host}){RESET}" if host and host != socket.gethostname() else
                     f"{DIM}-{RESET}" if not feats else f"{YELLOW}미완{RESET}")
            print(f"  {fm:<12} {m.split('_')[0]:>4}  좌표 {len(coords):>4}/{total} | "
                  f"특징 {len(feats):>4}/{len(coords):<4} {gb:6.1f} GB  {state}")


def render(args):
    jobs = trident_jobs()
    mem, util = gpu_usage()
    print(f"{BOLD}Sydney embedding monitor{RESET}  {time.strftime('%Y-%m-%d %H:%M:%S')}  "
          f"{DIM}{args.emb_root} (Ctrl+C로 종료){RESET}\n")
    print(f"{BOLD}실행 중인 TRIDENT{RESET}")
    if not jobs:
        print("  (없음)")
    for job in jobs:
        show_job(job, mem, util, args.tail)
        print()
    show_table(args.emb_root, {(j["job_dir"], j["coords_dir"]) for j in jobs if j["task"] != "seg"})


def main():
    p = argparse.ArgumentParser(description="Terminal monitor for TRIDENT embedding extraction")
    p.add_argument("--interval", type=float, default=10)
    p.add_argument("--once", action="store_true")
    p.add_argument("--tail", type=int, default=4, help="Launch log lines to show per running job (0 = none)")
    p.add_argument("--emb_root", default=EMB_ROOT)
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
