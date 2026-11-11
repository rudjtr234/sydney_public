# src/training/compare_sydney_mtl.py
# 같은 버전·배율로 학습한 FM 결과 여러 개를 한 그림으로 비교한다 (논문 주 지표 QWK·lenient accuracy + 등급별로 뜯어 보기).
# 기준선 = 같은 사례에서 병리의사 최초 판독(code_init) vs 재검토(code_review, 학습 정답).
#   metrics.png           항목별 test QWK·lenient accuracy. 점 = fold, 큰 점·막대 = fold 평균 ± SD, 회색 선 = 병리의사
#   grade_recall.png      항목·정답 등급별로 정확히 맞힌 비율 (5개 fold test 합침). 드문 고등급을 놓치는지 본다
#   error_size.png        항목별 오차 크기 비율 (정확 / 1등급 차이 / 2 / 3). QWK가 크게 깎이는 큰 오차가 얼마나 되는지
#   confusion_compare.png 항목(행) × 모델(열) 혼동행렬, 정답 등급별 % (행 합 100%)
#   pooled_metrics.csv    5개 fold test를 합친 전체 사례 기준 지표 (병리의사, FM, FM 확률 평균 앙상블)
#
# 사용: python -m src.training.compare_sydney_mtl --version 0.1.0 --mag 20

import argparse
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.training.plot_sydney_mtl import (BLUES, INK, INK2, LENIENT_MAP, SERIES, SURFACE, TASK_NAMES,  # noqa: E402
                                          TASKS, _fig, _style)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENCODERS = {"hibou_l": "Hibou-L", "hoptimus0": "H-Optimus-0", "virchow": "Virchow", "midnight12k": "Midnight-12k"}
HUMAN = "Pathologist"  # 최초 판독 vs 재검토
ERROR_STEPS = ["#b7d3f6", "#6da7ec", "#2a78d6", "#184f95"]  # dataviz 기본 sequential blue (150/300/450/600)


def lenient(y, p):
    return float(np.mean([pp in LENIENT_MAP[int(yy)] for yy, pp in zip(y, p)]))


def qwk(y, p):
    return float(cohen_kappa_score(y, p, labels=[0, 1, 2, 3], weights="quadratic"))


def load_runs(output_root, version, mag):
    """{display name: {"folds": DataFrame of fold test metrics, "pred": pooled test predictions}} in fixed FM order."""
    runs = {}
    for enc, name in ENCODERS.items():
        d = os.path.join(output_root, f"sydney_mtl_{enc}_{mag}x_v{version}")
        if not os.path.exists(os.path.join(d, "summary.json")):
            continue
        with open(os.path.join(d, "summary.json")) as f:
            folds = pd.DataFrame(json.load(f)["folds"])
        pred = pd.concat([pd.read_csv(p, dtype={"case_id": str, "code_label": str, "code_pred": str})
                          for p in glob.glob(os.path.join(d, "fold*", "test_predictions.csv"))])
        runs[name] = {"folds": folds, "pred": pred.set_index("case_id").sort_index()}
    return runs


def human_predictions(meta_csv, case_ids):
    """Original pathologist grades (code_init) laid out like a model's pooled predictions."""
    meta = pd.read_csv(meta_csv, dtype=str).groupby("case_id").first().loc[case_ids]
    out = pd.DataFrame(index=case_ids)
    for i, t in enumerate(TASKS):
        out[f"{t}_label"] = meta["code_review"].str[i].astype(int)
        out[f"{t}_pred"] = meta["code_init"].str[i].astype(int)
    return out


def pooled_table(preds, runs):
    rows = []
    ens = {"Ensemble H-Optimus-0 + Midnight-12k": ["H-Optimus-0", "Midnight-12k"], "Ensemble all FMs": list(runs)}
    for name, members in ens.items():
        if not all(m in runs for m in members):
            continue
        first = runs[members[0]]["pred"]
        e = pd.DataFrame(index=first.index)
        for t in TASKS:
            prob = sum(runs[m]["pred"].loc[e.index, [f"{t}_prob{g}" for g in range(4)]].to_numpy() for m in members)
            e[f"{t}_label"], e[f"{t}_pred"] = first[f"{t}_label"], prob.argmax(1)
        preds = {**preds, name: e}
    for name, p in preds.items():
        r = {"model": name}
        for t in TASKS:
            y, q = p[f"{t}_label"].to_numpy(), p[f"{t}_pred"].to_numpy()
            r[f"{t}_qwk"], r[f"{t}_lenient"] = qwk(y, q), lenient(y, q)
            r[f"{t}_balanced_acc"] = float(balanced_accuracy_score(y, q))
        r["mean_qwk"] = np.mean([r[f"{t}_qwk"] for t in TASKS])
        r["mean_lenient"] = np.mean([r[f"{t}_lenient"] for t in TASKS])
        rows.append(r)
    return pd.DataFrame(rows)


def plot_metrics(runs, human, path, title):
    fig, axes = _fig(2, 1, (12, 8.4))
    names = list(runs)
    width = 0.8 / len(names)
    for ax, (m, label) in zip(axes, [("qwk", "QWK"), ("lenient_accuracy", "Lenient accuracy")]):
        _style(ax)
        ax.grid(False, axis="x")
        for i, t in enumerate(TASKS):
            ref = human[f"{t}_{'qwk' if m == 'qwk' else 'lenient'}"]
            ax.plot([i - 0.44, i + 0.44], [ref, ref], color=INK2, lw=2, ls=(0, (4, 2)), zorder=1)
            for k, n in enumerate(names):
                v = runs[n]["folds"][f"{t}_{m}"].to_numpy()
                x = i - 0.4 + width * (k + 0.5)
                ax.scatter(np.full(len(v), x - width * 0.32), v, s=14, color=SERIES[k], alpha=0.5, linewidths=0,
                           zorder=2)  # fold 점은 평균 막대 왼쪽에 (겹치면 막대가 끊겨 보임)
                mu, sd = v.mean(), v.std(ddof=0)
                ax.plot([x, x], [mu - sd, mu + sd], color=SERIES[k], lw=2.5, solid_capstyle="round", zorder=3)
                ax.plot(x, mu, "o", ms=9, color=SERIES[k], mec=SURFACE, mew=2, zorder=4)
        ax.set_xticks(range(len(TASKS)))
        ax.set_xticklabels([TASK_NAMES[t] for t in TASKS], color=INK, fontsize=10)
        ax.set_ylabel(label, color=INK2, fontsize=10)
        lo = min(min(runs[n]["folds"][f"{t}_{m}"].min() for n in names for t in TASKS),
                 min(human[f"{t}_{'qwk' if m == 'qwk' else 'lenient'}"] for t in TASKS))
        ax.set_ylim(max(0, np.floor(lo * 10) / 10 - 0.05), 1.0)
    handles = [Line2D([], [], marker="o", ls="", ms=8, color=SERIES[k], mec=SURFACE, mew=2,
                      label=f"{n} (mean QWK {runs[n]['folds']['mean_qwk'].mean():.3f})") for k, n in enumerate(names)]
    handles.append(Line2D([], [], color=INK2, lw=2, ls=(0, (4, 2)),
                          label=f"{HUMAN}: initial vs review (mean QWK {human['mean_qwk']:.3f})"))
    axes[1].legend(handles=handles, frameon=False, fontsize=9, labelcolor=INK2, ncol=3, loc="upper center",
                   bbox_to_anchor=(0.5, -0.1))
    axes[0].set_title(title + "\nsmall dots = folds, large dot ± bar = fold mean ± SD, dashed = pathologist agreement "
                      "(all cases pooled)", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_grade_recall(preds, path, title):
    names = list(preds)
    fig, axes = _fig(1, len(TASKS), (4.0 * len(TASKS), 4.6))
    models = [n for n in names if n != HUMAN]
    width = 0.8 / len(models)
    for ax, t in zip(axes, TASKS):
        _style(ax)
        ax.grid(False, axis="x")
        y = preds[models[0]][f"{t}_label"].to_numpy()
        counts = np.bincount(y, minlength=4)
        for k, n in enumerate(models):
            p = preds[n][f"{t}_pred"].to_numpy()
            rec = [np.mean(p[y == g] == g) if counts[g] else np.nan for g in range(4)]
            ax.bar(np.arange(4) - 0.4 + width * (k + 0.5), rec, width, color=SERIES[k], edgecolor=SURFACE,
                   linewidth=1.5, label=n)
        hp = preds[HUMAN][f"{t}_pred"].to_numpy()
        hrec = [np.mean(hp[y == g] == g) if counts[g] else np.nan for g in range(4)]
        ax.scatter(np.arange(4), hrec, marker="_", s=420, color=INK, linewidths=2.5, zorder=4, label=HUMAN)
        ax.set_xticks(range(4))
        ax.set_xticklabels([f"{g}\nn={counts[g]}" for g in range(4)], color=INK2, fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.set_title(TASK_NAMES[t], color=INK, fontsize=10)
        ax.set_xlabel("True grade", color=INK2, fontsize=9)
    axes[0].set_ylabel("Share predicted exactly", color=INK2, fontsize=10)
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper right")
    fig.suptitle(title + "  |  recall per true grade, 5 folds' test pooled (black dash = pathologist initial read)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_error_size(preds, path, title):
    names = [HUMAN] + [n for n in preds if n != HUMAN]
    fig, axes = _fig(1, len(TASKS), (4.0 * len(TASKS), 3.6))
    for ax, t in zip(axes, TASKS):
        ax.set_facecolor(SURFACE)
        for s in ax.spines.values():
            s.set_visible(False)
        for r, n in enumerate(names):
            d = np.abs(preds[n][f"{t}_pred"].to_numpy() - preds[n][f"{t}_label"].to_numpy())
            share = np.bincount(d, minlength=4)[:4] / len(d)
            left = 0
            for e, v in enumerate(share):
                ax.barh(r, v, left=left, height=0.62, color=ERROR_STEPS[e], edgecolor=SURFACE, linewidth=1.5)
                left += v
            big = share[2] + share[3]
            ax.text(1.02, r, f"≥2: {big:.1%}", va="center", fontsize=8.5, color=INK2)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names if t == TASKS[0] else [], color=INK, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.0)
        ax.tick_params(colors=INK2, labelsize=8, length=0)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "50%", "100%"])
        ax.set_title(TASK_NAMES[t], color=INK, fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ERROR_STEPS]
    fig.legend(handles, ["exact", "off by 1", "off by 2", "off by 3"], frameon=False, fontsize=9, labelcolor=INK2,
               ncol=4, loc="lower center")
    fig.suptitle(title + "  |  size of grade error, 5 folds' test pooled (≥2 = errors QWK punishes most)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 0.97, 0.92))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_confusion_compare(preds, path, title):
    names = [HUMAN] + [n for n in preds if n != HUMAN]
    fig, axes = _fig(len(TASKS), len(names), (2.7 * len(names), 2.7 * len(TASKS) + 0.6))
    for r, t in enumerate(TASKS):
        for c, n in enumerate(names):
            ax = axes[r, c]
            y, p = preds[n][f"{t}_label"].to_numpy(), preds[n][f"{t}_pred"].to_numpy()
            cm = np.zeros((4, 4), dtype=int)
            np.add.at(cm, (y, p), 1)
            shown = cm / np.maximum(cm.sum(1, keepdims=True), 1)
            ax.imshow(shown, cmap=BLUES, vmin=0, vmax=1)
            for i in range(4):
                for j in range(4):
                    if cm[i, j]:
                        ax.text(j, i, f"{shown[i, j]:.0%}", ha="center", va="center", fontsize=7.5,
                                color="white" if shown[i, j] > 0.55 else INK)
            ax.set_xticks(range(4))
            ax.set_yticks(range(4))
            ax.tick_params(colors=INK2, labelsize=7.5, length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(f"{n}\nQWK {qwk(y, p):.3f}" if r == 0 else f"QWK {qwk(y, p):.3f}", color=INK, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{TASK_NAMES[t]}\ntrue grade", color=INK, fontsize=9)
            if r == len(TASKS) - 1:
                ax.set_xlabel("predicted grade", color=INK2, fontsize=8)
    fig.suptitle(title + "  |  rows = true grade, % of that true grade (5 folds' test pooled)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Compare SydneyMTL runs of several foundation models")
    p.add_argument("--version", required=True)
    p.add_argument("--mag", default="20")
    p.add_argument("--output_root", default=os.path.join(PROJECT_DIR, "outputs"))
    p.add_argument("--meta_csv", default=os.path.join(PROJECT_DIR, "metadata", "sydney_meta.csv"))
    a = p.parse_args()
    os.umask(0)

    runs = load_runs(a.output_root, a.version, a.mag)
    if len(runs) < 2:
        sys.exit(f"Need at least 2 runs of v{a.version} {a.mag}x in {a.output_root}, found {list(runs)}")
    splits = {open(os.path.join(a.output_root, f"sydney_mtl_{e}_{a.mag}x_v{a.version}", "splits.json")).read()
              for e, n in ENCODERS.items() if n in runs}
    if len(splits) != 1:
        print("WARNING: runs use different splits.json; fold-level comparison is not paired")
    case_ids = runs[next(iter(runs))]["pred"].index
    preds = {HUMAN: human_predictions(a.meta_csv, case_ids), **{n: r["pred"] for n, r in runs.items()}}
    table = pooled_table(preds, runs)
    human = table.set_index("model").loc[HUMAN].to_dict()

    out = os.path.join(a.output_root, f"compare_v{a.version}_{a.mag}x")
    os.makedirs(out, exist_ok=True)
    title = f"SydneyMTL v{a.version}, {a.mag}x, {len(case_ids)} cases"
    table.round(4).to_csv(os.path.join(out, "pooled_metrics.csv"), index=False)
    plot_metrics(runs, human, os.path.join(out, "metrics.png"), title)
    plot_grade_recall(preds, os.path.join(out, "grade_recall.png"), title)
    plot_error_size(preds, os.path.join(out, "error_size.png"), title)
    plot_confusion_compare(preds, os.path.join(out, "confusion_compare.png"), title)
    print(table[["model", "mean_qwk", "mean_lenient"] + [f"{t}_qwk" for t in TASKS]].round(3).to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
