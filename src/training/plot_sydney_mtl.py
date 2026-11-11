# src/training/plot_sydney_mtl.py
# train_sydney_mtl.py 결과를 그림(PNG)으로 저장한다. 학습 중 fold가 끝날 때마다·전체가 끝날 때 자동으로 불리고,
# 이미 끝난 결과에 다시 그릴 때는 직접 실행한다.
#   foldN/curves.png     epoch별 train/val loss, val 평균 QWK·lenient accuracy, 항목별 val QWK (점선 = best epoch)
#   foldN/confusion.png  항목별 test 혼동행렬 (행 = 정답, 열 = 예측, 칸 = 사례 수)
#   summary_metrics.png  항목별 test QWK·lenient accuracy·accuracy (fold 평균, 오차 막대 = 표준편차)
#   confusion_all.png    5개 fold의 test 예측을 합친 혼동행렬 (모든 사례가 한 번씩 test에 들어감)
#
# 사용: python -m src.training.plot_sydney_mtl --out_dir outputs/sydney_mtl/hibou_l_la_review

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import cohen_kappa_score  # noqa: E402

TASKS = ["hp", "neu", "mono", "atr", "im"]
TASK_NAMES = {"hp": "H. pylori", "neu": "Neutrophil", "mono": "Mononuclear", "atr": "Atrophy",
              "im": "Intestinal metaplasia"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # dataviz 기본 categorical 순서 (slot 1-5)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUES = LinearSegmentedColormap.from_list("seq_blue", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
LENIENT_MAP = {0: {0}, 1: {1, 2}, 2: {1, 2, 3}, 3: {2, 3}}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _fig(nrows, ncols, size):
    fig, axes = plt.subplots(nrows, ncols, figsize=size, facecolor=SURFACE)
    return fig, np.atleast_1d(axes)


def plot_curves(fold_dir, best_epoch=None):
    h = pd.read_csv(os.path.join(fold_dir, "history.csv"))
    if best_epoch is None:
        best_epoch = int(h.loc[h["val_loss"].idxmin(), "epoch"])
    fig, axes = _fig(1, 3, (16, 4.4))
    panels = [
        ("Total loss (sum of 5 tasks)", [("train_loss", "Train"), ("val_loss", "Validation")], None),
        ("Validation mean over 5 tasks", [("val_mean_qwk", "QWK"), ("val_mean_lenient_accuracy", "Lenient accuracy")],
         (0, 1)),
        ("Validation QWK by task", [(f"val_{t}_qwk", TASK_NAMES[t]) for t in TASKS], (-0.1, 1)),
    ]
    for ax, (title, series, ylim) in zip(axes, panels):
        _style(ax)
        for k, (col, label) in enumerate(series):
            ax.plot(h["epoch"], h[col], color=SERIES[k], linewidth=2, label=label)
        ax.axvline(best_epoch, color=INK2, linestyle="--", linewidth=1)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("Epoch", color=INK2, fontsize=9)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.suptitle(f"{os.path.basename(os.path.normpath(fold_dir))}  |  best epoch {best_epoch} (dashed)",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(fold_dir, "curves.png"), dpi=130, facecolor=SURFACE)
    plt.close(fig)


def _confusion_panels(pred, title, path, normalize_rows=False):
    fig, axes = _fig(1, len(TASKS), (4.0 * len(TASKS), 4.9))
    for ax, t in zip(axes, TASKS):
        y, p = pred[f"{t}_label"].to_numpy(), pred[f"{t}_pred"].to_numpy()
        cm = np.zeros((4, 4), dtype=int)
        np.add.at(cm, (y, p), 1)
        shown = cm / np.maximum(cm.sum(1, keepdims=True), 1) if normalize_rows else cm / max(cm.max(), 1)
        ax.imshow(shown, cmap=BLUES, vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                txt = f"{cm[i, j]}" + (f"\n{shown[i, j]:.0%}" if normalize_rows and cm[i].sum() else "")
                ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                        color="white" if shown[i, j] > 0.55 else INK)
        qwk = cohen_kappa_score(y, p, labels=[0, 1, 2, 3], weights="quadratic")
        lenient = np.mean([pp in LENIENT_MAP[int(yy)] for yy, pp in zip(y, p)])
        ax.set_title(f"{TASK_NAMES[t]}\nQWK {qwk:.3f} | lenient {lenient:.3f}", color=INK, fontsize=10)
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xlabel("Predicted grade", color=INK2, fontsize=9)
        ax.set_ylabel("True grade", color=INK2, fontsize=9)
        ax.tick_params(colors=INK2, labelsize=9, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_confusion(fold_dir):
    pred = pd.read_csv(os.path.join(fold_dir, "test_predictions.csv"), dtype={"code_label": str, "code_pred": str})
    _confusion_panels(pred, f"{os.path.basename(os.path.normpath(fold_dir))} test (n={len(pred)})  |  "
                            "rows = true grade, columns = predicted grade",
                      os.path.join(fold_dir, "confusion.png"))


def plot_fold(fold_dir, best_epoch=None):
    plot_curves(fold_dir, best_epoch)
    if os.path.exists(os.path.join(fold_dir, "test_predictions.csv")):
        plot_confusion(fold_dir)


def plot_run(out_dir):
    fold_dirs = sorted(glob.glob(os.path.join(out_dir, "fold*")), key=lambda d: int(d.rsplit("fold", 1)[1]))
    metrics = []
    for d in fold_dirs:
        if os.path.exists(os.path.join(d, "metrics.json")):
            with open(os.path.join(d, "metrics.json")) as f:
                metrics.append(json.load(f)["test"])
    if not metrics:
        return
    df = pd.DataFrame(metrics)
    fig, axes = _fig(1, 1, (11, 4.6))
    ax = axes[0]
    _style(ax)
    ax.grid(False, axis="x")
    names = [("qwk", "QWK"), ("lenient_accuracy", "Lenient accuracy"), ("accuracy", "Accuracy")]
    x, width = np.arange(len(TASKS)), 0.26
    for k, (m, label) in enumerate(names):
        mean = [df[f"{t}_{m}"].mean() for t in TASKS]
        std = [df[f"{t}_{m}"].std(ddof=0) for t in TASKS]
        bars = ax.bar(x + (k - 1) * width, mean, width - 0.03, yerr=std, color=SERIES[k], label=label,
                      error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 3})
        for b, v in zip(bars, mean):
            ax.text(b.get_x() + b.get_width() / 2, 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8,
                    color="white" if k == 0 else INK)
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_NAMES[t] for t in TASKS], color=INK, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.09))
    ax.set_title(f"Test metrics by task, mean of {len(df)} folds (error bar = SD)\n"
                 f"mean QWK {df['mean_qwk'].mean():.3f}  |  mean lenient accuracy "
                 f"{df['mean_lenient_accuracy'].mean():.3f}", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "summary_metrics.png"), dpi=130, facecolor=SURFACE)
    plt.close(fig)

    preds = [pd.read_csv(os.path.join(d, "test_predictions.csv")) for d in fold_dirs
             if os.path.exists(os.path.join(d, "test_predictions.csv"))]
    pooled = pd.concat(preds, ignore_index=True)
    _confusion_panels(pooled, f"All {len(preds)} folds' test predictions pooled (n={len(pooled)})  |  "
                              "rows = true grade, columns = predicted grade, % = share of the true grade",
                      os.path.join(out_dir, "confusion_all.png"), normalize_rows=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    a = p.parse_args()
    for d in sorted(glob.glob(os.path.join(a.out_dir, "fold*"))):
        if os.path.exists(os.path.join(d, "history.csv")):
            best = None
            if os.path.exists(os.path.join(d, "metrics.json")):
                with open(os.path.join(d, "metrics.json")) as f:
                    best = json.load(f)["best_epoch"]
            plot_fold(d, best)
    plot_run(a.out_dir)


if __name__ == "__main__":
    main()
