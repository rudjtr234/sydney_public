# src/training/compare_site_sydney.py
# 생검 부위 입력 실험 비교: 같은 FM·배율에서 논문 방식(부위 없음) vs 부위 입력 버전(head / film).
# 부위 입력 버전은 test를 두 번 한다 — 부위를 넣은 결과와 같은 모델에 부위를 '모름'으로 넣은 결과.
# 기준선 = 같은 사례에서 병리의사 최초 판독(code_init) vs 재검토(code_review, 학습 정답).
#   atr_by_site.png   FM마다 위축 QWK를 전체·부위별로 (5개 fold test 합침), 회색 선 = 병리의사
#   task_qwk.png      FM마다 5개 항목 test QWK (fold 평균 ± SD). 부위를 넣어서 다른 항목이 나빠지지 않았는지
#   site_compare.csv  FM × 방식별 fold 평균 mean QWK, 항목별 QWK, 부위별 위축 QWK·2등급 이상 오차 (합친 사례 기준)
#
# 사용: python -m src.training.compare_site_sydney --versions 0.1.0 0.2.0 0.2.1 --mag 20

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.training.compare_sydney_mtl import ENCODERS, HUMAN, human_predictions, qwk  # noqa: E402
from src.training.plot_sydney_mtl import INK, INK2, SERIES, SURFACE, TASKS, _fig, _style  # noqa: E402
from src.training.train_sydney_mtl import SITES  # noqa: E402

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SITE_NAMES = {"antrum": "Antrum", "body": "Body", "angle": "Angle"}  # other는 표본이 적어서 '전체'에만
MODE_NAMES = {"none": "Paper (no site)", "head": "A: site → head", "film": "B: site → features"}


def site_of_cases(meta_csv, case_ids):
    """case_id -> SITES name, mapped the same way as train_sydney_mtl.load_cases."""
    region = pd.read_csv(meta_csv, dtype=str).groupby("case_id")["region"].first().loc[case_ids]
    return region.where(region.isin(SITES[:-1]), SITES[-1])


def load_variants(output_root, versions, mag):
    """{encoder: [variant]} with variant = {name, mode, nosite, folds (DataFrame of fold test metrics), pred}."""
    out = {}
    for enc in ENCODERS:
        for v in versions:
            d = os.path.join(output_root, f"sydney_mtl_{enc}_{mag}x_v{v}")
            if not os.path.exists(os.path.join(d, "summary.json")):
                continue
            with open(os.path.join(d, "config.json")) as f:
                mode = json.load(f).get("site_mode", "none")
            fold_dirs = sorted(glob.glob(os.path.join(d, "fold*")), key=lambda x: int(x.rsplit("fold", 1)[1]))
            metrics = [json.load(open(os.path.join(fd, "metrics.json"))) for fd in fold_dirs]
            for nosite in ([False, True] if mode != "none" else [False]):
                key, csv = ("test_nosite", "test_predictions_nosite.csv") if nosite else ("test", "test_predictions.csv")
                pred = pd.concat([pd.read_csv(os.path.join(fd, csv), dtype={"case_id": str}) for fd in fold_dirs])
                out.setdefault(enc, []).append({
                    "name": f"{MODE_NAMES[mode]}{', site unknown' if nosite else ''} (v{v})",
                    "mode": mode, "nosite": nosite, "version": v,
                    "folds": pd.DataFrame([m[key] for m in metrics]),
                    "pred": pred.set_index("case_id").sort_index()})
    return out


def site_rows(pred, site, task="atr"):
    """{site or 'all': (qwk, share of errors >= 2 grades, n)} on pooled predictions."""
    out = {}
    for s in ["all", *SITE_NAMES]:
        m = np.ones(len(pred), bool) if s == "all" else (site.loc[pred.index] == s).to_numpy()
        y, q = pred.loc[m, f"{task}_label"].to_numpy(), pred.loc[m, f"{task}_pred"].to_numpy()
        out[s] = (qwk(y, q), float(np.mean(np.abs(y - q) >= 2)), int(m.sum()))
    return out


def table(variants, human, site):
    entries = [(HUMAN, HUMAN, None, human)]  # 병리의사 줄은 맨 위에 한 번
    entries += [(ENCODERS[enc], v["name"], v["folds"], v["pred"]) for enc, vs in variants.items() for v in vs]
    rows = []
    for model, name, folds, p in entries:
        r = {"model": model, "variant": name}
        if folds is not None:
            r["mean_qwk_fold_mean"] = folds["mean_qwk"].mean()
            r["mean_qwk_fold_sd"] = folds["mean_qwk"].std(ddof=0)
        for t in TASKS:
            r[f"{t}_qwk_pooled"] = qwk(p[f"{t}_label"].to_numpy(), p[f"{t}_pred"].to_numpy())
        for s, (k, big, n) in site_rows(p, site).items():
            r[f"atr_qwk_{s}"], r[f"atr_err2_{s}"], r[f"n_{s}"] = k, big, n
        rows.append(r)
    return pd.DataFrame(rows)


def _marker(v, k):
    """Same color = same method; hollow = the same model run with site unknown."""
    if v["nosite"]:  # 테두리 선이 바깥으로 그려져 커 보이므로 조금 작게
        return dict(marker="o", ms=7.5, color=SERIES[k], mfc=SURFACE, mec=SERIES[k], mew=2, ls="")
    return dict(marker="o", ms=9, color=SERIES[k], mfc=SERIES[k], mec=SURFACE, mew=1.5, ls="")


def _method_index(variants):
    modes = []
    for vs in variants.values():
        for v in vs:
            if (v["mode"], v["version"]) not in modes:
                modes.append((v["mode"], v["version"]))
    return {m: k for k, m in enumerate(modes)}


def plot_atr_by_site(variants, human, site, path, title):
    encs = list(variants)
    fig, axes = _fig(1, len(encs), (4.2 * len(encs), 4.6))
    idx = _method_index(variants)
    groups = ["all", *SITE_NAMES]
    href = site_rows(human, site)
    for ax, enc in zip(axes, encs):
        _style(ax)
        ax.grid(False, axis="x")
        vs = variants[enc]
        width = 0.8 / len(vs)
        lo = 0.0
        for i, g in enumerate(groups):
            ax.plot([i - 0.42, i + 0.42], [href[g][0]] * 2, color=INK2, lw=2, ls=(0, (4, 2)), zorder=1)
            for j, v in enumerate(vs):
                y = site_rows(v["pred"], site)[g][0]
                lo = min(lo, y)
                ax.plot(i - 0.4 + width * (j + 0.5), y, **_marker(v, idx[(v["mode"], v["version"])]), zorder=3)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([f"{'All' if g == 'all' else SITE_NAMES[g]}\nn={href[g][2]}" for g in groups],
                           color=INK, fontsize=9)
        ax.set_title(ENCODERS[enc], color=INK, fontsize=11, loc="left")
        ax.set_ylim(np.floor(lo * 10) / 10 - 0.05, 1.0)  # 표본이 적은 부위는 QWK가 0 아래로 갈 수 있다
        ax.axhline(0, color=INK2, lw=0.8, zorder=1)
    bottom = min(ax.get_ylim()[0] for ax in axes)  # FM끼리 나란히 비교하도록 같은 범위
    for ax in axes:
        ax.set_ylim(bottom, 1.0)
    axes[0].set_ylabel("Atrophy QWK (all cases pooled)", color=INK2, fontsize=10)
    _legend(fig, variants, idx)
    fig.suptitle(title + " — atrophy by biopsy site; dashed = pathologist initial vs review",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_task_qwk(variants, human, path, title):
    encs = list(variants)
    fig, axes = _fig(1, len(encs), (5.0 * len(encs), 4.6))
    idx = _method_index(variants)
    for ax, enc in zip(axes, encs):
        _style(ax)
        ax.grid(False, axis="x")
        vs = variants[enc]
        width = 0.8 / len(vs)
        for i, t in enumerate(TASKS):
            ref = qwk(human[f"{t}_label"].to_numpy(), human[f"{t}_pred"].to_numpy())
            ax.plot([i - 0.42, i + 0.42], [ref, ref], color=INK2, lw=2, ls=(0, (4, 2)), zorder=1)
            for j, v in enumerate(vs):
                x = i - 0.4 + width * (j + 0.5)
                f = v["folds"][f"{t}_qwk"]
                k = idx[(v["mode"], v["version"])]
                ax.plot([x, x], [f.mean() - f.std(ddof=0), f.mean() + f.std(ddof=0)], color=SERIES[k], lw=2,
                        solid_capstyle="round", zorder=2)
                ax.plot(x, f.mean(), **_marker(v, k), zorder=3)
        ax.set_xticks(range(len(TASKS)))
        ax.set_xticklabels([t.upper() for t in TASKS], color=INK, fontsize=9)  # 긴 이름은 패널 안에서 겹친다
        ax.set_title(ENCODERS[enc], color=INK, fontsize=11, loc="left")
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("Test QWK (fold mean ± SD)", color=INK2, fontsize=10)
    _legend(fig, variants, idx)
    fig.suptitle(title + " — all items; dashed = pathologist initial vs review", color=INK, fontsize=11,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def _legend(fig, variants, idx):
    seen, handles = set(), []
    for vs in variants.values():
        for v in vs:
            label = v["name"]
            if label not in seen:
                seen.add(label)
                handles.append(Line2D([], [], label=label, **_marker(v, idx[(v["mode"], v["version"])])))
    handles.append(Line2D([], [], color=INK2, lw=2, ls=(0, (4, 2)), label=f"{HUMAN}: initial vs review"))
    fig.legend(handles=handles, frameon=False, fontsize=9, labelcolor=INK2, ncol=min(len(handles), 6),
               loc="lower center", bbox_to_anchor=(0.5, 0.0))


def main():
    p = argparse.ArgumentParser(description="Compare SydneyMTL runs with and without biopsy-site input")
    p.add_argument("--versions", nargs="+", required=True, help="e.g. 0.1.0 0.2.0 0.2.1 (paper first)")
    p.add_argument("--mag", default="20")
    p.add_argument("--output_root", default=os.path.join(PROJECT_DIR, "outputs"))
    p.add_argument("--meta_csv", default=os.path.join(PROJECT_DIR, "metadata", "sydney_meta.csv"))
    a = p.parse_args()
    os.umask(0)

    variants = load_variants(a.output_root, a.versions, a.mag)
    if not variants:
        sys.exit(f"No runs of v{a.versions} {a.mag}x in {a.output_root}")
    case_ids = next(iter(variants.values()))[0]["pred"].index
    human = human_predictions(a.meta_csv, case_ids)
    site = site_of_cases(a.meta_csv, case_ids)

    out = os.path.join(a.output_root, f"compare_site_v{'_'.join(a.versions)}_{a.mag}x")
    os.makedirs(out, exist_ok=True)
    title = f"SydneyMTL {a.mag}x, {len(case_ids)} cases"
    tab = table(variants, human, site)
    tab.round(4).to_csv(os.path.join(out, "site_compare.csv"), index=False)
    plot_atr_by_site(variants, human, site, os.path.join(out, "atr_by_site.png"), title)
    plot_task_qwk(variants, human, os.path.join(out, "task_qwk.png"), title)
    cols = ["model", "variant", "mean_qwk_fold_mean", "atr_qwk_all", "atr_qwk_antrum", "atr_qwk_body",
            "atr_qwk_angle", "atr_err2_all"]
    print(tab[cols].round(3).to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
