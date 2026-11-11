# src/training/heatmap_sydney_mtl.py
# 학습이 끝난 SydneyMTL 결과에서 항목별 attention heatmap을 그린다 (모델이 슬라이드의 어디를 보고 등급을 냈는지).
#   사례가 test에 들어간 fold의 best_model.pt로 그 사례의 패치 특징을 다시 통과시켜 항목별 attention(패치 softmax)을 얻는다.
#   <run_dir>/heatmaps/<case>__<slide>.png  H&E + 항목 5개 heatmap (색 = 그 사례 안에서 attention 순위, 빨강 = 높음)
#   <run_dir>/heatmaps/<case>__top.png      항목별 attention 상위 패치 (원본 40x에서 패치 영역을 잘라 옴)
#   <run_dir>/heatmaps/index.csv            그린 사례, 고른 이유, fold, 정답/예측
# 사례를 주지 않으면 항목마다 "최고 등급을 맞힌 사례"와 "가장 크게 틀린 사례"를 --pick개씩 고른다.
#
# 사용 (프로젝트 폴더에서, sydney env):
#   python -m src.training.heatmap_sydney_mtl --run_dir outputs/sydney_mtl_hoptimus0_20x_v0.1.0
#   python -m src.training.heatmap_sydney_mtl --run_dir outputs/sydney_mtl_hoptimus0_20x_v0.1.0 --cases CASE-0001

import argparse
import glob
import json
import os
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import openslide  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.sydney_mtl import SydneyMTL  # noqa: E402
from src.training.plot_sydney_mtl import INK, INK2, SURFACE, TASK_NAMES, TASKS  # noqa: E402

WSI_DIR = "/path/to/sydney/wsi_tiff"
CMAP = plt.get_cmap("jet")


def load_predictions(run_dir):
    """All folds' test predictions in one table (each case is in exactly one fold's test set)."""
    dfs = []
    for f in sorted(glob.glob(os.path.join(run_dir, "fold*", "test_predictions.csv"))):
        d = pd.read_csv(f, dtype={"case_id": str, "slide_ids": str, "code_label": str, "code_pred": str})
        d["fold"] = int(os.path.basename(os.path.dirname(f))[4:])
        dfs.append(d)
    return pd.concat(dfs).set_index("case_id")


def pick_cases(pred, n):
    """Per task: n cases at the highest label grade predicted right (most confident), n biggest misses."""
    out = {}
    for t in TASKS:
        lab, prd = pred[f"{t}_label"], pred[f"{t}_pred"]
        prob_lab = pred[[f"{t}_prob{g}" for g in range(4)]].to_numpy()[np.arange(len(pred)), lab.to_numpy()]
        hit = pred[(lab == prd)].assign(p=prob_lab[(lab == prd).to_numpy()])
        top = hit[hit[f"{t}_label"] == hit[f"{t}_label"].max()].sort_values("p", ascending=False).head(n)
        for c in top.index:
            out.setdefault(c, []).append(f"{t} grade {lab[c]} correct")
        err = (prd - lab).abs()
        for c in err[err > 0].sort_values(ascending=False, kind="stable").head(n).index:
            out.setdefault(c, []).append(f"{t} label {lab[c]} pred {prd[c]}")
    return out


def case_attention(run_dir, row, feat_dir, device):
    """{slide_id: (coords [n, 2], patch_size_level0, {task: attention percentile [n]})} for one case."""
    ckpt = torch.load(os.path.join(run_dir, f"fold{row['fold']}", "best_model.pt"), map_location=device,
                      weights_only=False)
    sites = ckpt.get("sites", [])
    model = SydneyMTL(ckpt["in_dim"], {t: ckpt["num_grades"] for t in ckpt["tasks"]}, ckpt["adaptor_dim"],
                      ckpt.get("site_mode", "none"), len(sites)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    slides, feats = [], []
    for s in row["slide_ids"].split(";"):
        with h5py.File(os.path.join(feat_dir, f"{s}.h5")) as f:
            feats.append(f["features"][:])
            slides.append((s, f["coords"][:], float(f["coords"].attrs["patch_size_level0"])))
    with torch.no_grad():
        site = 1 + sites.index(row["site"]) if row.get("site") in sites else 0  # 부위 입력 모델은 그 사례의 부위로
        _, att = model(torch.from_numpy(np.concatenate(feats)).float().unsqueeze(0).to(device),
                       torch.tensor([site], device=device))
    # attention은 사례 전체 패치에 대한 softmax라 값 자체가 작다 → 사례 안 순위(0~1)로 바꿔서 그린다
    rank = {t: rankdata(att[t][0].cpu().numpy()) / att[t].shape[1] for t in TASKS}
    out, i = {}, 0
    for s, coords, ps in slides:
        out[s] = (coords, ps, {t: rank[t][i:i + len(coords)] for t in TASKS})
        i += len(coords)
    return out


def read_thumbnail(slide_path, downsample):
    wsi = openslide.OpenSlide(slide_path)
    level = wsi.get_best_level_for_downsample(downsample)
    size = tuple(int(d / downsample) for d in wsi.level_dimensions[0])
    img = wsi.read_region((0, 0), level, wsi.level_dimensions[level]).convert("RGB").resize(size, Image.BILINEAR)
    return wsi, np.asarray(img)


def blend(img, coords, ps, score, downsample, alpha=0.45):
    """Paint each patch with the colormap of its score; tissue outside patches keeps the H&E."""
    out = img.astype(np.float32).copy()
    s = ps / downsample
    for (x, y), v in zip(coords / downsample, score):
        x0, y0 = int(x), int(y)
        x1, y1 = min(int(x + s) + 1, out.shape[1]), min(int(y + s) + 1, out.shape[0])
        out[y0:y1, x0:x1] = (1 - alpha) * out[y0:y1, x0:x1] + alpha * 255 * np.array(CMAP(v)[:3])
    return out.astype(np.uint8)


def sections(coords, ps, wide):
    """Level-0 boxes (x0, y0, x1, y1) of tissue pieces: patches split where the gap along the long axis > 3 patches."""
    ax = 0 if wide else 1
    order = np.argsort(coords[:, ax], kind="stable")
    cuts = np.where(np.diff(coords[order, ax]) > 3 * ps)[0] + 1
    return [(c[:, 0].min() - ps, c[:, 1].min() - ps, c[:, 0].max() + 2 * ps, c[:, 1].max() + 2 * ps)
            for c in (coords[i] for i in np.split(order, cuts))]


def crop_concat(img, boxes, downsample, wide, gap=12):
    """Crop each tissue piece and put them side by side (wide slide) or on top of each other, dropping empty glass."""
    parts = [img[max(int(y0 / downsample), 0):int(y1 / downsample), max(int(x0 / downsample), 0):int(x1 / downsample)]
             for x0, y0, x1, y1 in boxes]
    if wide:
        h = max(p.shape[0] for p in parts)
        parts = [np.pad(p, ((0, h - p.shape[0]), (0, gap), (0, 0)), constant_values=255) for p in parts]
        return np.hstack(parts)[:, :-gap]
    w = max(p.shape[1] for p in parts)
    parts = [np.pad(p, ((0, gap), (0, w - p.shape[1]), (0, 0)), constant_values=255) for p in parts]
    return np.vstack(parts)[:-gap]


def task_title(row, t):
    p = row[f"{t}_prob{row[f'{t}_pred']}"]
    return f"{TASK_NAMES[t]}   label {row[f'{t}_label']} · pred {row[f'{t}_pred']} (p {p:.2f})"


def draw_slide(path, case_id, slide_id, row, img, coords, ps, ranks, downsample):
    wide = img.shape[1] >= img.shape[0]
    boxes = sections(coords, ps, wide)
    panels = [("H&E", crop_concat(img, boxes, downsample, wide))] + [
        (task_title(row, t), crop_concat(blend(img, coords, ps, ranks[t], downsample), boxes, downsample, wide))
        for t in TASKS]
    h, w = panels[0][1].shape[:2]
    ncols = 1 if w / h >= 3 else 2 if w / h >= 1.2 else 3 if w / h >= 0.5 else 6
    nrows = 6 // ncols
    pw = 16 / ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * (pw * h / w + 0.35) + 0.6), squeeze=False)
    for ax, (title, im) in zip(axes.T.ravel() if ncols == 2 else axes.ravel(), panels):
        ax.imshow(im)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.axis("off")
    fig.suptitle(f"{case_id} / {slide_id}   CODE label {row['code_label']} · pred {row['code_pred']}   "
                 f"(fold {row['fold']} test; color = attention rank within case, red = high)",
                 fontsize=11, color=INK2)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def draw_top(path, case_id, row, wsis, att, k):
    fig, axes = plt.subplots(len(TASKS), k, figsize=(1.9 * k, 2.1 * len(TASKS)))
    for r, t in enumerate(TASKS):
        cand = [(v, s, c, ps) for s, (coords, ps, ranks) in att.items() for v, c in zip(ranks[t], coords)]
        cand.sort(key=lambda x: -x[0])
        for j in range(k):
            ax = axes[r, j]
            ax.axis("off")
            if j < len(cand):
                _, s, (x, y), ps = cand[j]
                ax.imshow(wsis[s].read_region((int(x), int(y)), 0, (int(ps), int(ps))).convert("RGB").resize((224, 224)))
            if j == 0:
                ax.set_title(task_title(row, t), fontsize=8, color=INK, loc="left")
    fig.suptitle(f"{case_id}   top-{k} attention patches per item", fontsize=10, color=INK2)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="SydneyMTL attention heatmaps")
    p.add_argument("--run_dir", required=True, help="outputs/sydney_mtl_<fm>_<mag>x_v<ver>")
    p.add_argument("--cases", nargs="+", default=None, help="case_id list (default: pick per task)")
    p.add_argument("--pick", type=int, default=1, help="cases per task and kind when --cases is not given")
    p.add_argument("--downsample", type=float, default=16, help="heatmap resolution vs level 0 (40x)")
    p.add_argument("--top_k", type=int, default=6, help="top attention patches per item")
    p.add_argument("--gpu", type=int, default=-1)
    a = p.parse_args()
    os.umask(0)

    with open(os.path.join(a.run_dir, "config.json")) as f:
        cfg = json.load(f)
    pred = load_predictions(a.run_dir)
    reasons = {c: ["requested"] for c in a.cases} if a.cases else pick_cases(pred, a.pick)
    device = torch.device(f"cuda:{a.gpu}" if a.gpu >= 0 else "cpu")
    out_dir = os.path.join(a.run_dir, "heatmaps")
    os.makedirs(out_dir, exist_ok=True)

    index = []
    for case_id, why in reasons.items():
        row = pred.loc[case_id]
        att = case_attention(a.run_dir, row, cfg["feat_dir"], device)
        wsis = {}
        for s, (coords, ps, ranks) in att.items():
            wsis[s], img = read_thumbnail(os.path.join(WSI_DIR, f"{s}.tiff"), a.downsample)
            draw_slide(os.path.join(out_dir, f"{case_id}__{s}.png"), case_id, s, row, img, coords, ps, ranks,
                       a.downsample)
        draw_top(os.path.join(out_dir, f"{case_id}__top.png"), case_id, row, wsis, att, a.top_k)
        index.append({"case_id": case_id, "reason": "; ".join(why), "fold": row["fold"], "slides": row["slide_ids"],
                      "code_label": row["code_label"], "code_pred": row["code_pred"]})
        print(f"{case_id}: {'; '.join(why)}")
    pd.DataFrame(index).to_csv(os.path.join(out_dir, "index.csv"), index=False)
    print(f"{len(index)} cases -> {out_dir}")


if __name__ == "__main__":
    main()
