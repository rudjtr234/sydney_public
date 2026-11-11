# scripts/check_embeddings.py
# 추출 결과 점검: 인코더별 h5 개수/차원/패치 수, 특징이 없는 슬라이드 목록,
# 그리고 메타데이터를 채울 템플릿 CSV(slide_id, patient_id, label) 생성
#
#   python scripts/check_embeddings.py --job_dir /path/to/sydney/embedding \
#       --wsi_dir /path/to/sydney/wsi_tiff --out_dir /path/to/sydney_project/metadata

import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import read_feature_info  # noqa: E402

DEFAULT_ENCODERS = ["hibou_l", "hoptimus0", "virchow", "midnight12k"]
# TRIDENT 실행 시 쓰는 4개 인코더의 원래 출력 차원 (midnight12k는 TRIDENT 기본값 = CLS token 1536)
EXPECTED_DIM = {"hibou_l": 1024, "hoptimus0": 1536, "virchow": 2560, "midnight12k": 1536}


def list_wsis(wsi_dir):
    if not wsi_dir or not os.path.isdir(wsi_dir):
        return None
    stems = set()
    for root, _, files in os.walk(wsi_dir):
        for fn in files:
            stem, ext = os.path.splitext(fn)
            if ext.lower() in {".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn", ".bif", ".vms", ".svslide",
                               ".czi", ".sdpc", ".dcm", ".jpg", ".jpeg", ".png"}:
                stems.add(stem)
    return stems


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--job_dir", default="/path/to/sydney/embedding")
    p.add_argument("--wsi_dir", default="/path/to/sydney/wsi_tiff")
    p.add_argument("--coords_subdir", default="20x_224px_0px_overlap")
    p.add_argument("--encoders", nargs="+", default=DEFAULT_ENCODERS)
    p.add_argument("--out_dir", default=None, help="Write slide_inventory.csv + metadata_template.csv here")
    args = p.parse_args()

    base = os.path.join(args.job_dir, args.coords_subdir)
    wsis = list_wsis(args.wsi_dir)
    coords = {os.path.basename(f)[: -len("_patches.h5")] for f in glob.glob(os.path.join(base, "patches", "*_patches.h5"))}
    print(f"WSIs in {args.wsi_dir}: {len(wsis) if wsis is not None else 'n/a'}")
    print(f"Patch coords: {len(coords)}")

    inventory = {}
    for enc in args.encoders:
        files = sorted(glob.glob(os.path.join(base, f"features_{enc}", "*.h5")))
        dims, bad = set(), []
        for f in files:
            sid = os.path.splitext(os.path.basename(f))[0]
            try:
                info = read_feature_info(f)
            except Exception as e:  # noqa: BLE001 -- truncated/locked files are reported, not fatal
                bad.append(f"{sid} ({type(e).__name__})")
                continue
            dims.add(info["dim"])
            inventory.setdefault(sid, {})[f"n_patches_{enc}"] = info["n_patches"]
        n_patches = [inventory[s][f"n_patches_{enc}"] for s in inventory if f"n_patches_{enc}" in inventory[s]]
        dim_note = "" if dims <= {EXPECTED_DIM.get(enc)} else f"  [!] expected dim {EXPECTED_DIM.get(enc)}"
        stats = (f"patches/slide min {min(n_patches)} median {int(pd.Series(n_patches).median())} max {max(n_patches)}"
                 if n_patches else "")
        print(f"  {enc:<12} slides={len(files) - len(bad):5d} dim={sorted(dims)} {stats}{dim_note}")
        for b in bad:
            print(f"      unreadable: {b}")
        missing = sorted(coords - {os.path.splitext(os.path.basename(f))[0] for f in files})
        if missing:
            print(f"      no features yet for {len(missing)} slides with coords: {missing[:10]}{' ...' if len(missing) > 10 else ''}")

    if wsis is not None:
        no_coords = sorted(wsis - coords)
        if no_coords:
            print(f"[!] {len(no_coords)} WSIs have no patch coords (unfinished, error, or no tissue -- see "
                  f"{args.job_dir}/wsi_states/): {no_coords[:10]}{' ...' if len(no_coords) > 10 else ''}")

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        ids = sorted(set(inventory) | (wsis or set()))
        inv = pd.DataFrame([{"slide_id": s, **inventory.get(s, {})} for s in ids])
        inv.to_csv(os.path.join(args.out_dir, "slide_inventory.csv"), index=False)
        tmpl = os.path.join(args.out_dir, "metadata_template.csv")
        if os.path.exists(tmpl):
            print(f"[i] {tmpl} exists, not overwritten")
        else:
            pd.DataFrame({"slide_id": ids, "patient_id": "", "label": ""}).to_csv(tmpl, index=False)
        print(f"[✓] Wrote {args.out_dir}/slide_inventory.csv and metadata_template.csv")


if __name__ == "__main__":
    main()
