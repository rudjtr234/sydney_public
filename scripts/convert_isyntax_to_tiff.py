"""Philips iSyntax/i2syntax 1장 → TIFF.

기존 변환본과 같은 규칙으로 만든다 (새·재스캔 슬라이드를 같은 형식으로 추가하기 위함).
- BigTIFF, 레벨마다 페이지 1개 (2번째부터 reduced image), 256×256 타일, deflate + horizontal predictor
- 레벨 픽셀은 iSyntax의 각 레벨을 그대로 읽음 (다시 축소하지 않음)
- level0 크기를 2^(레벨수-1)의 배수로 맞추려고 오른쪽·아래에만 흰색 패딩 → 원본 좌표 유지
- 해상도 태그 (pixel/cm), ICC 프로파일 (모든 페이지), 첫 페이지 description JSON

사용: python convert_isyntax_to_tiff.py <입력 .isyntax/.i2syntax> <출력 .tiff>
"""
import argparse
import json
import math
import os

import numpy as np
import tifffile
from isyntax import ISyntax

TILE = 256


def tiles(s, level, w, h, pw, ph):
    for ty in range(0, ph, TILE):
        rows = min(TILE, h - ty) if ty < h else 0
        strip = np.full((TILE, pw + TILE, 3), 255, np.uint8)
        if rows > 0:
            strip[:rows, :w] = np.asarray(s.read_region(0, ty, w, rows, level=level))[..., :3]
        for tx in range(0, pw, TILE):
            yield strip[:, tx:tx + TILE]


def convert(src, dst):
    with ISyntax.open(src) as s:
        n = s.level_count
        step = 2 ** (n - 1)
        w0, h0 = s.dimensions
        pw0, ph0 = math.ceil(w0 / step) * step, math.ceil(h0 / step) * step
        icc = s.read_icc_profile()
        desc = json.dumps({
            "source_file": os.path.basename(src),
            "source_format": "Philips iSyntax",
            "converter": "pyisyntax + tifffile, lossless deflate, levels from iSyntax",
            "mpp_x": s.mpp_x,
            "mpp_y": s.mpp_y,
            "source_level_dimensions": [list(d) for d in s.level_dimensions],
            "level_downsamples": [2 ** i for i in range(n)],
            "padded_level0": [pw0, ph0],
            "padding": "white, right/bottom only; original pixels keep their coordinates",
        })
        icc = bytes(icc) if icc is not None and len(icc) else None
        tmp = dst + ".partial"
        with tifffile.TiffWriter(tmp, bigtiff=True) as tw:
            for lv in range(n):
                w, h = s.level_dimensions[lv]
                pw, ph = pw0 // 2 ** lv, ph0 // 2 ** lv
                tw.write(
                    tiles(s, lv, w, h, pw, ph), shape=(ph, pw, 3), dtype=np.uint8,
                    tile=(TILE, TILE), photometric="rgb", planarconfig="contig",
                    compression="adobe_deflate", predictor=True,
                    resolution=(1e4 / (s.mpp_x * 2 ** lv), 1e4 / (s.mpp_y * 2 ** lv)),
                    resolutionunit="CENTIMETER", subfiletype=0 if lv == 0 else 1,
                    description=desc if lv == 0 else None, software="convert_isyntax_to_tiff.py",
                    iccprofile=icc, metadata=None,
                )
                print(f"level {lv}: {w}x{h} -> {pw}x{ph}", flush=True)
        os.replace(tmp, dst)
    print(f"done: {dst} ({os.path.getsize(dst) / 1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    a = ap.parse_args()
    convert(a.src, a.dst)
