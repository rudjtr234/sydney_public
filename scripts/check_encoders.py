# scripts/check_encoders.py
# 대량 추출 전에 4개 인코더가 실제로 로드되는지(HF gated 접근, 의존성) + 출력 차원 확인
#
#   python scripts/check_encoders.py --gpu 0
#   ... --encoders virchow hibou_l

import argparse
import traceback

import torch
from trident.patch_encoder_models import encoder_factory

DEFAULT_ENCODERS = ["hibou_l", "hoptimus0", "virchow", "midnight12k"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoders", nargs="+", default=DEFAULT_ENCODERS)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")

    ok = True
    for name in args.encoders:
        try:
            enc = encoder_factory(name).to(device).eval()
            x = torch.rand(2, 3, 224, 224, device=device)
            with torch.inference_mode(), torch.autocast(device.type, dtype=enc.precision,
                                                         enabled=enc.precision != torch.float32):
                dim = enc(x).shape[-1]
            print(f"[OK]   {name:<12} dim={dim} precision={enc.precision}")
            del enc
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001 -- report every encoder, then fail at the end
            ok = False
            print(f"[FAIL] {name:<12} {type(e).__name__}: {e}")
            traceback.print_exc()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
