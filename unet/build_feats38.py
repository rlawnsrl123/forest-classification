"""20밴드 tif → 38피처 npy. 생성 규칙은 verify_feats.py 로 대전에서 역추적해 확정했다.

  0~19  원 밴드 (여름 B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12 / 낙엽기 동일 순서)
  20    s_ndvi    = (여름 B8   − 여름 B4)   / 합
  21    w_ndvi    = (낙엽기 B8 − 낙엽기 B4) / 합      ← 본 연구의 핵심 축
  22    s_ndmi    = (여름 B8   − 여름 B11)  / 합
  23    w_ndmi    = (낙엽기 B8 − 낙엽기 B11)/ 합
  24    dNDVI     = s_ndvi − w_ndvi
  25    dNDMI     = s_ndmi − w_ndmi
  26~37 공간 컨텍스트: (s_ndvi, w_ndvi, dNDVI) × 윈도우 (5,15) × (평균, 표준편차)
        순서는 소스 바깥루프 → 윈도우 → 평균/표준편차

검증: 이 코드로 만든 대전 결과와 준기 님 제공본의 최대 절대차 0.00025 (float16 해상도).

세 지역을 같은 코드로 만들어야 9칸 비교가 성립하므로 대전도 다시 만든다.

사용:
    python build_feats38.py --s2-dir ../data/s2 --out-dir ../data/feats
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (NIR_S, NIR_W, RED_S, RED_W, REGIONS, SWIR_S, SWIR_W,
                    to_reflectance)


def box_mean_std(a, k):
    """윈도우 k 의 이동 평균·표준편차(모집단). 가장자리는 reflect.

    ★ 적분영상은 NaN이 하나라도 있으면 그 뒤 전체로 번진다.
      들어오기 전에 유한값으로 정리해 두어야 한다.
    """
    p = k // 2
    ap = np.pad(np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64),
                p, mode="reflect")
    i1 = np.zeros((ap.shape[0] + 1, ap.shape[1] + 1))
    np.cumsum(np.cumsum(ap, 0), 1, out=i1[1:, 1:])
    i2 = np.zeros_like(i1)
    np.cumsum(np.cumsum(ap * ap, 0), 1, out=i2[1:, 1:])
    del ap
    H, W = a.shape

    def win(ii):
        return (ii[k:k + H, k:k + W] - ii[0:H, k:k + W]
                - ii[k:k + H, 0:W] + ii[0:H, 0:W]) / float(k * k)

    m = win(i1)
    del i1
    v = np.maximum(win(i2) - m * m, 0.0)
    del i2
    return m.astype(np.float32), np.sqrt(v).astype(np.float32)


def nd(a, b):
    return ((a - b) / np.maximum(a + b, 1e-6)).astype(np.float32)


def build(region, s2_dir, out_dir, force=False):
    import rasterio

    src_path = os.path.join(s2_dir, f"{region}_s2_20band.tif")
    out_path = os.path.join(out_dir, f"{region}_feats38.npy")
    if os.path.exists(out_path) and not force:
        print(f"[{region}] 이미 존재 — 건너뜀")
        return out_path
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    with rasterio.open(src_path) as src:
        H, W, B = src.height, src.width, src.count
        print(f"[{region}] {H}x{W} x{B} {src.dtypes[0]} → 38피처 float16", flush=True)
        out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16,
                                        shape=(38, H, W))
        keep = {}
        for b in range(B):
            arr = to_reflectance(src.read(b + 1), region)
            out[b] = arr.astype(np.float16)
            if b in (RED_S, NIR_S, SWIR_S, RED_W, NIR_W, SWIR_W):
                keep[b] = arr
            del arr

    s_ndvi = nd(keep[NIR_S], keep[RED_S])
    w_ndvi = nd(keep[NIR_W], keep[RED_W])
    s_ndmi = nd(keep[NIR_S], keep[SWIR_S])
    w_ndmi = nd(keep[NIR_W], keep[SWIR_W])
    del keep
    d_ndvi = (s_ndvi - w_ndvi).astype(np.float32)
    d_ndmi = (s_ndmi - w_ndmi).astype(np.float32)
    for i, v in enumerate((s_ndvi, w_ndvi, s_ndmi, w_ndmi, d_ndvi, d_ndmi), start=20):
        out[i] = v.astype(np.float16)
    del s_ndmi, w_ndmi, d_ndmi
    print(f"[{region}]   지수 6개 완료 ({time.time()-t0:.0f}초)", flush=True)

    i = 26
    for name, base in (("s_ndvi", s_ndvi), ("w_ndvi", w_ndvi), ("dNDVI", d_ndvi)):
        for k in (5, 15):
            m, s = box_mean_std(base, k)
            out[i] = m.astype(np.float16); i += 1
            out[i] = s.astype(np.float16); i += 1
            del m, s
            print(f"[{region}]   {name} w{k} 완료 ({time.time()-t0:.0f}초)", flush=True)
    assert i == 38, i
    out.flush()
    del out
    print(f"[{region}] 저장 {out_path}  ({time.time()-t0:.0f}초, "
          f"{os.path.getsize(out_path)/1e9:.2f} GB)", flush=True)
    return out_path


def check_against(built_path, ref_path, name="대전"):
    """제공본과 대조 — 생성 규칙이 맞는지 최종 확인."""
    a = np.load(built_path, mmap_mode="r")
    b = np.load(ref_path, mmap_mode="r")
    if a.shape != b.shape:
        print(f"[{name}] shape 불일치 {a.shape} vs {b.shape}")
        return
    worst, worst_i = 0.0, -1
    for i in range(a.shape[0]):
        x = np.asarray(a[i], dtype=np.float32)
        y = np.asarray(b[i], dtype=np.float32)
        m = np.isfinite(x) & np.isfinite(y)
        d = float(np.abs(x[m] - y[m]).max()) if m.any() else 0.0
        if d > worst:
            worst, worst_i = d, i
    print(f"[{name}] 제공본 대비 최대 절대차 {worst:.5f} (피처 {worst_i}) "
          f"→ {'일치' if worst < 0.01 else '불일치'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--regions", nargs="*", default=list(REGIONS))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check-against", default=None,
                    help="대전 제공본 feats38 경로 — 생성 규칙 최종 대조")
    a = ap.parse_args()
    for r in a.regions:
        p = build(r, a.s2_dir, a.out_dir, a.force)
        if r == "daejeon" and a.check_against:
            check_against(p, a.check_against)


if __name__ == "__main__":
    main()
