"""합성 래스터 → {region}_samples.npz — rf_reproduce.py 스모크 테스트용.

준기 님 npz의 피처 배열(38개)을 그대로 재현한다. 순서가 중요하다:
    0~19   원 밴드 (여름 10 + 낙엽기 10)
    20~25  지수    s_ndvi, w_ndvi, s_ndmi, w_ndmi, dNDVI, dNDMI
    21     = W_NDVI  ← 문서에서 낙엽기 NDVI 인덱스로 검증된 값
    26~37  공간 컨텍스트 (s_ndvi/w_ndvi/d_ndvi × 윈도우 5,15 × 평균,표준편차)

키: X(float32 N x 38), y(uint8), tile(int64), sp(int16)
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLASSES, REGIONS, TILE, tile_id

RED_S, NIR_S, SWIR_S, RED_W, NIR_W, SWIR_W = 3, 7, 9, 13, 17, 19


def box_stats(a, k):
    """윈도우 k의 이동 평균·표준편차 (적분영상)."""
    p = k // 2
    ap = np.pad(a, p, mode="edge").astype(np.float64)
    i1 = np.zeros((ap.shape[0] + 1, ap.shape[1] + 1))
    i2 = np.zeros_like(i1)
    np.cumsum(np.cumsum(ap, 0), 1, out=i1[1:, 1:])
    np.cumsum(np.cumsum(ap ** 2, 0), 1, out=i2[1:, 1:])
    H, W = a.shape

    def win(ii):
        return (ii[k:k + H, k:k + W] - ii[0:H, k:k + W]
                - ii[k:k + H, 0:W] + ii[0:H, 0:W]) / (k * k)

    m = win(i1)
    v = np.maximum(win(i2) - m ** 2, 0)
    return m.astype(np.float32), np.sqrt(v).astype(np.float32)


def build(region, cache_dir, out_dir, n=600_000, seed=0):
    x = np.load(os.path.join(cache_dir, region + "_x.npy"), mmap_mode="r")
    y = np.load(os.path.join(cache_dir, region + "_y.npy"))
    sp_path = os.path.join(cache_dir, region + "_sp.npy")
    sp = np.load(sp_path) if os.path.exists(sp_path) else np.zeros_like(y, np.int16)

    def ndx(a, b):
        return (a - b) / np.maximum(a + b, 1e-6)

    bands = np.asarray(x, dtype=np.float32)
    s_ndvi = ndx(bands[NIR_S], bands[RED_S])
    w_ndvi = ndx(bands[NIR_W], bands[RED_W])
    s_ndmi = ndx(bands[NIR_S], bands[SWIR_S])
    w_ndmi = ndx(bands[NIR_W], bands[SWIR_W])
    d_ndvi = s_ndvi - w_ndvi
    d_ndmi = s_ndmi - w_ndmi

    ctx = []
    for a in (s_ndvi, w_ndvi, d_ndvi):
        for k in (5, 15):
            m, s = box_stats(a, k)
            ctx += [m, s]

    feats = list(bands) + [s_ndvi, w_ndvi, s_ndmi, w_ndmi, d_ndvi, d_ndmi] + ctx
    assert len(feats) == 38, len(feats)

    rows, cols = np.nonzero(np.isin(y, CLASSES))
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(rows), min(n, len(rows)), replace=False)
    r, c = rows[pick], cols[pick]

    X = np.stack([f[r, c] for f in feats], axis=1).astype(np.float32)
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, region + "_samples.npz")
    np.savez_compressed(p, X=X, y=y[r, c].astype(np.uint8),
                        tile=tile_id(r.astype(np.int64), c.astype(np.int64)),
                        sp=sp[r, c].astype(np.int16))
    print("[%s] %d px x 38피처  타일 %d개  → %s"
          % (region, len(X), len(np.unique(tile_id(r, c))), p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="./cache_synth")
    ap.add_argument("--out-dir", default="./npz_synth")
    ap.add_argument("--n", type=int, default=600_000)
    a = ap.parse_args()
    for i, r in enumerate(REGIONS):
        build(r, a.cache_dir, a.out_dir, a.n, seed=i)


if __name__ == "__main__":
    main()
