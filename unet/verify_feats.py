"""대전으로 38피처 생성 규칙을 역추적한다.

대전만 20밴드 tif와 준기 님이 만든 feats38.npy 를 **둘 다** 갖고 있다.
그래서 후보 피처를 계산해 제공본과 맞춰보면, 문서에 순서가 안 적힌
공간 컨텍스트 12개(26~37)의 배치와 계산 방식을 확정할 수 있다.

확정되면 같은 규칙으로 홍천·순천 feats38을 만들어 9칸 표에 들어간다.

사용:
    python verify_feats.py --s2 ../data/s2/daejeon_s2_20band.tif \
        --feats "C:/Users/aigre/OneDrive/Desktop/data/daejeon_feats38.npy"
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import NIR_S, NIR_W, RED_S, RED_W, SWIR_S, SWIR_W


def box_mean_std(a, k, mode="reflect"):
    """윈도우 k 의 이동 평균·표준편차(모집단). 적분영상으로 O(1)/픽셀."""
    p = k // 2
    ap = np.pad(a.astype(np.float64), p, mode=mode)
    i1 = np.zeros((ap.shape[0] + 1, ap.shape[1] + 1))
    i2 = np.zeros_like(i1)
    np.cumsum(np.cumsum(ap, 0), 1, out=i1[1:, 1:])
    np.cumsum(np.cumsum(ap * ap, 0), 1, out=i2[1:, 1:])
    H, W = a.shape

    def win(ii):
        return (ii[k:k + H, k:k + W] - ii[0:H, k:k + W]
                - ii[k:k + H, 0:W] + ii[0:H, 0:W]) / float(k * k)

    m = win(i1)
    v = np.maximum(win(i2) - m * m, 0.0)
    return m.astype(np.float32), np.sqrt(v).astype(np.float32)


def nd(a, b):
    return ((a - b) / np.maximum(a + b, 1e-6)).astype(np.float32)


def candidates(bands):
    """26~37 자리에 올 수 있는 후보들을 이름과 함께 만든다."""
    s_ndvi = nd(bands[NIR_S], bands[RED_S])
    w_ndvi = nd(bands[NIR_W], bands[RED_W])
    s_ndmi = nd(bands[NIR_S], bands[SWIR_S])
    w_ndmi = nd(bands[NIR_W], bands[SWIR_W])
    base = {
        "s_ndvi": s_ndvi, "w_ndvi": w_ndvi,
        "s_ndmi": s_ndmi, "w_ndmi": w_ndmi,
        "dNDVI(s-w)": (s_ndvi - w_ndvi).astype(np.float32),
        "dNDVI(w-s)": (w_ndvi - s_ndvi).astype(np.float32),
        "dNDMI(s-w)": (s_ndmi - w_ndmi).astype(np.float32),
        "dNDMI(w-s)": (w_ndmi - s_ndmi).astype(np.float32),
    }
    out = dict(base)
    for src in ("s_ndvi", "w_ndvi", "dNDVI(s-w)", "dNDVI(w-s)"):
        for k in (5, 15):
            m, s = box_mean_std(base[src], k)
            out[f"{src}_w{k}_mean"] = m
            out[f"{src}_w{k}_std"] = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2", required=True)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--region", default="daejeon")
    ap.add_argument("--row", type=int, default=1200)
    ap.add_argument("--col", type=int, default=1000)
    ap.add_argument("--size", type=int, default=700)
    ap.add_argument("--margin", type=int, default=20,
                    help="크롭 가장자리 제외 폭(윈도우 15보다 커야 한다)")
    a = ap.parse_args()

    import rasterio
    from common import to_reflectance

    r0, c0, n, mg = a.row, a.col, a.size, a.margin
    with rasterio.open(a.s2) as src:
        print("s2 %dx%d x %d밴드 %s" % (src.height, src.width, src.count,
                                       src.dtypes[0]))
        win = rasterio.windows.Window(c0, r0, n, n)
        bands = np.stack([to_reflectance(src.read(b + 1, window=win), a.region)
                          for b in range(src.count)])
    prov = np.load(a.feats, mmap_mode="r")[:, r0:r0 + n, c0:c0 + n]
    prov = np.asarray(prov, dtype=np.float32)
    print("제공 feats38 크롭 %s" % (prov.shape,))

    cand = candidates(bands)
    inner = (slice(mg, n - mg), slice(mg, n - mg))
    ok = np.isfinite(prov[:, inner[0], inner[1]]).all(axis=0)
    for v in cand.values():
        ok &= np.isfinite(v[inner])
    print("비교 픽셀 %d개\n" % ok.sum())

    # 원 밴드 20개 먼저
    print("== 0~19 원 밴드 ==")
    worst = 0.0
    for b in range(20):
        d = np.abs(prov[b][inner][ok] - bands[b][inner][ok]).max()
        worst = max(worst, d)
    print("  최대 절대차 %.5f  %s" % (worst, "일치" if worst < 0.01 else "불일치"))

    print("\n== 20~37 파생 피처: 제공본과 가장 가까운 후보 ==")
    mapping = {}
    for i in range(20, 38):
        p = prov[i][inner][ok]
        best = None
        for name, v in cand.items():
            d = float(np.abs(p - v[inner][ok]).max())
            if best is None or d < best[1]:
                best = (name, d)
        mapping[i] = best
        flag = "OK" if best[1] < 0.01 else ("~" if best[1] < 0.05 else "??")
        print("  [%2d]  %-20s 최대차 %.5f  %s" % (i, best[0], best[1], flag))

    bad = [i for i, (_, d) in mapping.items() if d >= 0.01]
    print("\n확정 실패 인덱스: %s" % (bad if bad else "없음 — 전부 일치"))
    if not bad:
        print("\nFEAT_ORDER = [")
        for i in range(20, 38):
            print("    %-24s # [%d]" % ('"%s",' % mapping[i][0], i))
        print("]")


if __name__ == "__main__":
    main()
