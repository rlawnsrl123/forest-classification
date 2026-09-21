"""합성 데이터 생성 — 실제 tif가 오기 전에 파이프라인 전체를 검증하기 위한 것.

실제 데이터의 성질을 의도적으로 복제한다(수치는 문서 §3-4, §2의 표에서 가져옴):
  · 지역별 낙엽기 NDVI 오프셋   홍천 활0.369/침0.538, 대전 0.440/0.655, 순천 0.515/0.742
  · 클래스 비율               대전 32.8/46.1/17.1, 홍천 35.7/51.5/11.4, 순천 47.6/38.1/9.9
  · 낙엽송(sp 13) 노출도       홍천 33% / 대전 8.6% / 순천 0%  (낙엽기 NDVI가 활엽 수준)
  · 저장 형식 함정            대전만 float32, 나머지 int16(x10000)
  · 갱신년도 분포             약 1/3이 2024 미만 → 필터가 실제로 걸리는지 확인용
  · 공간 자기상관             클래스가 임분 단위로 뭉쳐 있어야 U-Net이 의미를 가진다

★ 이 데이터로 나온 숫자는 논문에 절대 쓰지 말 것. 파이프라인 검증 전용이다.

사용:
    python make_synthetic.py --out ./synth
"""
import argparse
import os

import numpy as np

# 낙엽기 NDVI 중앙값 (문서 §3-4)
W_NDVI = {
    "daejeon":   {1: 0.655, 2: 0.440},
    "hongcheon": {1: 0.538, 2: 0.369},
    "suncheon":  {1: 0.742, 2: 0.515},
}
# 여름은 세 클래스가 0.852~0.885로 거의 붙어 있다 (분리폭 0.033)
S_NDVI = {1: 0.852, 2: 0.885, 3: 0.873}

CLASS_FRAC = {
    "daejeon":   {1: 0.328, 2: 0.461, 3: 0.171},
    "hongcheon": {1: 0.357, 2: 0.515, 3: 0.114},
    "suncheon":  {1: 0.476, 2: 0.381, 3: 0.099},
}
LARCH_FRAC = {"daejeon": 0.086, "hongcheon": 0.33, "suncheon": 0.0}
SHAPE = {"daejeon": (1024, 1280), "hongcheon": (1536, 1792),
         "suncheon": (1280, 1536)}
IS_FLOAT = {"daejeon": True, "hongcheon": False, "suncheon": False}

RED_S, NIR_S, RED_W, NIR_W = 3, 7, 13, 17   # 밴드 인덱스(0-based)


def smooth_field(shape, scale, rng):
    """저해상도 난수를 확대해 만든 공간 자기상관 필드 (임분 크기 모사)."""
    h, w = shape
    small = rng.standard_normal((max(h // scale, 2), max(w // scale, 2)))
    ys = np.linspace(0, small.shape[0] - 1, h)
    xs = np.linspace(0, small.shape[1] - 1, w)
    y0 = np.clip(ys.astype(int), 0, small.shape[0] - 2)
    x0 = np.clip(xs.astype(int), 0, small.shape[1] - 2)
    fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]
    a = small[y0][:, x0]
    b = small[y0 + 1][:, x0]
    c = small[y0][:, x0 + 1]
    d = small[y0 + 1][:, x0 + 1]
    return (a * (1 - fy) * (1 - fx) + b * fy * (1 - fx)
            + c * (1 - fy) * fx + d * fy * fx)


def make_region(region, out_dir, seed=0):
    import rasterio
    from rasterio.transform import from_origin

    rng = np.random.default_rng(seed)
    H, W = SHAPE[region]
    frac = CLASS_FRAC[region]

    # ── 라벨: 공간 자기상관 필드를 분위수로 잘라 클래스 비율을 맞춘다 ──────
    f = smooth_field((H, W), 40, rng)
    q = np.quantile(f, [frac[1], frac[1] + frac[2], frac[1] + frac[2] + frac[3]])
    label = np.zeros((H, W), np.uint8)
    label[f <= q[0]] = 1
    label[(f > q[0]) & (f <= q[1])] = 2
    label[(f > q[1]) & (f <= q[2])] = 3          # 나머지는 0 = 비산림

    # ── 수종: 침엽 안에서 낙엽송(13)을 임분 단위로 배치 ────────────────────
    sp = np.zeros((H, W), np.int16)
    conif = label == 1
    sp[conif] = 11                                # 소나무
    sp[label == 2] = 32                           # 신갈
    if LARCH_FRAC[region] > 0:
        g = smooth_field((H, W), 25, rng)
        thr = np.quantile(g[conif], LARCH_FRAC[region])
        sp[conif & (g <= thr)] = 13               # 낙엽송

    # ── 갱신년도: 약 1/3이 2024 미만 (덩어리 단위) ────────────────────────
    ygrid = smooth_field((H, W), 60, rng)
    year = np.where(ygrid <= np.quantile(ygrid, 0.33), 2019, 2025).astype(np.int16)

    # ── 낙엽기 NDVI 목표값 ───────────────────────────────────────────────
    off = W_NDVI[region]
    wn = np.zeros((H, W), np.float32)
    wn[label == 1] = off[1]
    wn[label == 2] = off[2]
    wn[label == 3] = (off[1] + off[2]) / 2        # 혼효는 중간
    wn[label == 0] = 0.15
    # 낙엽송은 침엽 라벨이지만 낙엽기 NDVI가 활엽 수준 — 본 연구의 핵심 예외
    wn[sp == 13] = off[2]
    # 픽셀 잡음 + 임분 단위 변동
    wn += 0.04 * smooth_field((H, W), 8, rng) + 0.02 * rng.standard_normal((H, W))
    wn = np.clip(wn, -0.2, 0.95)

    sn = np.zeros((H, W), np.float32)
    for c, v in S_NDVI.items():
        sn[label == c] = v
    sn[label == 0] = 0.25
    sn += 0.03 * smooth_field((H, W), 8, rng) + 0.02 * rng.standard_normal((H, W))
    sn = np.clip(sn, -0.2, 0.95)

    # ── 20밴드 합성: NDVI가 목표값이 되도록 red/nir를 역산 ────────────────
    x = np.zeros((20, H, W), np.float32)
    for b in range(20):
        base = 0.05 + 0.02 * b / 20.0
        x[b] = base + 0.01 * smooth_field((H, W), 12, rng) \
            + 0.005 * rng.standard_normal((H, W))
    red_s = 0.06 + 0.01 * rng.standard_normal((H, W)).astype(np.float32)
    red_w = 0.07 + 0.01 * rng.standard_normal((H, W)).astype(np.float32)
    red_s = np.clip(red_s, 0.01, None)
    red_w = np.clip(red_w, 0.01, None)
    x[RED_S], x[NIR_S] = red_s, red_s * (1 + sn) / np.maximum(1 - sn, 1e-3)
    x[RED_W], x[NIR_W] = red_w, red_w * (1 + wn) / np.maximum(1 - wn, 1e-3)
    x = np.clip(x, 0, 1.2)

    # ── 저장 (형식 함정 재현) ────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    if IS_FLOAT[region]:
        arr, dt = x.astype(np.float32), "float32"
    else:
        arr, dt = (x * 10000).astype(np.int16), "int16"
    tf = from_origin(0, 0, 10, 10)
    prof = dict(driver="GTiff", height=H, width=W, crs="EPSG:5179",
                transform=tf, tiled=True, blockxsize=256, blockysize=256,
                compress="deflate")
    with rasterio.open(os.path.join(out_dir, region + "_s2_20band.tif"), "w",
                       count=20, dtype=dt, **prof) as d:
        d.write(arr)
    for name, a2, dt2 in (("label", label, "uint8"), ("year", year, "int16"),
                          ("species", sp, "int16")):
        with rasterio.open(os.path.join(out_dir, region + "_" + name + ".tif"),
                           "w", count=1, dtype=dt2, **prof) as d:
            d.write(a2.astype(dt2), 1)

    v = (np.isin(label, (1, 2, 3))) & (year >= 2024)
    print("[%s] %dx%d  유효 %d px  낙엽송 %.1f%%  dtype %s"
          % (region, H, W, v.sum(),
             100.0 * (sp[conif] == 13).mean() if conif.any() else 0.0, dt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./synth")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for i, r in enumerate(("daejeon", "hongcheon", "suncheon")):
        make_region(r, a.out, a.seed + i)
    print("\n합성 데이터 생성 완료: " + a.out)
    print("주의: 파이프라인 검증 전용. 이 숫자는 결과가 아니다.")


if __name__ == "__main__":
    main()
