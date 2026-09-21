"""영상 → memmap 캐시 + 유효 픽셀 마스크 + 동서 분할 경계 계산.

두 가지 입력 형태를 지원한다.

  --source feats38  (현재 받은 형태)
      {region}_feats38.npy   (38, H, W) float16 — RF가 쓰던 38피처 그대로
      복사하지 않고 원본을 그대로 memmap 한다. meta에 경로만 적는다.

  --source s2       (20밴드 tif가 올 경우)
      {region}_s2_20band.tif  대전 float32 / 홍천·순천 int16(x10000)
      → float32 (20, H, W) 로 펴서 cache_dir 에 저장

라벨은 공통:
      {region}_label.tif    0 배경 / 1 침엽 / 2 활엽 / 3 혼효 / 255 nodata
      {region}_year.tif     갱신년도
      {region}_species.tif  수종코드 (선택)

사용:
    python prepare_data.py --data-dir "C:/.../data" --cache-dir ./cache \
        --source feats38 --regions daejeon
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLASSES, N_BANDS, REGIONS, YEAR_MIN, to_reflectance


def _read1(path):
    import rasterio
    with rasterio.open(path) as src:
        return src.read(1)


def prepare_region(region, data_dir, cache_dir, source="feats38", force=False):
    import rasterio

    meta_path = os.path.join(cache_dir, f"{region}_meta.json")
    if os.path.exists(meta_path) and not force:
        print(f"[{region}] 캐시 존재 — 건너뜀 (--force 로 재생성)")
        return json.load(open(meta_path, encoding="utf-8"))

    os.makedirs(cache_dir, exist_ok=True)

    # ── 영상 ──────────────────────────────────────────────────────────────
    if source == "feats38":
        src_path = os.path.join(data_dir, f"{region}_feats38.npy")
        if not os.path.exists(src_path):
            raise FileNotFoundError(src_path)
        arr = np.load(src_path, mmap_mode="r")
        if arr.ndim != 3:
            raise ValueError(f"{region}: shape {arr.shape} — (C,H,W) 를 기대")
        B, H, W = arr.shape
        x_path = os.path.abspath(src_path)     # 복사하지 않는다
        print(f"[{region}] feats38 {B}피처 {H}x{W} {arr.dtype} (원본 직접 memmap)")
    elif source == "s2":
        src_path = os.path.join(data_dir, f"{region}_s2_20band.tif")
        if not os.path.exists(src_path):
            raise FileNotFoundError(src_path)
        x_path = os.path.join(cache_dir, f"{region}_x.npy")
        with rasterio.open(src_path) as src:
            H, W, B = src.height, src.width, src.count
            if B != N_BANDS:
                raise ValueError(f"{region}: 밴드 {B}개 (기대 {N_BANDS})")
            print(f"[{region}] s2 {H}x{W} x {B}밴드 {src.dtypes[0]} → float32 캐시")
            xm = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.float32,
                                           shape=(B, H, W))
            for b in range(B):      # 밴드별로 — 통째로 올리면 홍천에서 RAM을 먹는다
                xm[b] = to_reflectance(src.read(b + 1), region)
            xm.flush()
            del xm
        x_path = os.path.abspath(x_path)
    else:
        raise ValueError(source)

    # ── 라벨 ──────────────────────────────────────────────────────────────
    label = _read1(os.path.join(data_dir, f"{region}_label.tif"))
    year = _read1(os.path.join(data_dir, f"{region}_year.tif"))
    if label.shape != (H, W):
        raise ValueError(f"{region}: 라벨 {label.shape} != 영상 {(H, W)}")

    # 유효 픽셀: 라벨이 침엽/활엽/혼효 이고 갱신년도 2024+
    # 배경(0)·nodata(255)·죽림·미갱신은 학습·평가 모두에서 제외 (RF와 동일)
    valid = np.isin(label, CLASSES) & (year >= YEAR_MIN)
    y = np.where(valid, label, 0).astype(np.uint8)
    np.save(os.path.join(cache_dir, f"{region}_y.npy"), y)

    sp_src = os.path.join(data_dir, f"{region}_species.tif")
    if os.path.exists(sp_src):
        np.save(os.path.join(cache_dir, f"{region}_sp.npy"),
                _read1(sp_src).astype(np.int16))

    # ── 동서 분할: 유효 픽셀의 중앙 열. 서=학습, 동=평가 ──────────────────
    # (문서 10절 항목3 '평가셋 공간 분리 재설계'. 대각/비대각 교락 해소용)
    col_counts = valid.sum(axis=0).astype(np.int64)
    cum = np.cumsum(col_counts)
    split_col = int(np.searchsorted(cum, cum[-1] // 2))

    n_valid = int(valid.sum())
    meta = dict(
        region=region, source=source, x_path=x_path,
        H=int(H), W=int(W), bands=int(B),
        n_valid=n_valid, split_col=split_col,
        west_valid=int(valid[:, :split_col].sum()),
        east_valid=int(valid[:, split_col:].sum()),
        class_frac={int(c): float((y == c).sum() / max(n_valid, 1)) for c in CLASSES},
    )
    json.dump(meta, open(meta_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"[{region}] 유효 {n_valid:,}px  분할열 {split_col} "
          f"(서 {meta['west_valid']:,} / 동 {meta['east_valid']:,})")
    print(f"[{region}] 클래스 비율 "
          + "  ".join(f"{c}:{v*100:.1f}%" for c, v in meta["class_frac"].items()))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--cache-dir", default="./cache")
    ap.add_argument("--source", choices=["feats38", "s2"], default="feats38")
    ap.add_argument("--regions", nargs="*", default=list(REGIONS))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for r in a.regions:
        prepare_region(r, a.data_dir, a.cache_dir, a.source, a.force)


if __name__ == "__main__":
    main()
