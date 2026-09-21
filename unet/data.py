"""패치 샘플러 — memmap에서 128x128 패치를 뽑는다.

핵심 제약(RF와 동일하게 맞춰야 하는 것):
  · 학습/평가는 공간적으로 분리한다. 여기서는 동서 분할(서=학습, 동=평가).
    패치가 분할선을 넘지 않도록 시작 열을 제한한다 → 경계 누수 없음.
  · 갱신년도 2024+ 이고 라벨이 1/2/3 인 픽셀만 손실·평가에 들어간다.
    나머지는 IGNORE 로 마스킹 (U-Net은 패치 전체를 보지만 학습 신호는 유효 픽셀만).
"""
import json, os
import numpy as np
import torch
from torch.utils.data import Dataset

from common import CLASSES, IGNORE


class RegionCache:
    """한 지역의 memmap 묶음. 프로세스마다 lazy open.

    ★ __getstate__ 로 memmap 핸들을 떼고 보낸다.
      Windows DataLoader는 spawn 방식이라 Dataset이 통째로 pickle 되는데,
      np.memmap은 pickle 할 때 실제 데이터를 복사한다(홍천 3GB x 워커 수).
      핸들을 떼면 각 워커가 자기 프로세스에서 다시 mmap 한다.
    """

    def __init__(self, region, cache_dir):
        self.region, self.cache_dir = region, cache_dir
        self.meta = json.load(
            open(os.path.join(cache_dir, f"{region}_meta.json"), encoding="utf-8"))
        self._x = self._y = None

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_x"] = d["_y"] = None
        return d

    @property
    def x(self):
        if self._x is None:
            # feats38 소스는 원본을 그대로 memmap 한다(750MB 복사 회피)
            p = self.meta.get("x_path") or os.path.join(
                self.cache_dir, f"{self.region}_x.npy")
            self._x = np.load(p, mmap_mode="r")
        return self._x

    @property
    def y(self):
        if self._y is None:
            self._y = np.load(os.path.join(self.cache_dir, f"{self.region}_y.npy"),
                              mmap_mode="r")
        return self._y

    @property
    def split_col(self):
        return self.meta["split_col"]


def normalize(x, mean3, std3):
    """표준화 + NaN/Inf 제거. 학습·추론·AdaBN 전부 이 함수를 거쳐야 한다.

    ★ 대전 feats38 에는 NaN이 27,885픽셀(0.28%) 있다. 유효 라벨 픽셀에는 없지만
      패치에는 딸려 들어온다. NaN 하나가 컨볼루션으로 퍼지고 BatchNorm 통계를
      오염시켜 배치 전체를 NaN으로 만들고, 그 그래디언트가 가중치를 파괴한다.
      ignore_index는 라벨만 무시할 뿐 입력 NaN은 막지 못한다.
      표준화 뒤에 0으로 채우므로 결과적으로 '그 밴드의 평균값'으로 대치된다.
    """
    x = (x - mean3) / std3
    np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def patch_origins(valid, patch, stride, col_lo, col_hi, min_valid_frac):
    """유효 픽셀이 충분한 패치 좌상단 좌표 목록.

    적분영상으로 패치별 유효 픽셀 수를 O(1)에 센다.
    col_lo:col_hi 안에 패치가 완전히 들어가야 한다(분할선 넘지 않음).
    """
    ii = np.zeros((valid.shape[0] + 1, valid.shape[1] + 1), dtype=np.int32)
    np.cumsum(np.cumsum(valid.astype(np.int32), axis=0), axis=1, out=ii[1:, 1:])

    rows = np.arange(0, valid.shape[0] - patch + 1, stride)
    cols = np.arange(col_lo, col_hi - patch + 1, stride)
    if len(rows) == 0 or len(cols) == 0:
        return np.empty((0, 2), dtype=np.int32)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    cnt = (ii[rr + patch, cc + patch] - ii[rr, cc + patch]
           - ii[rr + patch, cc] + ii[rr, cc])
    keep = cnt >= min_valid_frac * patch * patch
    return np.stack([rr[keep], cc[keep]], axis=1).astype(np.int32)


class PatchDataset(Dataset):
    """학습용. 좌표 목록에서 뽑고 flip/rot90 증강."""

    def __init__(self, cache, origins, patch, mean, std, augment=True):
        self.cache, self.origins, self.patch = cache, origins, patch
        self.mean = mean.reshape(-1, 1, 1).astype(np.float32)
        self.std = std.reshape(-1, 1, 1).astype(np.float32)
        self.augment = augment
        self._rng = None

    def __len__(self):
        return len(self.origins)

    @property
    def rng(self):
        """워커별 독립 난수. torch는 numpy 전역 RNG를 워커마다 갈아주지 않아서,
        np.random 을 그대로 쓰면 모든 워커가 똑같은 증강을 낸다."""
        if self._rng is None:
            self._rng = np.random.default_rng(torch.initial_seed() % (2 ** 32))
        return self._rng

    def __getitem__(self, i):
        r, c = self.origins[i]
        p = self.patch
        x = np.asarray(self.cache.x[:, r:r + p, c:c + p], dtype=np.float32)
        y = np.asarray(self.cache.y[r:r + p, c:c + p])

        x = normalize(x, self.mean, self.std)
        # 라벨 1/2/3 → 0/1/2, 그 외(무효)는 IGNORE
        t = np.full(y.shape, IGNORE, dtype=np.int64)
        for k, cl in enumerate(CLASSES):
            t[y == cl] = k

        if self.augment:
            rng = self.rng
            if rng.random() < 0.5:
                x, t = x[:, :, ::-1], t[:, ::-1]
            if rng.random() < 0.5:
                x, t = x[:, ::-1, :], t[::-1, :]
            k = int(rng.integers(4))
            if k:
                x, t = np.rot90(x, k, (1, 2)), np.rot90(t, k, (0, 1))
        return (torch.from_numpy(np.ascontiguousarray(x)),
                torch.from_numpy(np.ascontiguousarray(t)))


def band_stats(cache, patch, col_lo, col_hi, n_sample=200, seed=0):
    """학습 지역의 학습 영역에서만 밴드별 평균·표준편차를 잰다.

    ★ 평가 지역 통계를 쓰면 그 자체가 AdaBN이 된다. 기본 실험에서는 절대 섞지 말 것.
    """
    rng = np.random.default_rng(seed)
    H = cache.meta["H"]
    B = cache.meta["bands"]
    acc = np.zeros(B, np.float64)
    acc2 = np.zeros(B, np.float64)
    n = np.zeros(B, np.float64)
    for _ in range(n_sample):
        r = rng.integers(0, H - patch)
        c = rng.integers(col_lo, max(col_lo + 1, col_hi - patch))
        blk = np.asarray(cache.x[:, r:r + patch, c:c + patch], dtype=np.float64)
        ok = np.isfinite(blk)
        blk = np.where(ok, blk, 0.0)
        acc += blk.sum(axis=(1, 2))
        acc2 += (blk ** 2).sum(axis=(1, 2))
        n += ok.sum(axis=(1, 2))          # NaN 픽셀은 분모에서도 뺀다
    n = np.maximum(n, 1)
    mean = acc / n
    std = np.sqrt(np.maximum(acc2 / n - mean ** 2, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


def class_weights(cache, col_lo, col_hi):
    """class_weight='balanced' 와 동일: w_c = N / (k * n_c)."""
    y = np.asarray(cache.y[:, col_lo:col_hi])
    counts = np.array([(y == c).sum() for c in CLASSES], dtype=np.float64)
    counts = np.maximum(counts, 1)
    w = counts.sum() / (len(CLASSES) * counts)
    return w.astype(np.float32)
