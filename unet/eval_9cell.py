"""9칸 표 생성 — 학습 3지역 × 평가 3지역, macro F1 + 탄소 오차.

준기 님 RF 9칸 표(03-[7])와 나란히 놓을 수 있게 프로토콜을 고정한다.
  · 평가는 모든 지역에서 동일하게 '동쪽 절반'만 사용
    → 대각(지역 내)과 비대각(지역 간)의 평가셋 성격이 같아진다.
       문서 6-3절이 지적한 '대각/비대각 교락'을 설계로 제거한 것.
  · 유효 픽셀(2024+ 라벨, 클래스 1/2/3)만 혼동행렬에 넣는다.
  · 탄소계수는 확정값(common.C).

AdaBN(--adabn): 학습된 모델의 BatchNorm 통계를 평가 지역 영상으로 다시 잰다.
  라벨을 쓰지 않으므로 α(RF의 지역별 정규화)와 같은 성격의 조작이다.
  기본값은 평가 지역의 '서쪽'(= 평가에 쓰지 않는 영역)으로 통계를 재추정한다.
  --adabn-source east 로 하면 전이학습 문헌의 통상적 transductive 설정이 된다.

사용:
    python eval_9cell.py --cache-dir ./cache --runs ./runs --out ninecell.json
    python eval_9cell.py --cache-dir ./cache --runs ./runs --adabn --out ninecell_adabn.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ABBR, CLASSES, NAMES, REGIONS, carbon_err_from_cm,
                    macro_f1_from_cm)
from data import RegionCache, normalize, patch_origins
from model import build_model, reset_bn


def _patch_grid(H, W, patch, col_lo, col_hi, stride=None):
    """동쪽 영역을 빠짐없이 덮는 패치 좌상단 좌표. 가장자리는 안쪽으로 당긴다."""
    stride = stride or patch
    rows = list(range(0, max(H - patch, 0) + 1, stride))
    if rows[-1] != max(H - patch, 0):
        rows.append(max(H - patch, 0))
    cols = list(range(col_lo, max(col_hi - patch, col_lo) + 1, stride))
    if cols[-1] != max(col_hi - patch, col_lo):
        cols.append(max(col_hi - patch, col_lo))
    return [(r, c) for r in rows for c in cols]


@torch.no_grad()
def predict_region(model, cache, mean, std, patch, col_lo, col_hi, dev, batch=32,
                   overlap=True):
    """col_lo:col_hi 구간 전체에 대한 예측 맵(1/2/3, 미예측 0)을 만든다.

    overlap=True 면 stride = patch/2 로 훑고 확률을 누적 평균한다.
    U-Net은 패치 가장자리에서 문맥이 잘려 약한데, 겹치지 않게 이어붙이면
    그 경계가 평가 결과에 그대로 들어가 모델을 부당하게 깎는다(RF에는 없는 불리함).
    가운데를 더 믿도록 코사인 창으로 가중한다.
    """
    H = cache.meta["H"]
    Wr = col_hi - col_lo
    stride = patch // 2 if overlap else patch
    grid = _patch_grid(H, cache.meta["W"], patch, col_lo, col_hi, stride)
    m3, s3 = mean.reshape(-1, 1, 1), std.reshape(-1, 1, 1)

    prob = np.zeros((len(CLASSES), H, Wr), dtype=np.float32)
    wsum = np.zeros((H, Wr), dtype=np.float32)
    if overlap:
        w1 = np.hanning(patch + 2)[1:-1].astype(np.float32)
        win = np.maximum(np.outer(w1, w1), 1e-3)
    else:
        win = np.ones((patch, patch), dtype=np.float32)

    model.eval()
    for i in range(0, len(grid), batch):
        chunk = grid[i:i + batch]
        xs = np.stack([
            normalize(np.asarray(cache.x[:, r:r + patch, c:c + patch],
                                 dtype=np.float32), m3, s3)
            for r, c in chunk])
        t = torch.from_numpy(xs).to(dev).to(memory_format=torch.channels_last)
        with torch.autocast(dev, dtype=torch.bfloat16):
            out = model(t).float().softmax(1).cpu().numpy()
        for (r, c), p in zip(chunk, out):
            cc = c - col_lo
            prob[:, r:r + patch, cc:cc + patch] += p * win
            wsum[r:r + patch, cc:cc + patch] += win

    seen = wsum > 0
    pred = np.zeros((H, Wr), dtype=np.uint8)
    pred[seen] = (prob[:, seen].argmax(0) + 1).astype(np.uint8)  # 0/1/2 → 1/2/3
    return pred


@torch.no_grad()
def adapt_bn(model, cache, mean, std, patch, col_lo, col_hi, dev,
             n_batches=40, batch=16, seed=0):
    """AdaBN: BN running 통계를 대상 지역 영상으로 라벨 없이 재추정."""
    n = reset_bn(model)
    valid = np.asarray(cache.y) > 0
    origins = patch_origins(valid, patch, patch // 2, col_lo, col_hi, 0.20)
    if len(origins) == 0:
        raise RuntimeError("AdaBN용 패치가 없습니다")
    rng = np.random.default_rng(seed)
    sel = origins[rng.permutation(len(origins))[:n_batches * batch]]
    m3, s3 = mean.reshape(-1, 1, 1), std.reshape(-1, 1, 1)

    model.train()   # running 통계 갱신 모드. 가중치는 건드리지 않는다
    for i in range(0, len(sel), batch):
        chunk = sel[i:i + batch]
        xs = np.stack([
            normalize(np.asarray(cache.x[:, r:r + patch, c:c + patch],
                                 dtype=np.float32), m3, s3)
            for r, c in chunk])
        t = torch.from_numpy(xs).to(dev).to(memory_format=torch.channels_last)
        with torch.autocast(dev, dtype=torch.bfloat16):
            model(t)
    model.eval()
    return n, len(sel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="./cache")
    ap.add_argument("--runs", default="./runs")
    ap.add_argument("--arch", default="unet")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--regions", nargs="*", default=list(REGIONS))
    ap.add_argument("--adabn", action="store_true")
    ap.add_argument("--adabn-source", choices=["west", "east"], default="west")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--no-overlap", action="store_true",
                    help="겹침 추론을 끄고 패치를 그대로 이어붙인다(진단용)")
    ap.add_argument("--out", default="ninecell.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    caches = {r: RegionCache(r, a.cache_dir) for r in a.regions}
    results = {}

    for tr_region in a.regions:
        tag = (tr_region + "_" + a.arch.replace(":", "-") + "_s"
               + str(a.seed) + a.tag_suffix)
        ckpt_path = os.path.join(a.runs, tag + ".pt")
        if not os.path.exists(ckpt_path):
            print("체크포인트 없음, 건너뜀: " + ckpt_path)
            continue
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        mean, std, patch = ck["mean"], ck["std"], ck["patch"]

        for ev_region in a.regions:
            cache = caches[ev_region]
            split = cache.split_col
            W = cache.meta["W"]

            model = build_model(ck["arch"], ck["bands"], len(CLASSES),
                                base=ck.get("base", 32),
                                depth=ck.get("depth", 4)).to(dev)
            model.load_state_dict(ck["state"])
            model = model.to(memory_format=torch.channels_last)

            note = ""
            if a.adabn:
                lo, hi = (0, split) if a.adabn_source == "west" else (split, W)
                nbn, npx = adapt_bn(model, cache, mean, std, patch, lo, hi, dev)
                note = "AdaBN(%s, BN %d층, 패치 %d)" % (a.adabn_source, nbn, npx)

            pred = predict_region(model, cache, mean, std, patch, split, W, dev,
                                  a.batch, overlap=not a.no_overlap)
            y_true = np.asarray(cache.y[:, split:])
            m = np.isin(y_true, CLASSES)
            cm = np.zeros((3, 3), np.int64)
            np.add.at(cm, (y_true[m].astype(np.int64) - 1,
                           pred[m].astype(np.int64) - 1), 1)

            f1, per = macro_f1_from_cm(cm)
            net, gross = carbon_err_from_cm(cm)
            key = ABBR[tr_region] + "_" + ABBR[ev_region]
            results[key] = dict(train=tr_region, eval=ev_region,
                                macro_f1=f1, per_class=per,
                                carbon_net=net, carbon_gross=gross,
                                n_px=int(m.sum()), cm=cm.tolist(), note=note)
            print("%s→%s  F1 %.3f  탄소 %+.2f%%  총 %.2f%%  혼효F1 %.3f  %s"
                  % (ABBR[tr_region], ABBR[ev_region], f1, net, gross, per[2], note),
                  flush=True)

    _print_tables(results, a.regions)
    json.dump(dict(adabn=a.adabn, adabn_source=a.adabn_source, arch=a.arch,
                   seed=a.seed, cells=results),
              open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n저장: " + a.out)


def _print_tables(res, regions):
    cols = [ABBR[r] for r in regions]
    for title, fn in (("macro F1", lambda d: "%.3f" % d["macro_f1"]),
                      ("탄소 순오차(%)", lambda d: "%+.2f" % d["carbon_net"]),
                      ("탄소 총오차(%)", lambda d: "%.2f" % d["carbon_gross"]),
                      ("혼효 F1", lambda d: "%.3f" % d["per_class"][2])):
        print("\n== " + title + " ==")
        print("학습\\평가  " + "  ".join("%8s" % c for c in cols))
        for r in regions:
            row = []
            for e in regions:
                d = res.get(ABBR[r] + "_" + ABBR[e])
                row.append("%8s" % (fn(d) if d else "-"))
            print("%-9s  %s" % (ABBR[r], "  ".join(row)))

    inner = [d for k, d in res.items() if d["train"] == d["eval"]]
    outer = [d for k, d in res.items() if d["train"] != d["eval"]]
    if inner and outer:
        print("\n지역 내 |순오차| 평균 %.2f  / 총오차 %.2f  (n=%d)"
              % (np.mean([abs(d["carbon_net"]) for d in inner]),
                 np.mean([d["carbon_gross"] for d in inner]), len(inner)))
        print("지역 간 |순오차| 평균 %.2f  / 총오차 %.2f  (n=%d)"
              % (np.mean([abs(d["carbon_net"]) for d in outer]),
                 np.mean([d["carbon_gross"] for d in outer]), len(outer)))


if __name__ == "__main__":
    main()
