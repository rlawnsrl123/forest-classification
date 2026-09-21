"""RF 9칸 표 재현 + 프로토콜 재설계 + 시드 반복 — npz 3개만 있으면 된다.

이 스크립트 하나가 인수인계 문서 10절의 3개 항목을 덮는다.
  · POC 1단계  : 03-[6] 9칸 표 재현 → 데이터·환경 검증
  · 할 일 3    : --protocol ew  (동서 블록 평가셋 재설계, 대각/비대각 교락 해소)
  · 할 일 4    : --seeds 5      (진짜 표준편차 확보. 현재 노이즈 0.239%p는 [유도]값)

tile = (row//500)*10000 + (col//500) 이므로 npz만으로 동서 분할이 가능하다
(tif 없이도 열 블록 = tile % 10000).

사용:
    python rf_reproduce.py --npz-dir ./npz --protocol legacy
    python rf_reproduce.py --npz-dir ./npz --protocol ew --seeds 5
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ABBR, CLASSES, N_EVAL_PX, N_TRAIN_PX, REGIONS,
                    carbon_err_from_cm, confusion, macro_f1_from_cm)

LIGHT = dict(n_estimators=80, min_samples_leaf=20, max_features="sqrt")
FULL = dict(n_estimators=150, min_samples_leaf=5, max_features="sqrt")


def load(npz_dir, region):
    z = np.load(os.path.join(npz_dir, region + "_samples.npz"))
    d = {k: z[k] for k in z.files}
    d["col_blk"] = d["tile"] % 10000
    d["row_blk"] = d["tile"] // 10000
    return d


def split_masks(d, protocol, rng):
    """(학습 가능 마스크, 평가 가능 마스크) 반환."""
    if protocol == "ew":
        # 유효 픽셀 기준 중앙 열 블록에서 자른다. 서=학습, 동=평가
        blks = np.sort(np.unique(d["col_blk"]))
        counts = np.array([(d["col_blk"] == b).sum() for b in blks])
        cut = blks[np.searchsorted(np.cumsum(counts), counts.sum() // 2)]
        return d["col_blk"] < cut, d["col_blk"] >= cut
    # legacy: 타일 30%를 홀드아웃. 지역 내는 홀드아웃에서, 지역 간은 전체에서 평가
    tiles = np.unique(d["tile"])
    held = set(rng.choice(tiles, max(1, int(len(tiles) * 0.3)), replace=False).tolist())
    held_m = np.isin(d["tile"], list(held))
    return ~held_m, held_m


def sample(idx, n, rng):
    return rng.choice(idx, min(n, len(idx)), replace=False)


def run(npz_dir, protocol, cfg, seed, regions):
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(seed)
    data = {r: load(npz_dir, r) for r in regions}
    masks = {r: split_masks(data[r], protocol, rng) for r in regions}

    cells = {}
    for tr in regions:
        d = data[tr]
        tr_idx = sample(np.nonzero(masks[tr][0])[0], N_TRAIN_PX, rng)
        clf = RandomForestClassifier(class_weight="balanced", random_state=seed,
                                     n_jobs=-1, **cfg)
        clf.fit(d["X"][tr_idx], d["y"][tr_idx])

        for ev in regions:
            de = data[ev]
            if protocol == "legacy" and tr != ev:
                pool = np.arange(len(de["y"]))          # 비대각은 지역 전체
            else:
                pool = np.nonzero(masks[ev][1])[0]
            ev_idx = sample(pool, N_EVAL_PX, rng)
            yp = clf.predict(de["X"][ev_idx])
            cm = confusion(de["y"][ev_idx], yp)
            f1, per = macro_f1_from_cm(cm)
            net, gross = carbon_err_from_cm(cm)
            cells[ABBR[tr] + "_" + ABBR[ev]] = dict(
                train=tr, eval=ev, macro_f1=f1, per_class=per,
                carbon_net=net, carbon_gross=gross, n_px=int(len(ev_idx)),
                cm=cm.tolist())
            print("  %s→%s  F1 %.3f  탄소 %+.2f%%  총 %.2f%%"
                  % (ABBR[tr], ABBR[ev], f1, net, gross), flush=True)
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--protocol", choices=["legacy", "ew"], default="legacy")
    ap.add_argument("--cfg", choices=["light", "full"], default="light")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed0", type=int, default=42)
    ap.add_argument("--regions", nargs="*", default=list(REGIONS))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = LIGHT if a.cfg == "light" else FULL
    runs = []
    for i in range(a.seeds):
        s = a.seed0 + i
        print("\n=== seed %d (%s, %s) ===" % (s, a.protocol, a.cfg))
        runs.append(run(a.npz_dir, a.protocol, cfg, s, a.regions))

    keys = list(runs[0].keys())
    print("\n== 요약 (%d시드) ==" % a.seeds)
    print("%-10s %16s %16s" % ("조합", "macro F1", "탄소 순오차(%)"))
    summary = {}
    for k in keys:
        f1s = np.array([r[k]["macro_f1"] for r in runs])
        cs = np.array([r[k]["carbon_net"] for r in runs])
        summary[k] = dict(macro_f1_mean=f1s.mean(), macro_f1_sd=f1s.std(ddof=1) if a.seeds > 1 else None,
                          carbon_mean=cs.mean(), carbon_sd=cs.std(ddof=1) if a.seeds > 1 else None,
                          train=runs[0][k]["train"], eval=runs[0][k]["eval"])
        if a.seeds > 1:
            print("%-10s %8.3f ±%.3f %9.2f ±%.2f"
                  % (k, f1s.mean(), f1s.std(ddof=1), cs.mean(), cs.std(ddof=1)))
        else:
            print("%-10s %8.3f        %9.2f" % (k, f1s.mean(), cs.mean()))

    if a.seeds > 1:
        sds = [v["carbon_sd"] for v in summary.values()]
        print("\n★ 탄소 오차 시드 SD: 평균 %.3f%%p  최대 %.3f%%p"
              % (np.mean(sds), np.max(sds)))
        print("  (문서의 노이즈 0.239%p는 실행 2회의 최댓값에서 온 [유도]값 — 이 값이 실측)")

    out = a.out or ("rf_%s_%s_%dseed.json" % (a.protocol, a.cfg, a.seeds))
    json.dump(dict(protocol=a.protocol, cfg=a.cfg, seeds=a.seeds,
                   summary=summary, runs=runs),
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n저장: " + out)


if __name__ == "__main__":
    main()
