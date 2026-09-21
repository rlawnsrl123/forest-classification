"""RF 9칸 표 — eval_9cell.py(U-Net)와 **완전히 같은 프로토콜**.

  · 학습: 학습 지역 서쪽의 학습 띠 (0 : split*(1-val_col_frac))
          U-Net이 검증 띠를 떼고 학습하므로 RF도 같은 띠만 쓴다
  · 평가: 평가 지역 동쪽 전체 유효 픽셀
  · 지표: macro F1, 탄소 순오차·총오차 (확정 계수)

이렇게 해야 9칸 표에서 바뀌는 것이 모델뿐이 된다.

사용:
    python rf_9cell.py --cache-dir ./cache --seeds 3 --out rf_9cell.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ABBR, CLASSES, N_TRAIN_PX, REGIONS, carbon_err_from_cm,
                    confusion, macro_f1_from_cm)
from data import RegionCache

LIGHT = dict(n_estimators=80, min_samples_leaf=20, max_features="sqrt")
FULL = dict(n_estimators=150, min_samples_leaf=5, max_features="sqrt")


def gather(cache, rows, cols):
    x = cache.x
    out = np.stack([np.asarray(x[i][rows, cols], dtype=np.float32)
                    for i in range(x.shape[0])], axis=1)
    np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def band_pixels(cache, lo, hi):
    y = np.asarray(cache.y)[:, lo:hi]
    r, c = np.nonzero(y > 0)
    return r, c + lo, np.asarray(cache.y)[r, c + lo]


def run(cache_dir, regions, cfg, seed, n_train, n_eval, val_col_frac):
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(seed)
    caches = {r: RegionCache(r, cache_dir) for r in regions}

    # 평가 픽셀은 조합마다 동일해야 하므로 미리 한 번만 뽑는다
    ev = {}
    for r in regions:
        c = caches[r]
        rr, cc, yy = band_pixels(c, c.split_col, c.meta["W"])
        if n_eval and len(rr) > n_eval:
            k = rng.choice(len(rr), n_eval, replace=False)
            rr, cc, yy = rr[k], cc[k], yy[k]
        ev[r] = (gather(c, rr, cc), yy)
        print("  [%s] 평가 픽셀 %d" % (r, len(yy)), flush=True)

    cells = {}
    for tr in regions:
        c = caches[tr]
        hi = int(c.split_col * (1.0 - val_col_frac))
        rr, cc, yy = band_pixels(c, 0, hi)
        k = rng.choice(len(rr), min(n_train, len(rr)), replace=False)
        Xtr, ytr = gather(c, rr[k], cc[k]), yy[k]
        clf = RandomForestClassifier(class_weight="balanced", random_state=seed,
                                     n_jobs=-1, **cfg)
        clf.fit(Xtr, ytr)
        print("  [%s] 학습 완료 (띠 0:%d, %d px)" % (tr, hi, len(ytr)), flush=True)

        for evr in regions:
            Xe, ye = ev[evr]
            cm = confusion(ye, clf.predict(Xe))
            f1, per = macro_f1_from_cm(cm)
            net, gross = carbon_err_from_cm(cm)
            cells[ABBR[tr] + "_" + ABBR[evr]] = dict(
                train=tr, eval=evr, macro_f1=f1, per_class=per,
                carbon_net=net, carbon_gross=gross, n_px=int(len(ye)),
                cm=cm.tolist())
            print("    %s→%s  F1 %.3f  탄소 %+.2f%%  총 %.2f%%  혼효F1 %.3f"
                  % (ABBR[tr], ABBR[evr], f1, net, gross, per[2]), flush=True)
    return cells


def print_tables(res, regions):
    cols = [ABBR[r] for r in regions]
    for title, fn in (("macro F1", lambda d: "%.3f" % d["macro_f1"]),
                      ("탄소 순오차(%)", lambda d: "%+.2f" % d["carbon_net"]),
                      ("탄소 총오차(%)", lambda d: "%.2f" % d["carbon_gross"]),
                      ("혼효 F1", lambda d: "%.3f" % d["per_class"][2])):
        print("\n== " + title + " ==")
        print("학습\\평가  " + "  ".join("%8s" % c for c in cols))
        for r in regions:
            row = [("%8s" % fn(res[ABBR[r] + "_" + ABBR[e]]))
                   if ABBR[r] + "_" + ABBR[e] in res else "%8s" % "-"
                   for e in regions]
            print("%-9s  %s" % (ABBR[r], "  ".join(row)))
    inner = [d for d in res.values() if d["train"] == d["eval"]]
    outer = [d for d in res.values() if d["train"] != d["eval"]]
    if inner and outer:
        print("\n지역 내 |순오차| %.2f  총오차 %.2f  (n=%d)"
              % (np.mean([abs(d["carbon_net"]) for d in inner]),
                 np.mean([d["carbon_gross"] for d in inner]), len(inner)))
        print("지역 간 |순오차| %.2f  총오차 %.2f  (n=%d)"
              % (np.mean([abs(d["carbon_net"]) for d in outer]),
                 np.mean([d["carbon_gross"] for d in outer]), len(outer)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="./cache")
    ap.add_argument("--regions", nargs="*", default=list(REGIONS))
    ap.add_argument("--cfg", choices=["light", "full"], default="light")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed0", type=int, default=42)
    ap.add_argument("--n-train", type=int, default=N_TRAIN_PX)
    ap.add_argument("--n-eval", type=int, default=0,
                    help="0 = 동쪽 전체. 양수면 그만큼만 표본")
    ap.add_argument("--val-col-frac", type=float, default=0.25,
                    help="U-Net 검증 띠와 동일하게 서쪽에서 떼는 비율")
    ap.add_argument("--out", default="rf_9cell.json")
    a = ap.parse_args()

    cfg = LIGHT if a.cfg == "light" else FULL
    runs = []
    for i in range(a.seeds):
        print("\n=== seed %d ===" % (a.seed0 + i))
        runs.append(run(a.cache_dir, a.regions, cfg, a.seed0 + i,
                        a.n_train, a.n_eval or None, a.val_col_frac))

    avg = {}
    for k in runs[0]:
        avg[k] = dict(runs[0][k])
        for f in ("macro_f1", "carbon_net", "carbon_gross"):
            vals = [r[k][f] for r in runs]
            avg[k][f] = float(np.mean(vals))
            avg[k][f + "_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
        avg[k]["per_class"] = np.mean([r[k]["per_class"] for r in runs], 0).tolist()
    print_tables(avg, a.regions)

    json.dump(dict(cfg=a.cfg, seeds=a.seeds, protocol="east-west",
                   val_col_frac=a.val_col_frac, avg=avg, runs=runs),
              open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n저장: " + a.out)


if __name__ == "__main__":
    main()
