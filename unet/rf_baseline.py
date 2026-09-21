"""동일 분할 RF 대조군 — U-Net과 **완전히 같은 픽셀**로 RF를 돌린다.

문서의 RF 수치(대전 동서 블록 0.572 등)는 분할 시드·표본이 달라서
U-Net과 직접 비교하면 모델 차이인지 분할 차이인지 섞인다.
여기서는 prepare_data가 만든 같은 캐시·같은 분할열을 쓰므로
**바뀌는 것은 모델뿐**이다.

사용:
    python rf_baseline.py --region daejeon --cache-dir ./cache --seeds 3
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CLASSES, NAMES, N_EVAL_PX, N_TRAIN_PX, carbon_err_from_cm,
                    confusion, macro_f1_from_cm)
from data import RegionCache

LIGHT = dict(n_estimators=80, min_samples_leaf=20, max_features="sqrt")
FULL = dict(n_estimators=150, min_samples_leaf=5, max_features="sqrt")


def gather(cache, rows, cols):
    """memmap에서 지정 픽셀의 전체 피처를 뽑는다(피처별로 읽어야 빠르다)."""
    x = cache.x
    out = np.stack([np.asarray(x[i][rows, cols], dtype=np.float32)
                    for i in range(x.shape[0])], axis=1)
    np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def run(region, cache_dir, cfg, seed, n_train, n_eval, train_col_hi=None):
    from sklearn.ensemble import RandomForestClassifier

    cache = RegionCache(region, cache_dir)
    y_map = np.asarray(cache.y)
    split = cache.split_col
    rng = np.random.default_rng(seed)

    # U-Net이 검증 띠를 떼고 학습하므로, 공정하게 하려면 RF도 같은 띠만 써야 한다
    tr_hi = train_col_hi or split
    out = {}
    for side, sl in (("west", slice(0, tr_hi)), ("east", slice(split, None))):
        r, c = np.nonzero(y_map[:, sl] > 0)
        if side == "east":
            c = c + split
        out[side] = (r, c)

    n = min(n_train, len(out["west"][0]))
    pick = rng.choice(len(out["west"][0]), n, replace=False)
    tr_r, tr_c = out["west"][0][pick], out["west"][1][pick]
    n = min(n_eval, len(out["east"][0]))
    pick = rng.choice(len(out["east"][0]), n, replace=False)
    ev_r, ev_c = out["east"][0][pick], out["east"][1][pick]

    Xtr, ytr = gather(cache, tr_r, tr_c), y_map[tr_r, tr_c]
    Xev, yev = gather(cache, ev_r, ev_c), y_map[ev_r, ev_c]

    clf = RandomForestClassifier(class_weight="balanced", random_state=seed,
                                 n_jobs=-1, **cfg)
    clf.fit(Xtr, ytr)
    cm = confusion(yev, clf.predict(Xev))
    f1, per = macro_f1_from_cm(cm)
    net, gross = carbon_err_from_cm(cm)
    return dict(seed=seed, macro_f1=f1, per_class=per, carbon_net=net,
                carbon_gross=gross, n_train=len(ytr), n_eval=len(yev),
                cm=cm.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    ap.add_argument("--cache-dir", default="./cache")
    ap.add_argument("--cfg", choices=["light", "full"], default="light")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed0", type=int, default=42)
    ap.add_argument("--n-train", type=int, default=N_TRAIN_PX)
    ap.add_argument("--n-eval", type=int, default=N_EVAL_PX)
    ap.add_argument("--train-col-hi", type=int, default=None,
                    help="학습에 쓸 열 상한. U-Net의 학습 띠와 맞출 때 지정")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = LIGHT if a.cfg == "light" else FULL
    runs = []
    for i in range(a.seeds):
        r = run(a.region, a.cache_dir, cfg, a.seed0 + i, a.n_train, a.n_eval,
                a.train_col_hi)
        runs.append(r)
        print("seed %d  macro F1 %.3f  ( %s )  탄소 순 %+.2f%%  총 %.2f%%"
              % (r["seed"], r["macro_f1"],
                 " ".join("%s %.3f" % (NAMES[c], v)
                          for c, v in zip(CLASSES, r["per_class"])),
                 r["carbon_net"], r["carbon_gross"]), flush=True)

    f1 = np.array([r["macro_f1"] for r in runs])
    cn = np.array([r["carbon_net"] for r in runs])
    mix = np.array([r["per_class"][2] for r in runs])
    print("\n== RF 대조군 (%s, %d시드, 동서 분할) ==" % (a.cfg, a.seeds))
    if a.seeds > 1:
        print("macro F1   %.3f ± %.3f" % (f1.mean(), f1.std(ddof=1)))
        print("혼효 F1    %.3f ± %.3f" % (mix.mean(), mix.std(ddof=1)))
        print("탄소 순오차 %+.2f ± %.2f %%p" % (cn.mean(), cn.std(ddof=1)))
    else:
        print("macro F1 %.3f  혼효 F1 %.3f  탄소 순오차 %+.2f%%"
              % (f1.mean(), mix.mean(), cn.mean()))

    out = a.out or ("rf_%s_ew_%s.json" % (a.region, a.cfg))
    json.dump(dict(region=a.region, cfg=a.cfg, protocol="east-west", runs=runs),
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("저장: " + out)


if __name__ == "__main__":
    main()
