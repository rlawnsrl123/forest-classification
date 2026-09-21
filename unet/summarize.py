"""9칸 결과 JSON들을 모아 시드 평균·표준편차 표로 만든다.

    python summarize.py --rf rf_9cell.json \
        --unet unet_9cell.json unet_9cell_s43.json unet_9cell_s44.json \
        --adabn unet_9cell_adabn_s42.json unet_9cell_adabn_s43.json unet_9cell_adabn_s44.json
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0] if "\\" in __file__ else ".")
from common import ABBR, REGIONS

KEYS = [ABBR[a] + "_" + ABBR[b] for a in REGIONS for b in REGIONS]
# 문서 03-[7] 확정 계수 기준값
DOC = {"dae_dae": -1.05, "dae_hon": 2.73, "dae_sun": -1.57,
       "hon_dae": -1.51, "hon_hon": 0.12, "hon_sun": -3.48,
       "sun_dae": 3.12, "sun_hon": 5.92, "sun_sun": 0.84}


def load(paths):
    out = []
    for p in paths:
        d = json.load(open(p, encoding="utf-8"))
        out.append(d.get("cells") or d.get("avg"))
    return out


def stat(runs, key, field, idx=None):
    v = [(r[key]["per_class"][idx] if idx is not None else r[key][field])
         for r in runs]
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)


def table(title, runs, field, idx=None, fmt="%+6.2f"):
    print("\n== %s ==" % title)
    print("%-7s %16s %16s %16s" % ("학습\\평가", "dae", "hon", "sun"))
    for a in REGIONS:
        row = []
        for b in REGIONS:
            m, s = stat(runs, ABBR[a] + "_" + ABBR[b], field, idx)
            row.append(("%s ±%.2f" % (fmt % m, s)).rjust(16))
        print("%-7s %s" % (ABBR[a], " ".join(row)))


def agg(runs, field, idx=None):
    inn, out = [], []
    for k in KEYS:
        m, _ = stat(runs, k, field, idx)
        (inn if k[:3] == k[4:] else out).append(m)
    return np.mean(np.abs(inn)), np.mean(np.abs(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rf", nargs="+", required=True)
    ap.add_argument("--unet", nargs="+", required=True)
    ap.add_argument("--adabn", nargs="*", default=[])
    a = ap.parse_args()

    sets = [("RF", load(a.rf)), ("U-Net", load(a.unet))]
    if a.adabn:
        sets.append(("U-Net+AdaBN", load(a.adabn)))

    for name, runs in sets:
        print("\n" + "#" * 62)
        print("# %s  (시드 %d개)" % (name, len(runs)))
        print("#" * 62)
        table("macro F1", runs, "macro_f1", fmt="%6.3f")
        table("탄소 순오차(%)", runs, "carbon_net")
        table("탄소 총오차(%)", runs, "carbon_gross", fmt="%6.2f")
        table("혼효 F1", runs, "macro_f1", idx=2, fmt="%6.3f")

    print("\n" + "=" * 70)
    print("%-14s %10s %10s %8s %10s %10s" % (
        "", "지역내|순|", "지역간|순|", "배율", "지역내 총", "지역간 총"))
    print("%-14s %10.2f %10.2f %8.1f %10.2f %10.2f" % (
        "문서 RF", 0.67, 3.05, 3.05 / 0.67, 4.67, 5.84))
    for name, runs in sets:
        i1, o1 = agg(runs, "carbon_net")
        i2, o2 = agg(runs, "carbon_gross")
        print("%-14s %10.2f %10.2f %8.1f %10.2f %10.2f"
              % (name, i1, o1, o1 / max(i1, 1e-9), i2, o2))

    print("\n== 부호 일치: 문서 03-[7] 대비 ==")
    for name, runs in sets:
        same = sum(1 for k in KEYS
                   if np.sign(stat(runs, k, "carbon_net")[0]) == np.sign(DOC[k]))
        off = sum(1 for k in KEYS if k[:3] != k[4:]
                  and np.sign(stat(runs, k, "carbon_net")[0]) == np.sign(DOC[k]))
        print("  %-14s 9칸 중 %d칸 일치 (지역 간 6칸 중 %d칸)" % (name, same, off))

    print("\n== macro F1 평균 ==")
    for name, runs in sets:
        v = [stat(runs, k, "macro_f1")[0] for k in KEYS]
        vi = [stat(runs, k, "macro_f1")[0] for k in KEYS if k[:3] == k[4:]]
        vo = [stat(runs, k, "macro_f1")[0] for k in KEYS if k[:3] != k[4:]]
        print("  %-14s 전체 %.3f  지역내 %.3f  지역간 %.3f"
              % (name, np.mean(v), np.mean(vi), np.mean(vo)))


if __name__ == "__main__":
    main()
