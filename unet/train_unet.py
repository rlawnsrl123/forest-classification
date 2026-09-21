"""한 지역으로 U-Net 학습. 프로토콜은 RF와 동일(서쪽만 학습, 2024+ 라벨).

사용:
    python train_unet.py --region hongcheon --cache-dir ./cache --out ./runs
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLASSES, IGNORE, NAMES, SEED, carbon_err_from_cm, macro_f1_from_cm
from data import PatchDataset, RegionCache, band_stats, class_weights, patch_origins
from model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    ap.add_argument("--cache-dir", default="./cache")
    ap.add_argument("--out", default="./runs")
    ap.add_argument("--arch", default="unet")
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--base", type=int, default=32, help="U-Net 첫 층 채널 수")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--min-valid", type=float, default=0.20)
    ap.add_argument("--val-col-frac", type=float, default=0.15,
                    help="서쪽의 오른쪽 끝 몇 %% 를 검증 띠로 뗄지. "
                         "무작위 패치로 나누면 stride<patch 라 학습 패치와 겹쳐 "
                         "검증이 새기 때문에 반드시 열 단위로 자른다")
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    tag = a.region + "_" + a.arch.replace(":", "-") + "_s" + str(a.seed) + a.tag_suffix

    cache = RegionCache(a.region, a.cache_dir)
    split = cache.split_col
    valid = np.asarray(cache.y) > 0
    print("[%s] %dx%d  분할열 %d"
          % (a.region, cache.meta["H"], cache.meta["W"], split))

    # 서쪽을 다시 [학습 띠 | 검증 띠] 로 자른다. 패치는 각 띠 안에 완전히 들어가야
    # 한다 → 학습·검증 패치가 한 픽셀도 겹치지 않는다.
    val_col = int(split * (1.0 - a.val_col_frac))
    tr_o = patch_origins(valid, a.patch, a.stride, 0, val_col, a.min_valid)
    val_o = patch_origins(valid, a.patch, a.patch, val_col, split, a.min_valid)
    if len(tr_o) == 0:
        raise SystemExit("유효 패치 0개 — min-valid 를 낮추거나 데이터를 확인하세요")
    print("학습 띠 0:%d (패치 %d) / 검증 띠 %d:%d (패치 %d)"
          % (val_col, len(tr_o), val_col, split, len(val_o)))

    mean, std = band_stats(cache, a.patch, 0, val_col, seed=a.seed)
    w = class_weights(cache, 0, val_col)
    print("클래스 가중치 " + "  ".join(
        "%s:%.3f" % (NAMES[c], v) for c, v in zip(CLASSES, w)))

    tr = DataLoader(PatchDataset(cache, tr_o, a.patch, mean, std, True),
                    batch_size=a.batch, shuffle=True, num_workers=a.workers,
                    pin_memory=True, drop_last=True,
                    persistent_workers=a.workers > 0)
    va = DataLoader(PatchDataset(cache, val_o, a.patch, mean, std, False),
                    batch_size=a.batch, shuffle=False, num_workers=0)

    model = build_model(a.arch, cache.meta["bands"], len(CLASSES),
                        base=a.base, depth=a.depth).to(dev)
    model = model.to(memory_format=torch.channels_last)
    crit = nn.CrossEntropyLoss(weight=torch.from_numpy(w).to(dev),
                               ignore_index=IGNORE)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps,
                                                pct_start=0.1)

    def evaluate():
        model.eval()
        cm = np.zeros((3, 3), np.int64)
        with torch.no_grad():
            for xv, yv in va:
                xv = xv.to(dev).to(memory_format=torch.channels_last)
                with torch.autocast(dev, dtype=torch.bfloat16):
                    pv = model(xv).argmax(1).cpu().numpy()
                yv = yv.numpy()
                mv = yv != IGNORE
                np.add.at(cm, (yv[mv], pv[mv]), 1)
        model.train()
        return cm

    step, t0, losses = 0, time.time(), []
    best = (-1.0, None, 0)
    model.train()
    while step < a.steps:
        for x, y in tr:
            if step >= a.steps:
                break
            x = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(dev, non_blocking=True)
            with torch.autocast(dev, dtype=torch.bfloat16):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
            step += 1
            if step % 200 == 0:
                print("  step %5d/%d  loss %.4f  %.1f it/s"
                      % (step, a.steps, float(np.mean(losses[-200:])),
                         step / (time.time() - t0)), flush=True)
            if len(val_o) and step % a.val_every == 0:
                vf1, _ = macro_f1_from_cm(evaluate())
                flag = ""
                if vf1 > best[0]:
                    best = (vf1, {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}, step)
                    flag = "  ← best"
                print("      검증 띠 macro F1 %.3f%s" % (vf1, flag), flush=True)

    # 검증 띠 최고 시점으로 되돌린다 — 과적합 구간을 들고 나가지 않게
    if best[1] is not None:
        model.load_state_dict(best[1])
        print("\n최고 검증 시점 step %d (macro F1 %.3f) 로 복원" % (best[2], best[0]))

    cm = evaluate()
    model.eval()
    f1, per = macro_f1_from_cm(cm)
    net, gross = carbon_err_from_cm(cm)
    print("\n[검증 띠] macro F1 %.3f  " % f1
          + "  ".join("%s %.3f" % (NAMES[c], v) for c, v in zip(CLASSES, per)))
    print("              탄소 순오차 %+.2f%%  총오차 %.2f%%" % (net, gross))

    ckpt = os.path.join(a.out, tag + ".pt")
    torch.save(dict(state=model.state_dict(), arch=a.arch, region=a.region,
                    mean=mean, std=std, patch=a.patch, bands=cache.meta["bands"],
                    base=a.base, depth=a.depth,
                    seed=a.seed, split_col=split), ckpt)
    json.dump(dict(tag=tag, region=a.region, arch=a.arch, steps=a.steps,
                   val_macro_f1=f1, val_per_class=per,
                   val_carbon_net=net, val_carbon_gross=gross,
                   minutes=(time.time() - t0) / 60),
              open(os.path.join(a.out, tag + ".json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("저장: %s  (%.1f분)" % (ckpt, (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
