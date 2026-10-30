"""창의작품경진대회 포스터용 시각자료 생성.

출력: poster/out/figN_*.svg (인쇄용, 텍스트 유지) + .png (300dpi 미리보기)
수치 출처: unet/RESULTS.md, unet/*.json, 프로젝트_인수인계.md (셀 출력 원값)

실행: py poster/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

# ── 스타일 ───────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "Malgun Gothic",
    "axes.unicode_minus": False,
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.labelsize": 15,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#5A6470",
    "axes.labelcolor": "#2B3440",
    "xtick.color": "#2B3440",
    "ytick.color": "#2B3440",
    "svg.fonttype": "none",       # SVG 안 글자를 텍스트로 유지 → 인쇄 시 안 깨짐
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

INK = "#2B3440"
MUTED = "#7A8591"
GRID = "#E3E7EB"
CON = "#2D6A4F"     # 침엽
BRD = "#E09F3E"     # 활엽
MIX = "#8D99AE"     # 혼효
TEAL = "#3D6B7A"    # 템플릿 헤더 톤
OVER = "#C8553D"    # 과대추정
UNDER = "#3A7CA5"   # 과소추정
REG = {"홍천": "#3A7CA5", "대전": "#6C757D", "순천": "#C8553D"}
KEYS = ["dae", "hon", "sun"]
KNAME = {"dae": "대전", "hon": "홍천", "sun": "순천"}


def save(fig, name):
    fig.savefig(OUT / f"{name}.svg")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print("saved", name)


def ygrid(ax):
    ax.yaxis.grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)


# ── 데이터 로드 ──────────────────────────────────────────
rf = json.loads((ROOT / "unet/rf_9cell.json").read_text(encoding="utf-8"))["avg"]


def cell(t, e, k):
    return rf[f"{t}_{e}"][k]


# ═════════════════════════════════════════════════════════
# fig1. 연구 흐름도
# ═════════════════════════════════════════════════════════
def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 3.6); ax.axis("off")
    steps = [
        ("Sentinel-2", "여름 + 낙엽기\n다시기 합성", TEAL),
        ("38개 피처", "밴드 20 · 지수 6\n공간 컨텍스트 12", TEAL),
        ("분류 모델", "Random Forest\nU-Net", TEAL),
        ("임상 지도", "침엽 · 활엽 · 혼효", CON),
        ("탄소축적량", "면적 × 임상별 계수\n→ 오차(%) 평가", OVER),
    ]
    w, h, gap = 2.3, 2.5, 0.55
    x0 = (14 - (len(steps) * w + (len(steps) - 1) * gap)) / 2
    for i, (t, s, c) in enumerate(steps):
        x = x0 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0.55), w, h, boxstyle="round,pad=0,rounding_size=0.18",
                                    fc=c, ec="none"))
        ax.text(x + w / 2, 0.55 + h - 0.6, t, ha="center", va="center", color="white",
                fontsize=19, fontweight="bold")
        ax.text(x + w / 2, 0.55 + 0.85, s, ha="center", va="center", color="white",
                fontsize=13.5, linespacing=1.4)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + w + gap - 0.08, 0.55 + h / 2), xytext=(x + w + 0.08, 0.55 + h / 2),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=2.5, mutation_scale=22))
    ax.text(x0, 0.12, "라벨: 산림청 임상도 1:5,000 (갱신년도 2024년 이후만 사용)",
            fontsize=12.5, color=MUTED)
    save(fig, "fig1_pipeline")


# ═════════════════════════════════════════════════════════
# fig2. 다시기 근거 — 여름 vs 낙엽기 NDVI
# ═════════════════════════════════════════════════════════
def fig2_seasonal():
    cls = ["침엽수림", "혼효림", "활엽수림"]
    summer = [0.852, 0.873, 0.885]
    winter = [0.646, 0.539, 0.438]
    cols = [CON, MIX, BRD]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, vals, title, spread in [(axes[0], summer, "여름 (6~9월)", 0.033),
                                    (axes[1], winter, "낙엽기 (11~2월)", 0.208)]:
        b = ax.bar(cls, vals, color=cols, width=0.62)
        for r, v in zip(b, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center",
                    fontsize=15, color=INK)
        ax.set_title(title, color=INK, pad=12)
        ax.set_ylim(0, 1.05)
        ygrid(ax)
        ax.text(0.5, -0.2, f"클래스 간 폭 {spread:.3f}", transform=ax.transAxes, ha="center",
                fontsize=17, fontweight="bold", color=OVER if spread > 0.1 else MUTED)
    axes[0].set_ylabel("NDVI 중앙값 (대전)")
    fig.suptitle("낙엽기를 함께 보면 세 임상이 약 6배 더 잘 갈린다", fontsize=19,
                 fontweight="bold", color=INK, y=1.07)
    save(fig, "fig2_seasonal_ndvi")


# ═════════════════════════════════════════════════════════
# fig3. 평가 프로토콜 (동서 분할)
# ═════════════════════════════════════════════════════════
def fig3_protocol():
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.4); ax.axis("off")
    x, y, W, H = 0.5, 0.9, 9.0, 2.6
    split = W / 2
    ax.add_patch(Rectangle((x, y), split * 0.75, H, fc=CON, alpha=0.85))
    ax.add_patch(Rectangle((x + split * 0.75, y), split * 0.25, H, fc=MIX, alpha=0.9))
    ax.add_patch(Rectangle((x + split, y), split, H, fc=BRD, alpha=0.85))
    ax.add_patch(Rectangle((x, y), W, H, fill=False, ec=INK, lw=1.5))
    ax.text(x + split * 0.375, y + H / 2, "학습", ha="center", va="center", color="white",
            fontsize=22, fontweight="bold")
    ax.text(x + split * 0.875, y + H / 2, "검증", ha="center", va="center", color="white",
            fontsize=17, fontweight="bold")
    ax.text(x + split * 1.5, y + H / 2, "평가\n(모든 조합 공통)", ha="center", va="center",
            color="white", fontsize=20, fontweight="bold", linespacing=1.3)
    ax.text(x + split * 0.5, y + H + 0.3, "서쪽 절반", ha="center", fontsize=15, color=INK)
    ax.text(x + split * 1.5, y + H + 0.3, "동쪽 절반", ha="center", fontsize=15, color=INK)
    ax.text(x, 0.35, "· 픽셀 무작위 분할 금지 — 인접 픽셀이 학습·평가에 섞이면 성능이 부풀려짐\n"
                     "· 지역 내 / 지역 간 조합이 같은 평가 영역을 쓰도록 설계 (평가셋 교락 제거)",
            fontsize=13, color=MUTED, va="center", linespacing=1.6)
    save(fig, "fig3_protocol")


# ═════════════════════════════════════════════════════════
# fig4. 9조합 히트맵 — 탄소 순오차 / macro F1 (RF, 시드 3개 평균)
# ═════════════════════════════════════════════════════════
def fig4_heatmaps():
    C = np.array([[cell(t, e, "carbon_net") for e in KEYS] for t in KEYS])
    F = np.array([[cell(t, e, "macro_f1") for e in KEYS] for t in KEYS])
    labels = [KNAME[k] for k in KEYS]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    fig.subplots_adjust(wspace=0.45)

    lim = 6
    im = axes[0].imshow(C, cmap="RdBu_r", vmin=-lim, vmax=lim)
    for i in range(3):
        for j in range(3):
            v = C[i, j]
            axes[0].text(j, i, f"{v:+.2f}%", ha="center", va="center", fontsize=19,
                         fontweight="bold" if i != j else "normal",
                         color="white" if abs(v) > 3.2 else INK)
    axes[0].set_title("탄소축적량 순오차", color=INK, pad=12)
    cb = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    cb.outline.set_visible(False)

    im2 = axes[1].imshow(F, cmap="Greens", vmin=0.40, vmax=0.65)
    for i in range(3):
        for j in range(3):
            v = F[i, j]
            axes[1].text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=19,
                         color="white" if v > 0.56 else INK)
    axes[1].set_title("분류 성능 (macro F1)", color=INK, pad=12)
    cb2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cb2.outline.set_visible(False)

    for ax in axes:
        ax.set_xticks(range(3), labels); ax.set_yticks(range(3), labels)
        ax.set_xlabel("평가 지역"); ax.set_ylabel("학습 지역")
        ax.xaxis.set_label_position("top"); ax.xaxis.tick_top()
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        for k in range(3):  # 대각선(지역 내) 테두리
            ax.add_patch(Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False, ec=INK, lw=2.5))
    fig.text(0.5, -0.02, "빨강 = 과대추정 · 파랑 = 과소추정 · 굵은 테두리 = 지역 내   ·   Random Forest, 시드 3개 평균, 확정 탄소계수",
             ha="center", fontsize=12.5, color=MUTED)
    save(fig, "fig4_transfer_heatmaps")


# ═════════════════════════════════════════════════════════
# fig5. 지역 내 vs 지역 간 |순오차| / 총오차
# ═════════════════════════════════════════════════════════
def fig5_within_vs_cross():
    within = [abs(cell(k, k, "carbon_net")) for k in KEYS]
    cross = [abs(cell(t, e, "carbon_net")) for t in KEYS for e in KEYS if t != e]
    gw = [cell(k, k, "carbon_gross") for k in KEYS]
    gc = [cell(t, e, "carbon_gross") for t in KEYS for e in KEYS if t != e]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (a, b), title in [(axes[0], (within, cross), "순오차 |net| — 지역 총량이 틀린 정도"),
                              (axes[1], (gw, gc), "총오차 gross — 틀린 픽셀의 양")]:
        m = [np.mean(a), np.mean(b)]
        bars = ax.bar(["지역 내", "지역 간"], m, color=[MIX, OVER], width=0.55)
        rng = np.random.default_rng(0)
        for xi, pts in enumerate([a, b]):
            ax.scatter(xi + 0.38 + rng.uniform(-0.04, 0.04, len(pts)), pts, s=46, color=INK, zorder=3,
                       alpha=0.75)
        for r, v in zip(bars, m):
            ax.text(r.get_x() + r.get_width() / 2, v * 0.5, f"{v:.2f}%", ha="center",
                    va="center", fontsize=20, fontweight="bold", color="white", zorder=4)
        ax.set_title(title, fontsize=15.5, color=INK, pad=10)
        ax.set_ylim(0, max(max(a), max(b), max(m)) * 1.25)
        ygrid(ax)
    axes[0].set_ylabel("탄소 오차 (%)")
    r1 = np.mean(cross) / np.mean(within)
    r2 = np.mean(gc) / np.mean(gw)
    fig.suptitle(f"틀리는 양은 {r2:.1f}배인데, 총량 오차는 {r1:.1f}배", fontsize=19,
                 fontweight="bold", color=INK, y=1.03)
    fig.text(0.5, -0.03, "점 = 개별 조합 (지역 내 3, 지역 간 6)   ·   지역 간 오차는 서로 상쇄되지 않고 한쪽으로 쏠린다",
             ha="center", fontsize=12.5, color=MUTED)
    save(fig, "fig5_within_vs_cross")


# ═════════════════════════════════════════════════════════
# fig6. F1로는 탄소 오차를 예측할 수 없다
# ═════════════════════════════════════════════════════════
def fig6_f1_vs_carbon():
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for t in KEYS:
        for e in KEYS:
            f, c = cell(t, e, "macro_f1"), cell(t, e, "carbon_net")
            same = t == e
            ax.scatter(f, c, s=260, color=REG[KNAME[e]], marker="o" if same else "D",
                       ec="white", lw=1.5, zorder=3)
            off = {("dae", "dae"): (8, -34), ("hon", "hon"): (-40, 16), ("hon", "sun"): (14, 8)}.get((t, e), (12, 6))
            ax.annotate(f"{KNAME[t]}→{KNAME[e]}", (f, c), xytext=off, textcoords="offset points",
                        fontsize=13.5, color=INK)
    ax.axhline(0, color=MUTED, lw=1.2, ls="--")
    ax.set_xlabel("macro F1 (분류 성능)")
    ax.set_ylabel("탄소 순오차 (%)")
    ygrid(ax)
    xs = [cell(t, e, "macro_f1") for t in KEYS for e in KEYS]
    ys = [cell(t, e, "carbon_net") for t in KEYS for e in KEYS]
    r = np.corrcoef(xs, ys)[0, 1]
    ax.set_title("분류 성능이 비슷해도 탄소 오차의 크기·방향은 제각각", color=INK, pad=12, fontsize=17)
    ax.set_xlim(0.47, 0.64)
    fig.text(0.5, -0.03, f"● 지역 내   ◆ 지역 간   (색 = 평가 지역: 파랑 홍천 · 회색 대전 · 빨강 순천)   ·   상관계수 r = {r:.2f}",
             ha="center", fontsize=12.5, color=MUTED)
    save(fig, "fig6_f1_vs_carbon")


# ═════════════════════════════════════════════════════════
# fig7. 근본 원인 — 지역별 분광 기준선(오프셋)
# ═════════════════════════════════════════════════════════
def fig7_offset():
    data = {  # 낙엽기 NDVI 클래스 중앙값 (03-[15])
        "홍천": (0.369, 0.538, 0.626),
        "대전": (0.440, 0.655, 0.657),
        "순천": (0.515, 0.742, 0.760),
    }
    fig, ax = plt.subplots(figsize=(12, 5.2))
    regions = list(data)
    for i, r in enumerate(regions):
        y = len(regions) - 1 - i
        b, c, pine = data[r]
        bnd = (b + c) / 2
        ax.plot([b, c], [y, y], color=GRID, lw=10, solid_capstyle="round", zorder=1)
        ax.scatter(b, y, s=420, color=BRD, zorder=3, ec="white", lw=2)
        ax.scatter(c, y, s=420, color=CON, zorder=3, ec="white", lw=2)
        ax.scatter(pine, y, s=230, marker="^", color=INK, zorder=4)
        ax.plot([bnd, bnd], [y - 0.28, y + 0.28], color=REG[r], lw=3.5, zorder=2)
        ax.text(b, y - 0.42, f"{b:.3f}", ha="center", fontsize=13, color=INK)
        ax.text(c, y - 0.42, f"{c:.3f}", ha="center", fontsize=13, color=INK)
        ax.text(bnd, y + 0.36, f"경계 {bnd:.3f}", ha="center", fontsize=13, color=REG[r],
                fontweight="bold")
    ax.set_yticks(range(3), regions[::-1], fontsize=17, fontweight="bold")
    ax.set_xlim(0.32, 0.80); ax.set_ylim(-0.7, 2.7)
    ax.set_xlabel("낙엽기 NDVI 중앙값")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    # 핵심 대비: 순천 활엽 ≒ 홍천 침엽
    ax.annotate("", xy=(0.538, 2.0), xytext=(0.515, 0.0),
                arrowprops=dict(arrowstyle="<->", color=OVER, lw=2, ls="--",
                                connectionstyle="arc3,rad=-0.25"))
    ax.text(0.645, 1.6, "순천 활엽(0.515)\n≒ 홍천 침엽(0.538)", fontsize=14, color=OVER,
            fontweight="bold", va="center", linespacing=1.4)
    ax.scatter([], [], s=200, color=BRD, label="활엽수림")
    ax.scatter([], [], s=200, color=CON, label="침엽수림")
    ax.scatter([], [], s=140, marker="^", color=INK, label="같은 수종(소나무)")
    ax.legend(loc="upper left", frameon=False, fontsize=13, ncol=3, bbox_to_anchor=(0, 1.12))
    fig.text(0.5, -0.04, "지역 간 차이 0.204 > 클래스 간 차이 0.169   ·   대기보정 자료(S2_SR)에서도 남는 실제 분광 차이",
             ha="center", fontsize=13, color=MUTED)
    save(fig, "fig7_regional_offset")


# ═════════════════════════════════════════════════════════
# fig8. 낙엽송 — 침엽수인데 겨울엔 활엽수처럼 보인다
# ═════════════════════════════════════════════════════════
def fig8_larch():
    sp = ["잣나무", "소나무", "낙엽송", "신갈나무", "기타 활엽"]
    v = [0.685, 0.632, 0.369, 0.353, 0.341]
    c = [CON, CON, OVER, BRD, BRD]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    bars = ax.barh(sp[::-1], v[::-1], color=c[::-1], height=0.62)
    for r, val in zip(bars, v[::-1]):
        ax.text(val + 0.008, r.get_y() + r.get_height() / 2, f"{val:.3f}", va="center", fontsize=14)
    ax.set_xlim(0, 0.8); ax.set_xlabel("낙엽기 NDVI (홍천)")
    ax.set_title("수종별 낙엽기 NDVI", color=INK, pad=10)
    ax.text(0.45, 2.0, "← 침엽수(라벨 1)인데\n    활엽수와 구분 불가", fontsize=13.5, color=OVER,
            va="center", fontweight="bold")
    ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)

    ax = axes[1]
    tr = ["홍천", "대전", "순천"]
    mis = [37.8, 62.1, 96.2]
    bars = ax.bar(tr, mis, color=[REG[t] for t in tr], width=0.58)
    for r, val in zip(bars, mis):
        ax.text(r.get_x() + r.get_width() / 2, val + 2, f"{val:.1f}%", ha="center",
                fontsize=17, fontweight="bold")
    ax.set_ylim(0, 110); ax.set_ylabel("낙엽송 오분류율")
    ax.set_xlabel("학습 지역  (평가: 홍천)")
    ax.set_title("학습 지역에 낙엽송이 적을수록 더 틀린다", color=INK, pad=10, fontsize=15.5)
    ygrid(ax)
    fig.text(0.5, -0.09, "침엽수 중 낙엽송 비율: 홍천 33% · 대전 8.6% · 순천 0%",
             ha="center", fontsize=13, color=MUTED)
    save(fig, "fig8_larch")


# ═════════════════════════════════════════════════════════
# fig9. 해결 시도 — 입력 정규화는 과보정 (구 계수, 03-[16])
# ═════════════════════════════════════════════════════════
def fig9_normalization():
    combos = ["대전→홍천", "대전→순천", "홍천→대전", "홍천→순천", "순천→대전", "순천→홍천"]
    before = [1.59, -1.79, -1.11, -2.34, 2.12, 4.20]
    after = [-1.73, 1.20, 1.94, 2.98, -0.56, -2.03]
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for c, b, a in zip(combos, before, after):
        better = abs(a) < abs(b)
        col = CON if better else OVER
        ax.plot([0, 1], [b, a], color=col, lw=2.8, marker="o", ms=11)
        ax.text(-0.04, b, c, ha="right", va="center", fontsize=13.5)
        nudge = {-1.73: 0.13, -2.03: -0.13}.get(a, 0)
        ax.text(1.04, a + nudge, f"{a:+.2f}", ha="left", va="center", fontsize=13.5, color=col)
    ax.axhline(0, color=MUTED, lw=1.2, ls="--")
    ax.set_xticks([0, 1], ["보정 전", "지역별 정규화 후"], fontsize=15)
    ax.set_xlim(-0.55, 1.25)
    ax.set_ylabel("탄소 순오차 (%)")
    ygrid(ax)
    ax.set_title("6조합 전부 부호가 뒤집힘 — 평균 1.9배 과보정", color=INK, pad=12, fontsize=17)
    ax.plot([], [], color=CON, lw=3, label="개선 (3)")
    ax.plot([], [], color=OVER, lw=3, label="악화 (3)")
    ax.legend(frameon=False, loc="upper center", fontsize=13, ncol=2)
    fig.text(0.5, -0.03, "지역 중앙값을 빼면 '순천에 침엽이 많다'는 구성 정보까지 지워진다  (구 탄소계수 기준)",
             ha="center", fontsize=12, color=MUTED)
    save(fig, "fig9_normalization")


# ═════════════════════════════════════════════════════════
# fig10. 모델을 바꿔도 안 된다 / 라벨 없는 보정(AdaBN)은 된다
# ═════════════════════════════════════════════════════════
def fig10_models():
    models = ["Random Forest", "U-Net", "U-Net + AdaBN"]
    within = [(1.40, 0.01), (0.84, 0.12), (0.75, 0.16)]
    cross = [(3.19, 0.03), (3.25, 0.94), (1.68, 0.40)]
    x = np.arange(3); w = 0.36
    fig, ax = plt.subplots(figsize=(11, 5.6))
    b1 = ax.bar(x - w / 2, [m for m, _ in within], w, yerr=[s for _, s in within], color=MIX,
                capsize=6, label="지역 내", error_kw=dict(ecolor=INK, lw=1.5))
    b2 = ax.bar(x + w / 2, [m for m, _ in cross], w, yerr=[s for _, s in cross],
                color=[OVER, OVER, CON], capsize=6, label="지역 간", error_kw=dict(ecolor=INK, lw=1.5))
    for bars, d in [(b1, within), (b2, cross)]:
        for r, (m, s) in zip(bars, d):
            ax.text(r.get_x() + r.get_width() / 2, m + s + 0.1, f"{m:.2f}", ha="center",
                    fontsize=16, fontweight="bold")
    ax.set_xticks(x, models, fontsize=16)
    ax.set_ylabel("탄소 |순오차| (%)")
    ax.set_ylim(0, 4.8)
    ygrid(ax)
    ax.legend(frameon=False, fontsize=14, loc="upper left")
    ax.annotate("-48%", xy=(2 + w / 2, 2.25), xytext=(1 + w / 2 + 0.25, 4.35), fontsize=18,
                fontweight="bold", color=CON,
                arrowprops=dict(arrowstyle="-|>", color=CON, lw=2, connectionstyle="arc3,rad=-0.2"))
    ax.set_title("모델 구조를 바꿔도 쏠림은 그대로, 대상 지역 통계로 보정하면 절반",
                 color=INK, pad=12, fontsize=17)
    fig.text(0.5, -0.03, "평균 ± 표준편차 (시드 3개)   ·   AdaBN: 라벨 없이 평가 지역 영상으로 BatchNorm 통계만 다시 계산",
             ha="center", fontsize=12.5, color=MUTED)
    save(fig, "fig10_model_comparison")


# ═════════════════════════════════════════════════════════
# fig11. 무엇이 오차를 결정하는가 — 모델 용량 vs 학습 지역
# ═════════════════════════════════════════════════════════
def fig11_capacity():
    fig, ax = plt.subplots(figsize=(10, 3.4))
    items = ["모델 용량 2배\n(트리 80→150)", "학습 지역 교체\n(홍천→순천)"]
    vals = [0.33, 6.7]
    bars = ax.barh(items, vals, color=[MIX, OVER], height=0.55)
    for r, v in zip(bars, vals):
        ax.text(v + 0.12, r.get_y() + r.get_height() / 2, f"{v}%p", va="center", fontsize=19,
                fontweight="bold")
    ax.set_xlim(0, 8); ax.invert_yaxis()
    ax.set_xlabel("탄소 오차 변화량")
    ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.set_title("약 20배 — 모델이 아니라 학습 데이터가 결정한다", color=INK, pad=10, fontsize=17)
    save(fig, "fig11_capacity_vs_region")


if __name__ == "__main__":
    fig1_pipeline()
    fig2_seasonal()
    fig3_protocol()
    fig4_heatmaps()
    fig5_within_vs_cross()
    fig6_f1_vs_carbon()
    fig7_offset()
    fig8_larch()
    fig9_normalization()
    fig10_models()
    fig11_capacity()
