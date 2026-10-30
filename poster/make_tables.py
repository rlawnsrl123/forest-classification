"""포스터용 표를 편집 가능한 PowerPoint 표로 생성 (이미지 아님 → 인쇄 시 글자 안 깨짐).

출력: poster/out/poster_tables.pptx — 슬라이드 1장에 표 1개. 복사해서 포스터에 붙여넣기.
실행: py poster/make_tables.py
"""
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

FONT = "맑은 고딕"
HEAD = RGBColor(0x3D, 0x6B, 0x7A)   # 템플릿 헤더 톤
ALT = RGBColor(0xF1, 0xF4, 0xF6)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
INK = RGBColor(0x2B, 0x34, 0x40)
OVER = RGBColor(0xC8, 0x55, 0x3D)
UNDER = RGBColor(0x3A, 0x7C, 0xA5)
GOOD = RGBColor(0x2D, 0x6A, 0x4F)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(35.4375), Inches(47.25)   # 템플릿과 동일
BLANK = prs.slide_layouts[6]
W = Inches(15.5)                                                    # 템플릿 칸 폭(약 17") 안


def table(title, rows, col_w=None, font=26, highlight=None, note=None):
    """rows[0]이 머리글. highlight: {(r,c): RGBColor} 글자색 강조(굵게)."""
    s = prs.slides.add_slide(BLANK)
    tb = s.shapes.add_textbox(Inches(1), Inches(0.6), W, Inches(1))
    tb.text_frame.text = title
    r = tb.text_frame.paragraphs[0].runs[0]
    r.font.size, r.font.bold, r.font.name, r.font.color.rgb = Pt(32), True, FONT, INK

    nr, nc = len(rows), len(rows[0])
    rh = Inches(0.85)
    shp = s.shapes.add_table(nr, nc, Inches(1), Inches(1.9), W, rh * nr)
    t = shp.table
    if col_w:
        tot = sum(col_w)
        for i, w in enumerate(col_w):
            t.columns[i].width = int(W * w / tot)
    for i in range(nr):
        t.rows[i].height = rh
        for j in range(nc):
            c = t.cell(i, j)
            c.text = str(rows[i][j])
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            c.fill.fore_color.rgb = HEAD if i == 0 else (ALT if i % 2 == 0 else WHITE)
            p = c.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 and i > 0 else PP_ALIGN.CENTER
            run = p.runs[0] if p.runs else p.add_run()
            run.font.name, run.font.size = FONT, Pt(font)
            run.font.color.rgb = WHITE if i == 0 else INK
            run.font.bold = i == 0 or j == 0
            if highlight and (i, j) in highlight:
                run.font.color.rgb, run.font.bold = highlight[(i, j)], True
    if note:
        nb = s.shapes.add_textbox(Inches(1), Inches(1.9) + rh * nr + Inches(0.2), W, Inches(0.8))
        nb.text_frame.text = note
        nb.text_frame.word_wrap = True
        rr = nb.text_frame.paragraphs[0].runs[0]
        rr.font.size, rr.font.name, rr.font.color.rgb = Pt(18), FONT, RGBColor(0x7A, 0x85, 0x91)


# T1. 세 지역 — 대비되는 임상 구성
table("세 연구 지역", [
    ["", "대전", "홍천", "순천"],
    ["성격", "도시 파편화", "연속 산림", "상록 침엽 조림"],
    ["침엽 / 활엽 / 혼효", "33 / 46 / 17%", "36 / 52 / 11%", "48 / 38 / 10%"],
    ["우점 수종", "리기다소나무", "낙엽송, 잣나무", "편백, 곰솔"],
    ["침엽 중 낙엽송", "8.6%", "33%", "0%"],
    ["라벨 픽셀 (2024+ 갱신)", "103만", "474만", "187만"],
], col_w=[1.5, 1, 1, 1], highlight={(4, 2): OVER},
    note="세 지역은 무작위 표본이 아니라 임상 구성이 대비되도록 고른 대조 사례")

# T2. 탄소계수
table("임상별 탄소계수 (확정)", [
    ["", "침엽수림", "활엽수림", "혼효림"],
    ["탄소저장량 (tC/ha)", "79.25", "101.02", "92.01"],
], col_w=[1.6, 1, 1, 1],
    note="출처: 강원 NFI 2013 / 온실가스종합정보센터 공표값 · 혼효림은 국가 인벤토리 규약(50:50)")

# T3. 9조합 (RF, 시드 3개 평균)
rf = json.loads((ROOT / "unet/rf_9cell.json").read_text(encoding="utf-8"))["avg"]
K = ["dae", "hon", "sun"]
N = {"dae": "대전", "hon": "홍천", "sun": "순천"}
rows = [["학습 → 평가", "macro F1", "탄소 순오차", "탄소 총오차"]]
hl = {}
order = [(k, k) for k in K] + [(t, e) for t in K for e in K if t != e]
for i, (t, e) in enumerate(order, start=1):
    d = rf[f"{t}_{e}"]
    rows.append([f"{N[t]} → {N[e]}" + ("  (지역 내)" if t == e else ""),
                 f"{d['macro_f1']:.3f}", f"{d['carbon_net']:+.2f}%", f"{d['carbon_gross']:.2f}%"])
    if t != e and abs(d["carbon_net"]) > 3:
        hl[(i, 2)] = OVER if d["carbon_net"] > 0 else UNDER
table("지역 간 전이 9조합 — Random Forest", rows, col_w=[1.8, 1, 1.1, 1.1], font=24, highlight=hl,
      note="순오차 = 지역 탄소총량이 틀린 정도(+ 과대 / − 과소) · 총오차 = 틀린 픽셀의 양 · 시드 3개 평균")

# T4. 모델 비교
table("모델별 탄소 |순오차| (시드 3개, 평균 ± 표준편차)", [
    ["", "지역 내", "지역 간", "지역 간 F1"],
    ["Random Forest", "1.40 ± 0.01%", "3.19 ± 0.03%", "0.536"],
    ["U-Net", "0.84 ± 0.12%", "3.25 ± 0.94%", "0.468"],
    ["U-Net + AdaBN", "0.75 ± 0.16%", "1.68 ± 0.40%", "0.520"],
], col_w=[1.5, 1, 1, 0.9], highlight={(2, 2): OVER, (3, 2): GOOD},
    note="AdaBN: 라벨 없이 평가 지역 영상으로 BatchNorm 통계만 다시 계산 (가중치는 그대로)")

# T5. 원인 후보 검증 요약
table("원인 가설 검증", [
    ["가설", "실험", "결과"],
    ["모델 용량 부족", "트리 80→150", "기각 (0.33%p)"],
    ["모델 구조 (공간 정보)", "RF → U-Net", "기각 (3.19 → 3.25%)"],
    ["침엽 수종 다양성", "수종 1→4종", "기각 (0.014%p)"],
    ["낙엽송 노출", "비율 0→33% 통제", "부분 (1.04%p)"],
    ["지역 분광 오프셋", "경계선 계산", "지지 (부호 6/6, r=0.93)"],
], col_w=[1.4, 1.2, 1.4], font=24,
    highlight={(5, 2): GOOD, (4, 2): RGBColor(0xB0, 0x7A, 0x1E)})

prs.save(OUT / "poster_tables.pptx")
print("saved poster_tables.pptx")
