"""공통 상수·유틸 — RF 실험과 프로토콜을 정확히 맞추기 위한 단일 출처.

근거 문서: 프로젝트_인수인계.md §2·§9, 팀공유_중간정리.md §3-3
이 파일의 값을 바꾸면 U-Net 결과가 준기 님 9칸 표와 비교 불가능해진다.
"""
import sys

import numpy as np

# Windows 콘솔 기본 코드페이지(cp949)에서 한글 출력이 깨지는 것을 막는다.
# 모든 스크립트가 common을 import 하므로 여기 한 번이면 된다.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── 탄소계수 [확정] 강원 NFI 2013 / GIR 공표 (tC/ha) ───────────────────────
# 임시값 55/65/60 과 혼동 금지. 활엽−침엽 = +21.77
C = {1: 79.25, 2: 101.02, 3: 92.01}

# ── 라벨 FRTP_CD ──────────────────────────────────────────────────────────
# '1' 침엽 / '2' 활엽 / '3' 혼효 / '4' 죽림(표본 부족, 제외) / '0' 비산림
# 주의: shp 원본에서는 VARCHAR. 래스터화된 tif에서는 정수.
CLASSES = (1, 2, 3)
NAMES = {1: "침엽", 2: "활엽", 3: "혼효"}
IGNORE = 255              # CrossEntropyLoss ignore_index

# ── 프로토콜 ──────────────────────────────────────────────────────────────
TILE = 500                # 공간 분할 단위(px). 픽셀 랜덤 분할 금지
YEAR_MIN = 2024           # 갱신년도 2024+ 만 사용 (2020+로 낮추면 α 추정 오염)
N_TRAIN_PX = 200_000      # RF 기준 학습 픽셀 수
N_EVAL_PX = 150_000       # RF 기준 평가 픽셀 수
SEED = 42

REGIONS = ("daejeon", "hongcheon", "suncheon")
ABBR = {"daejeon": "dae", "hongcheon": "hon", "suncheon": "sun"}

# ── 저장 형식 함정 (문서 §9) ──────────────────────────────────────────────
# 대전만 float32(반사도 그대로), 홍천·순천은 int16(×10000) → 읽을 때 /10000
DTYPE_SCALE = {"daejeon": 1.0, "hongcheon": 1e-4, "suncheon": 1e-4}

# 낙엽송 수종 코드 (라벨 아님. 분석·조작용)
SP_LARCH = 13

N_BANDS = 20              # 여름 10 + 낙엽기 10

# ── 밴드 순서 [준기 님 확인] ──────────────────────────────────────────────
# 1~10 = 6~9월(여름), 11~20 = 11~2월(낙엽기). 각각 B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12
# 0-based 인덱스로: 여름 RED=2(B4) NIR=6(B8) SWIR=8(B11) / 낙엽기 +10
RED_S, NIR_S, SWIR_S = 2, 6, 8
RED_W, NIR_W, SWIR_W = 12, 16, 18

# ── 38피처 배열 ───────────────────────────────────────────────────────────
#   0~19  원 밴드 (위 순서)
#   20~25 지수  s_ndvi, w_ndvi, s_ndmi, w_ndmi, dNDVI, dNDMI
#   26~37 공간 컨텍스트 (s/w/d ndvi × 윈도우 5,15 × 평균,표준편차)
# 실측 검증: 피처[21] 중앙값 대전 침엽 0.656 / 활엽 0.440 = 문서 §3-4의 0.655 / 0.440
F_S_NDVI, F_W_NDVI = 20, 21
N_FEATS = 38


def to_reflectance(arr, region):
    """지역별 저장 형식을 반사도(0~1 스케일)로 통일."""
    s = DTYPE_SCALE[region]
    return arr.astype(np.float32) * s if s != 1.0 else arr.astype(np.float32)


def tile_id(row, col):
    """RF와 동일한 타일 ID. (row//500)*10000 + (col//500). 크게 묶기만 가능."""
    return (row // TILE) * 10000 + (col // TILE)


def carbon_err(y_true, y_pred):
    """탄소축적량 총량 오차(%).

    (Σ 예측픽셀수×계수 − Σ 실제픽셀수×계수) / Σ 실제픽셀수×계수 × 100
    픽셀 면적이 동일하므로 면적 환산 없이 픽셀 수로 계산 — RF 코드와 동일.
    """
    t = sum((y_true == v).sum() * C[v] for v in CLASSES)
    p = sum((y_pred == v).sum() * C[v] for v in CLASSES)
    return (p - t) * 100.0 / t


def carbon_err_from_cm(cm):
    """혼동행렬(3x3, 행=실제 1/2/3, 열=예측 1/2/3)에서 순오차·총오차를 계산.

    순오차 = 실제 탄소 오차(상쇄 후). 총오차 = 오분류 절대량(상쇄 전).
    문서 §3-2의 '총오차는 1.25배인데 순오차는 4.6배'가 이 두 값의 관계.
    """
    cm = np.asarray(cm, dtype=np.float64)
    coef = np.array([C[v] for v in CLASSES])
    true_tot = (cm.sum(axis=1) * coef).sum()
    net = ((cm.sum(axis=0) - cm.sum(axis=1)) * coef).sum()
    gross = sum(
        cm[i, j] * abs(coef[j] - coef[i])
        for i in range(3) for j in range(3) if i != j
    )
    return net * 100.0 / true_tot, gross * 100.0 / true_tot


def macro_f1_from_cm(cm):
    """3클래스 macro F1. sklearn 없이 혼동행렬에서 직접."""
    cm = np.asarray(cm, dtype=np.float64)
    f1s = []
    for i in range(len(cm)):
        tp = cm[i, i]
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        rec = tp / cm[i, :].sum() if cm[i, :].sum() else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s)), [float(x) for x in f1s]


def confusion(y_true, y_pred):
    """3x3 혼동행렬 (1/2/3 → 0/1/2 인덱스)."""
    cm = np.zeros((3, 3), dtype=np.int64)
    m = np.isin(y_true, CLASSES) & np.isin(y_pred, CLASSES)
    np.add.at(cm, (y_true[m] - 1, y_pred[m] - 1), 1)
    return cm
