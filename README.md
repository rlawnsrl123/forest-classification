# forest-classification

위성영상(Sentinel-2) 기반 산림 유형 분류가 **LCA 탄소 인벤토리로 쓸 만한지** 검증하는 졸업프로젝트.
핵심 질문: *F1이 좋은 분류기가 탄소 총량도 맞히는가, 지역이 바뀌어도 맞히는가.*

## 구조

```
files/
  01_explore.ipynb      대전 탐색 (아카이브)
  02_regions.ipynb      지역 데이터 생성 — GEE export, 라벨 래스터화, 정합 검증
  03_experiments.ipynb  전이 실험 (주의: 일부 셀이 구 탄소계수로 덮어씀)
  04_correction.ipynb   α 보정 실험 (쌍별 설계 — 맨 위 경고 셀 참조)
unet/                   U-Net · AdaBN · RF 재현 코드 (동료)
docs/
  NOTES.md              Colab / GEE 조작 레퍼런스
  team/                 팀 공유 문서 · 보고서 초안
```

데이터와 모델(`*.tif`, `*.npz`, `*.npy`, `*.pkl`, shp)은 저장소에 없다.
Google Drive `MyDrive/forest/` 에 있다 (`data/`, `outputs/`).

## 고정값

- 탄소계수 `C = {1: 79.25, 2: 101.02, 3: 92.01}` tC/ha (침엽 / 활엽 / 혼효)
- 라벨: 0 배경 / 1 침엽 / 2 활엽 / 3 혼효 / 255 산림 밖
- 유효 픽셀: `(label >= 1) & (label <= 3) & (year >= 2024)` — **연도 조건만으로는 배경이 섞인다**
