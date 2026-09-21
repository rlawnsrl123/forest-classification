# U-Net / 전이 강건성 실험 — 5090 작업 브랜치

산림 유형 분류 프로젝트의 **공간 모델 가지**. 준기 님의 RF·α 줄기와 같은 문제를
다른 처방으로 맡는다.

> **질문**: 지역 간 분광 오프셋이 만드는 탄소 오차 쏠림이, 공간 모델(U-Net)과
> 사전학습 인코더에서도 그대로 나타나는가?
>
> - 그대로면 → "모델 바꿔서 풀 문제가 아니라 보정으로 풀 문제" = α의 명분이 강해짐
> - 줄어들면 → α와 결합하는 얘기가 됨
>
> 어느 쪽이든 결과다. 그리고 `보고서_초고.md` 7절이 "공간 모델에서 동일 구조가
> 나타나는지는 후속 과제"라고 명시해 둔 **그 공백이 정확히 이것**이다.

---

## 데이터 (2026-09-21 수령 완료)

| 위치 | 내용 |
|---|---|
| `../../data/s2/{region}_s2_20band.tif` | 20밴드 원본. 대전 float32 / 홍천·순천 int16(×10000) |
| `../../data/feats/{region}_feats38.npy` | 38피처 float16. `build_feats38.py` 로 생성 |
| `../../data/feats/{region}_{label,year,species}.tif` | 라벨·갱신년도·수종 |
| `C:/Users/aigre/OneDrive/Desktop/data/daejeon_feats38.npy` | 준기 님 제공 원본 — **검증 기준** |

**격자**: 대전 3506×2813 / 홍천 4420×9309 / 순천 3924×3758 (label tif와 일치 확인)

### 38피처 생성 규칙 [역추적 확정]

대전만 20밴드 tif와 준기 님 feats38 을 둘 다 갖고 있어서, 후보를 계산해 맞춰
규칙을 복원했다 (`verify_feats.py`). **38개 전부 최대 절대차 0.00025** = float16 해상도.

```
0~19   원 밴드 (여름 10 + 낙엽기 10, 각 B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12)
20~25  s_ndvi, w_ndvi, s_ndmi, w_ndmi, dNDVI(여름−낙엽기), dNDMI(여름−낙엽기)
26~37  (s_ndvi, w_ndvi, dNDVI) × 윈도우(5,15) × (평균, 표준편차)
```

`W_NDVI = 21` 확정. 문서가 판별력 상위로 꼽은 [32]·[36]은 각각
`w_ndvi_w15_mean`, `dNDVI_w15_mean` 으로, "[36]은 부호 반대"라는 기술과도 맞는다
(dNDVI = 여름 − 낙엽기이므로 낙엽기 NDVI가 커질수록 작아진다).

세 지역을 **같은 코드 경로**로 다시 만들어 쓴다. 그래야 9칸 비교가 성립한다.

---

## 실행 순서

`python` 은 `../../.venv/Scripts/python.exe` 를 가리킨다(레포 밖, 커밋 대상 아님).

```bash
# 0. 환경 점검 (sm_120 커널까지 실제 연산으로 확인)
python env_check.py

# 1. 38피처 생성 — 세 지역을 같은 코드 경로로. 대전은 제공본과 대조
python build_feats38.py --s2-dir ../../data/s2 --out-dir ../../data/feats \
    --check-against "<준기님 daejeon_feats38.npy>"
python prepare_data.py --data-dir ../../data/feats --cache-dir ./cache --source feats38

# 2. RF 9칸 (대조군)
python rf_9cell.py --cache-dir ./cache --seeds 3 --out rf_9cell.json

# 3. U-Net 9칸 — 지역별 학습 후 평가
for R in daejeon hongcheon suncheon; do
  python train_unet.py --region $R --cache-dir ./cache --out ./runs \
    --base 16 --depth 3 --stride 32 --min-valid 0.10 --wd 1e-3 \
    --steps 4000 --val-col-frac 0.25 --val-every 250 --seed 42 --tag-suffix _B
done
python eval_9cell.py --cache-dir ./cache --runs ./runs --seed 42 --tag-suffix _B \
    --out unet_9cell.json

# 4. AdaBN — 딥러닝판 α (라벨 없이 대상 지역 통계로 BN 재추정)
python eval_9cell.py --cache-dir ./cache --runs ./runs --seed 42 --tag-suffix _B \
    --adabn --out unet_9cell_adabn_s42.json

# 5. 집계
python summarize.py --rf rf_9cell.json --unet unet_9cell*.json --adabn unet_9cell_adabn_*.json

# (선택) 사전학습 인코더 비교 — 다음 단계
python train_unet.py --region daejeon --arch smp:resnet34+pre --cache-dir ./cache --out ./runs
python eval_9cell.py --arch smp:resnet34+pre --cache-dir ./cache --runs ./runs --out unet_pre_9cell.json
```

### 데이터 없이 파이프라인만 검증할 때

`make_synthetic.py` 가 문서의 지역별 NDVI 경계·클래스 비율·낙엽송 노출도·dtype 함정을
심은 가짜 데이터를 만든다. 전 구간 스모크 테스트용이며 **여기서 나온 숫자는 결과가 아니다.**

```bash
python make_synthetic.py --out ./synth
python prepare_data.py --data-dir ./synth --cache-dir ./cache_synth --source s2
python train_unet.py --region daejeon --cache-dir ./cache_synth --out ./runs_synth --steps 300
python eval_9cell.py --cache-dir ./cache_synth --runs ./runs_synth --out synth_9cell.json
```

---

## 프로토콜 — 절대 건드리지 말 것

`common.py`가 단일 출처다. 이 값이 흔들리면 준기 님 9칸 표와 비교가 불가능해진다.

| 항목 | 값 | 근거 |
|---|---|---|
| 탄소계수 | 침 79.25 / 활 101.02 / 혼 92.01 tC/ha | 강원 NFI 2013, GIR 공표 [확정] |
| 라벨 | `FRTP_CD` 1·2·3만. 죽림(4)·비산림(0) 제외 | 표본 부족 |
| 갱신년도 | **2024+ 만** | 2020+로 낮추면 지역별 노후화 차이가 섞임 |
| 공간 분할 | 500px 타일 / 동서 블록 | 픽셀 랜덤 분할은 인접 누수로 성능 과대평가 |
| 주지표 | macro F1, 탄소 순오차·총오차 | F1만으로는 탄소 오차를 예측 못 함 |

### 평가셋을 동서 분할로 바꾼 이유

기존 RF 표는 **대각(지역 내)은 홀드아웃 30% 타일, 비대각(지역 간)은 지역 전체**에서
평가했다. 두 평가셋의 클래스 비율이 달라서 "지역 내 이득"의 일부가 모델 효과가
아니라 평가셋 구성 차이일 수 있다(문서 6-3절이 직접 지적).

여기서는 **모든 지역에서 동쪽 절반만 평가**한다. 학습 패치는 서쪽 안에 완전히
들어가야만 하도록 강제해서 경계 누수도 없다. `rf_reproduce.py --protocol ew`로
RF도 같은 프로토콜로 돌릴 수 있으므로, **RF와 U-Net을 같은 자 위에서** 비교한다.

---

## 코드 함정 (문서 §9에서 가져옴)

- **대전만 float32**, 홍천·순천은 int16(×10000) → `common.DTYPE_SCALE`이 처리
- `FRTP_CD`는 원본 shp에서 VARCHAR (`'1'` 문자열). 래스터화된 tif에서는 정수
- 낙엽송은 `sp == 13`. **라벨이 아니라 수종** — 라벨은 침엽(1)
- 컬럼명 SHP 10자 절단: `KOFTR_GROUP_CD` → `KOFTR_GROU`
- shapefile은 `.shp .dbf .shx .prj` 한 세트. `.dbf` 없으면 속성 전멸

## 환경

5090은 Blackwell(sm_120)이라 CUDA 12.8 미만 빌드는 커널이 없어 못 돈다.
여기서는 **PyTorch cu130** 휠을 썼다 (드라이버 591.55 = CUDA 13.1).

```
py -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
.venv/Scripts/python -m pip install numpy rasterio scikit-learn segmentation-models-pytorch
```

## 파일

| 파일 | 역할 |
|---|---|
| `common.py` | 계수·프로토콜 상수, 탄소 오차·macro F1 계산 |
| `prepare_data.py` | tif → memmap 캐시 + 유효 마스크 + 동서 분할 경계 |
| `data.py` | 패치 샘플러(적분영상으로 유효 패치만), 밴드 통계, 클래스 가중치 |
| `model.py` | U-Net(자체) / smp 백본, `reset_bn`(AdaBN 1단계) |
| `train_unet.py` | 한 지역 학습 (서쪽만) |
| `eval_9cell.py` | 9칸 표 + AdaBN |
| `rf_reproduce.py` | RF 9칸 재현 · 교락 해소판 · 시드 반복 (npz만 필요) |
| `make_synthetic.py` | 합성 데이터 — 데이터 오기 전 파이프라인 검증용 |
| `env_check.py` | GPU/커널/패키지 점검 |
